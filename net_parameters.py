'''shared classes for ice.py and test_loader.py'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class IceNet(nn.Module):
    def __init__(self, channels):
        super(IceNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)           # pool is not a layer
        # inputs, num filters, kernel size   (batch size not described!)
        self.conv1a = nn.Conv2d(channels, 16, 3)  # 60 > 58
        self.conv1b = nn.Conv2d(16, 32, 3)        # > 56 > 28
        self.drop0 = nn.Dropout2d()               # NC
        self.conv2a = nn.Conv2d(32, 32, 3)         # 28 -> 26
        self.conv2b = nn.Conv2d(32, 64, 3)         # 26 -> 24 ->12
        self.drop1 = nn.Dropout2d()               # NC
        self.conv3a = nn.Conv2d(64, 64, 3)         # 12 -> 10 -> 5
        self.conv3b = nn.Conv2d(64, 128, 3)         # 12 -> 10 -> 5
        self.drop2 = nn.Dropout2d()

        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.drop_fc = nn.Dropout()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.best_xval_loss = 10
        self.training = True

    def forward(self, x, v=0):
        x = F.relu(self.conv1a(x))
        if v:
            print(0, x.size())
        x = self.pool(F.relu(self.conv1b(x)))
        if v:
            print(1, x.size())
        x = F.dropout(x, training=self.training)
        if v:
            print(2, x.size())
        x = F.relu(self.conv2a(x))
        x = self.pool(F.relu(self.conv2b(x)))
        x = F.dropout(x, training=self.training)
        if v:
            print(3, x.size())
        x = F.relu(self.conv3a(x))
        x = self.pool(F.relu(self.conv3b(x)))
        if v:
            print(4, x.size())
        x = F.dropout(x, training=self.training)
        if v:
            print(5, x.size())
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        if v:
            print(6, x.size())
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class IcebergDataset(Dataset):
    '''dataset containing icebergs for Kaggle comp
    internally, df, X, and y are indexed by i as 0...n-1
    '''
    def __init__(self, _X, _y, _df, transform=None, training=True):
        # generate difference channel
        x0 = _X[:, 0]
        x1 = _X[:, 1]
        x2 = np.subtract(x0, x1)
        x2 = np.subtract(x2, np.mean(x2))
        x2 = np.divide(x2, np.max(x2))
        self.X = np.stack([x0, x1, x2], axis=1)
        print('dataset init:', self.X.min(), self.X.max(), self.X.shape)

        self.transform = transform
        self.training = training

        if self.training:
            self.y = _y
            assert len(self.X) == len(self.y)
        self.df = _df
        assert len(self.X) == len(self.df)
        self.n = len(self.X)
        self.latest_pred = {}
        self.latest_prob = {}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if not self.training:  # for test set:
            x = self.X[i]
            x = torch.from_numpy(x).float()
            x = self.transform(x)
            return x, self.df.index[i]  # data, id

        x, y = self.X[i], self.y[i].reshape(1, 1)
        x, y = torch.from_numpy(x).float(), y

        if self.transform:
            x = self.transform(x)

        return x, y, self.df.index[i]   # iloc to avoid non-0 index

    def show(self, n, rando=True, show_coords=False):
        # show approximately n samples and some data about them
        rows = int(n ** .3)
        cols = int(n // rows)
        channels = 3
        fig, axes = plt.subplots(channels*rows, cols)
        for row in range(rows):
            for col in range(cols):
                # pull an image
                if rando:
                    index = np.random.randint(0, self.n)
                else:
                    index = row * cols + col

                # get some data from df
                if self.training:
                    arr, label, ID = self.__getitem__(index)
                else:
                    arr, ID = self.__getitem__(index)
                    label = 'test'

                p = self.latest_pred.get(ID)
                prob = self.latest_prob.get(ID)
                l = label[0][0]
                correct = p == l
                arr = arr.numpy()
                # angle = str(self.df.angle.iloc[index])
                coords = [self.df.band_1_x[index],
                          self.df.band_1_y[index],
                          self.df.band_2_x[index],
                          self.df.band_2_y[index]]
                img = np.multiply(np.add(arr, 1), 127.5)
                axes[row*channels, col].set_title(str(l)+', p=' + str(p))
                axes[row*channels+1, col].set_title(str(prob))
                # axes[row*channels+1, col].set_title(angle)
                axes[row*channels+2, col].set_title(' '.join(
                    [str(c) for c in coords]))

                for ch in range(channels):
                    # Image.fromarray(img[ch]).show()   # for PIL
                    loc = ch + row*channels, col
                    axes[loc].axis('off')
                    axes[loc].title.set_fontsize(8)
                    if ch < img.shape[0]:
                        # print image if applicable
                        axes[loc].imshow(Image.fromarray(img[ch]))

                        if ch == 0:
                            if not correct:
                                axes[loc].plot(2, 2, 'rx')

                        # add dot for highlight coords ?
                        if show_coords:
                            if ch == 0:
                                axes[loc].plot(coords[0], coords[1], 'rx')
                            elif ch == 1:
                                axes[loc].plot(coords[2], coords[3], 'bx')

        plt.show()
