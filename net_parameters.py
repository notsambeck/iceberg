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
        # inputs, num filters, kernel size
        self.conv1 = nn.Conv2d(channels, 32, 5)
        self.drop0 = nn.Dropout2d()
        self.conv2 = nn.Conv2d(32, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d()
        self.conv3 = nn.Conv2d(16, 16, 5)
        self.drop2 = nn.Dropout2d()

        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.drop_fc = nn.Dropout()
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)
        self.best_xval_loss = 10
        self.training = True

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.dropout(x, training=self.training)
        x = self.pool(F.relu(self.conv2(x)))
        x = F.dropout(x, training=self.training)
        x = self.pool(F.relu(self.conv3(x)))
        x = F.dropout(x, training=self.training)
        # print(x.size())
        x = x.view(-1, 16 * 5 * 5)
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
        # self.df = df
        self.X = _X
        self.training = training
        if self.training:
            self.y = _y
            assert len(self.X) == len(self.y)
        self.df = _df
        assert len(self.X) == len(self.df)
        self.transform = transform
        self.n = len(self.X)
        global _last_pred
        global _last_prob

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

                p = _last_pred.get(ID)
                prob = _last_prob.get(ID)
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
