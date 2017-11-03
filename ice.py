import os

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # noqa

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
# import torchvision.transforms as transforms

# LOAD TRAINING DATA

imdata = pd.read_json('data/train.json')

for b in 'band_1', 'band_2':
    imdata[b] = imdata[b].apply(lambda x: np.array(x).reshape(75, 75))

x1 = np.stack(imdata.band_1)
x2 = np.stack(imdata.band_2)
minimum = np.min([x1, x2])
maximum = np.max([x1, x2])
difference = (maximum - minimum)


# IMAGE NORMS / STATS

def normalize(arr):
    return np.divide(np.subtract(arr, minimum), difference)


def make_stats_frame():
    df = pd.DataFrame()

    for b in 'band_1', 'band_2':
        df[b] = imdata[b].apply(normalize)
        df[b + '_max'] = imdata[b].apply(np.max)
        df[b + '_min'] = imdata[b].apply(np.min)
        df[b + '_mean'] = imdata[b].apply(np.mean)
        df[b + '_median'] = imdata[b].apply(np.median)
        df[b + '_coords'] = imdata[b].apply(
            lambda x: np.unravel_index(np.argmax(x), x.shape))
        df[b + '_x'] = df[b + '_coords'].apply(lambda x: x[0])
        df[b + '_y'] = df[b + '_coords'].apply(lambda x: x[1])
        df['is_iceberg'] = imdata.is_iceberg
        df['inc_angle'] = imdata.inc_angle

    return df


df_path = 'data/stat_frame.csv'
if os.path.exists(df_path):
    df = pd.read_csv(df_path, index_col=0)
else:
    df = make_stats_frame()
    df.to_csv(df_path)

y = df.is_iceberg.values

'''
note that 100% of images w/ inc_angle == 'na' are class 0

logistic regression - baseline predictions:

from sklearn.linear_model import LogisticRegression
cs = ['band_1_max', 'band_1_min', 'band_1_median', 'band_1_mean',
      'band_2_max', 'band_2_min', 'band_2_median', 'band_2_mean']
X = df[cs].values
lr = LogisticRegression()

lr.fit(X, y)
print(lr.score(X, y))


does it make sense to use an interolated inc_angle? no.

logistic regression with mean fill on inc_angle: 0.6963840399
logistic regression without inc_angle: 0.6963840399
logistic regression with LinearRegression fill: 0.695137157107
'''

# prep image data for neural net
# rescale to +/- 1

X = normalize(np.stack([x1, x2, np.subtract(x2, x1)], axis=1))
# X = normalize(np.stack([x1, x2], axis=1))
X = np.add(np.multiply(X, 2), -1)

# >>> X.shape
# (1604, 3, 75, 75)
y.reshape(-1, 1)

split = 1152

X_train = X[:split]
y_train = y[:split]
df_train = df.iloc[:split]
X_test = X[split:]
y_test = y[split:]
df_test = df.iloc[split:]


# dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor  # Uncomment this to run on GPU


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, target = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        return torch.from_numpy(image), torch.from_numpy(target)


class IcebergDataset(torch.utils.data.Dataset):
    '''dataset containing icebergs for Kaggle comp'''
    def __init__(self, X, y, df, transform=None):
        # self.df = df
        self.X = X
        self.y = y
        self.df = df
        self.transform = transform
        assert len(self.X) == len(self.y)  # == len(self.df)
        self.n = len(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        sample = self.X[i], self.y[i].reshape(1, 1)

        if self.transform:
            self.transform(sample)

        return sample

    def show(self, n, rows=2, rando=True):
        # show some samples
        channels = self.X.shape[1]
        fig, axes = plt.subplots(channels*rows, n)
        for row in range(rows):
            for i in range(n):
                # pull an image
                if rando:
                    index = np.random.randint(0, self.n)
                else:
                    index = i

                # get some data from df
                arr, label = self.__getitem__(index)
                angle = str(self.df.inc_angle.iloc[index])
                crds = ' '.join([str(self.df.band_1_x[index]),
                                 str(self.df.band_1_y[index]),
                                 str(self.df.band_2_x[index]),
                                 str(self.df.band_2_y[index])])
                img = np.multiply(np.add(arr, 1), 127.5)
                axes[row*channels, i].set_title(str(label)+str(index))
                axes[row*channels+1, i].set_title(angle)
                axes[row*channels+2, i].set_title(crds)

                for ch in range(channels):
                    # Image.fromarray(img[ch]).show()   # for PIL
                    loc = ch + row*channels, i
                    axes[loc].imshow(Image.fromarray(img[ch]))
                    axes[loc].axis('off')
                    axes[loc].title.set_fontsize(6)
        plt.show()


train_set = IcebergDataset(X_train, y_train, df_train, transform=ToTensor())
test_set = IcebergDataset(X_test, y_test, df_test, transform=ToTensor())

# dataset.show(5)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                           shuffle=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=32,
                                          shuffle=True, num_workers=4)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)

        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 5 * 5)
        # print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.float()


# criterion = nn.LogSoftmax()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


def train(n):

    for epoch in range(n):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            image, target = data

            image = image.float()
            target = target.view(-1)
            # print('target:', target.size(), 'image', image.size())

            # wrap them in Variable
            image, target = Variable(image), Variable(target)

            # print(type(image), len(image), type(image[0]))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(image)
            # print(outputs.size())
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i == 35:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 35))

        val_loss = 0.0

        if epoch % 10 == 9:
            print('\n## validation ## ')
            correct, total = 0, 0
            for i, data in enumerate(test_loader, 0):
                # get the inputs
                image, target = data

                image = image.float()
                target = target.view(-1)
                # print('target:', target.size(), 'image', image.size())

                # wrap them in Variable
                image, target = Variable(image), Variable(target)
                outputs = net(image)
                loss = criterion(outputs, target)
                _, preds = torch.max(outputs, 1)
                # print(preds, target)
                # return preds, target
                correct += (preds == target).sum().float()
                total += len(preds)

                # print statistics
                val_loss += loss.data[0]

            print('[%d, %5d] validation loss: %.3f' %
                  (epoch + 1, i + 1, val_loss / 14))
            print('accuracy:', correct[0], 'of', total)

    print('Finished Training')
