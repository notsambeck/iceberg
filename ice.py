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
import torchvision.transforms as transforms

# LOAD TRAINING DATA

df = pd.read_json('data/train.json')

for b in 'band_1', 'band_2':
    df[b] = df[b].apply(lambda x: np.array(x).reshape(75, 75))

x1 = np.stack(df.band_1)
x2 = np.stack(df.band_2)
minimum = np.min([x1, x2])
maximum = np.max([x1, x2])
difference = (maximum - minimum)


# IMAGE NORMS / STATS

def normalize(arr):
    return np.divide(np.subtract(arr, minimum), difference)


for b in 'band_1', 'band_2':
    df[b] = df[b].apply(normalize)
    df[b + '_max'] = df[b].apply(np.max)
    df[b + '_min'] = df[b].apply(np.min)
    df[b + '_mean'] = df[b].apply(np.mean)
    df[b + '_median'] = df[b].apply(np.median)


y = df.is_iceberg.values

# logistic regression - baseline predictions:

'''
from sklearn.linear_model import LogisticRegression

cs = ['band_1_max', 'band_1_min', 'band_1_median', 'band_1_mean',
      'band_2_max', 'band_2_min', 'band_2_median', 'band_2_mean']

X = df[cs].values

lr = LogisticRegression()

lr.fit(X, y)
print('logistic regression baseline score w/o inc_angle:')
print(lr.score(X, y))
print()


does it make sense to use an interolated inc_angle? no.
also, note that 100% of images w/ inc_angle == 'na' are class 0


logistic regression with mean fill on inc_angle:
0.6963840399

logistic regression without inc_angle:
0.6963840399

logistic regression with LinearRegression fill:
0.695137157107
'''

# prep image data for neural net
# rescale to +/- 1

X = normalize(np.stack([x1, x2, np.subtract(x2, x1)], axis=1))
X = np.add(np.multiply(X, 2), -1)


# >>> X.shape
# (1604, 3, 75, 75)
y.reshape(-1, 1)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'target': torch.from_numpy(target)}


class IcebergDataset(torch.utils.data.Dataset):
    '''dataset containing icebergs for Kaggle comp'''
    def __init__(self, X, y, transform=None):
        # self.df = df
        self.X = X
        self.y = y
        self.transform = transform
        assert len(self.X) == len(self.y)  # == len(self.df)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        sample = {'image': self.X[i], 'target': self.y[i].reshape(1, 1)}

        if self.transform:
            self.transform(sample)

        return sample

    def show(self, i):
        # show some samples
        arr = self.__getitem__(i)['image']
        img = np.multiply(np.add(arr, 1), 127.5)
        for ch in range(img.shape[0]):
            Image.fromarray(img[ch]).show()


data = IcebergDataset(X, y, transform=transforms.Compose([
    ToTensor()]))

data.show(5)

dataloader = torch.utils.data.DataLoader(data, batch_size=32,
                                         shuffle=True, num_workers=4)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        labels, inputs = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
