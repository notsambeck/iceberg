'''
icebergs

dataframes and _last_pred indexed to id e.g. 'bc92da1'


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
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # noqa

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim

# import torchvision
import affine_transforms as af
from net_parameters import IceNet
from ice_transforms import center_crop
import pickle

from PIL import Image


class IcebergDataset(Dataset):
    '''dataset containing icebergs for Kaggle comp
    internally, df, X, and y are indexed by i as 0...n-1
    '''
    def __init__(self, _X, _y, _df, transform=None, kind='training'):
        # generate difference channel

        self.X = _X
        self.transform = transform
        self.kind = kind
        if self.kind != 'test':
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
        # all test set:
        x = self.X[i]

        if self.transform:
            x = self.transform(torch.from_numpy(x).float())

        if self.kind == 'xval' or self.kind == 'test':
            x = center_crop(x, center=(self.df.x.iloc[i], self.df.y.iloc[i]))
        else:
            if type(x) is not np.ndarray:
                x = x.numpy()
            flip = np.random.choice([0, 1])
            if flip:
                x = np.flip(x, 2).copy()
            x = torch.from_numpy(center_crop(x))

        if self.kind == 'training' or self.kind == 'xval':
            return x, self.df.index[i], self.y[i].reshape(1, 1)
        else:
            return x, self.df.index[i]

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
                if self.kind != 'test':
                    arr, ID, label = self.__getitem__(index)
                else:
                    arr, ID = self.__getitem__(index)
                    label = 'test'

                if type(arr) is not np.ndarray:
                    arr = arr.numpy()

                p = self.latest_pred.get(ID)
                prob = self.latest_prob.get(ID)

                l = label[0][0]
                correct = p == l
                # angle = str(self.df.angle.iloc[index])
                coords = [self.df.x.iloc[index], self.df.y.iloc[index]]
                img = np.multiply(arr - arr.min(),
                                  255/(arr.max() - arr.min()))

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
                                axes[loc].plot(coords[0], coords[1], 'bx')

        plt.show()


print('loading X, y pickle')
train = 'data/train_dataset.pkl'
if os.path.exists(train):
    with open(train, 'rb') as f:
        X = pickle.load(f)
        y = pickle.load(f)
    assert X.shape[0] == y.shape[0]

df_path = 'data/train_normalized_stats.csv'
print('loading stats dataframe from csv')
dataf = pd.read_csv(df_path)
dataf.set_index('id', inplace=True)
assert len(dataf) == X.shape[0]


split = 32 * 32

X_train = X[:split]
y_train = y[:split]
df_train = dataf.iloc[:split]

X_test = X[split:]
y_test = y[split:]
df_test = dataf.iloc[split:]


# global dictionary that stores predictions to avoid altering dfs
# {index: pred}
_last_pred = {}
_last_prob = {}

# dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor  # to run on GPU

net = IceNet(3)    # this is dumb, you must set #channels (3 != 2)
net.float()        # floating point
net.cuda()         # gpu


# criterion = nn.LogSoftmax()
criterion = nn.CrossEntropyLoss()
'''
optimizer = optim.SGD(net.parameters(), lr=0.0003, momentum=0.9,
                      weight_decay=1e-3)

optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

optimizer = optim.Adam(net.parameters(), lr=.00003, weight_decay=1e-5)

for later on:
'''
optimizer = optim.SGD(net.parameters(), lr=0.03, momentum=0.9)

# DEFINE TRANSFORMATIONS


def set_rate(rate):
    # BUILD NEW OPIMIZER (for variable-rate training)
    global optimizer
    optimizer = optim.SGD(net.parameters(), lr=rate, momentum=0.9)
    return True


# DEFINE DATASETS

train_trs = af.RandomAffine(rotation_range=10.0,
                            translation_range=.05,
                            zoom_range=[.7, 1.3])

xval_trs = None

train_dataset = IcebergDataset(X_train,
                               y_train,
                               df_train,
                               transform=train_trs)

xval_dataset = IcebergDataset(X_test,
                              y_test,
                              df_test,
                              kind='xval',
                              transform=xval_trs)

xval2_dataset = IcebergDataset(X_test,
                               y_test,
                               df_test,
                               kind='training',
                               transform=xval_trs)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                           shuffle=True, num_workers=4)

xval_loader = torch.utils.data.DataLoader(xval_dataset, batch_size=32,
                                          shuffle=False, num_workers=4)


xval2_loader = torch.utils.data.DataLoader(xval2_dataset, batch_size=32,
                                           shuffle=False, num_workers=4)


def train(n, path='model/nov8.torch'):

    for epoch in range(n):  # loop over the dataset multiple times

        running_loss = 0.0
        net.training = True
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            image, ID, target = data

            image = image.cuda().float()
            target = target.view(-1)
            # print('target:', target.size(), 'image', image.size())

            # wrap them in Variable
            image, target = Variable(image.cuda()), Variable(target.cuda())

            # print(type(image), len(image), type(image[0]))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(image, v=0)
            # print(outputs.size())
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]

        print('%d - loss: %.3f' % (epoch + 1, running_loss / i))

        val_loss = 0.0
        net.training = False

        if epoch % 10 == 9:
            # print('\n## validation ## ')
            correct, total = 0, 0
            for i, data in enumerate(xval_loader, 0):
                # get the inputs
                image, ID, target = data

                image = image.cuda().float()
                target = target.view(-1)
                # print('target:', target.size(), 'image', image.size())

                # wrap them in Variable
                image, target = Variable(image.cuda()), Variable(target.cuda())

                optimizer.zero_grad()
                outputs = net(image)

                loss = criterion(outputs, target)
                _, preds = torch.max(outputs, 1)
                # print(preds, target)
                # return preds, target
                correct += (preds == target).sum().float()
                total += len(preds)

                # print statistics
                val_loss += loss.data[0]

            print('val: loss: {:.3}  accuracy: {} of {}'.format(
                  val_loss / i, correct.data.cpu().numpy(), total))

            if val_loss / i < net.best_xval_loss:
                print('Save. #save')
                net.best_xval_loss = val_loss / i
                torch.save(net, path)

            val_loss = 0.0
            net.training = False

            correct, total = 0, 0
            for i, data in enumerate(xval2_loader, 0):
                # get the inputs
                image, ID, target = data

                image = image.cuda().float()
                target = target.view(-1)
                # print('target:', target.size(), 'image', image.size())

                # wrap them in Variable
                image, target = Variable(image.cuda()), Variable(target.cuda())

                optimizer.zero_grad()
                outputs = net(image)

                loss = criterion(outputs, target)
                _, preds = torch.max(outputs, 1)
                # print(preds, target)
                # return preds, target
                correct += (preds == target).sum().float()
                total += len(preds)

                # print statistics
                val_loss += loss.data[0]

            print('val with transform: loss: {:.3}  accuracy: {} of {}'.format(
                  val_loss / i, correct.data.cpu().numpy(), total))

    print('Finished Training')


def write_preds(loader=xval_loader):
    '''
    write predictions, probs to dataset.latest_pred, dataset.latest_prob
    '''
    cs = ['id',  'val0', 'val1', 'target', 'pred']
    pred_df = pd.DataFrame(columns=cs)
    correct, total = 0, 0
    for i, data in enumerate(loader, 0):
        # get the inputs
        image, ID, target = data

        image = image.cuda().float()
        y = target.view(-1)
        # print('target:', target.size(), 'image', image.size())

        # wrap them in Variable
        image, target = Variable(image.cuda()), Variable(y.cuda())
        outputs = net(image)
        values = nn.Softmax()(outputs)
        # print(outputs)
        vals, preds = torch.max(outputs, 1)   # value, loc of value (argmax)
        tdf = pd.DataFrame(columns=cs)
        tdf['id'] = ID
        tdf['val0'] = values[:, 0].cpu().data.numpy()
        tdf['val1'] = values[:, 1].cpu().data.numpy()
        tdf['target'] = y
        tdf['pred'] = preds.cpu().data.numpy()
        # print(tdf.head())
        pred_df = pred_df.append(tdf)
        # return preds, target
        correct += (preds == target).sum().float()
        total += len(preds)

    pred_df.set_index('id', inplace=True)
    for i in pred_df.index:
        loader.dataset.latest_pred[i] = pred_df.pred[i]
        loader.dataset.latest_prob[i] = pred_df.val1[i]
    return True
