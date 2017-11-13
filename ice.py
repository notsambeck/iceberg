"""
icebergs

dataframe indexed to id, e.g. 'bc92da1'


note that 100% of images w/ inc_angle == 'na' are class 0

logistic regression - baseline predictions:

from sklearn.linear_model import LogisticRegression
cs = ['band_1_max', 'band_1_min', 'band_1_median', 'band_1_mean',
      'band_2_max', 'band_2_min', 'band_2_median', 'band_2_mean']
X = df[cs].values
lr = LogisticRegression()

lr.fit(X, y)
print(lr.score(X, y))


does it make sense to use an interpolated inc_angle? no.

logistic regression with mean fill on inc_angle: 0.6963840399
logistic regression without inc_angle: 0.6963840399
logistic regression with LinearRegression fill: 0.695137157107
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # noqa

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim

import affine_transforms as af
from net_parameters import IceNet
from ice_transforms import center_crop
import pickle

from PIL import Image


class IcebergDataset(Dataset):
    """dataset containing icebergs for Kaggle comp
    internally, df, X, and y are indexed by i as 0...n-1
    kind in ['test', 'training', 'xval']
    """
    def __init__(self, _x, _df, transform=None, kind='training'):
        # generate difference channel

        self.X = _x
        self.transform = transform
        self.kind = kind
        self.df = _df
        assert len(self.X) == len(self.df)
        self.n = len(self.X)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # get ith element of dataset, including df entry
        # getitem calls applicable transforms on the X component
        x = self.X[i]

        # do general transforms
        if self.transform:
            x = self.transform(torch.from_numpy(x).float())

        # center crop with optional flip
        if type(x) is not np.ndarray:
            x = x.numpy()
        flip = np.random.choice([0, 1])
        if flip:
            x = np.flip(x, 2).copy()
        x = torch.from_numpy(center_crop(x))

        d = self.df.iloc[i].to_dict()
        return x, d

    def show(self, n=40, random_sample=True, show_metadata=False):
        # show approximately n samples and some data about them
        rows = int(n ** .3)
        cols = int(n // rows)
        channels = 3
        fig, axes = plt.subplots(channels*rows, cols)

        for row in range(rows):
            for col in range(cols):
                # choose image to pull
                if random_sample:
                    index = np.random.randint(0, self.n)
                else:
                    index = row * cols + col

                # get some data from df
                arr, data = self.__getitem__(index)

                # convert any torch.tensors
                if type(arr) is not np.ndarray:
                    arr = arr.numpy()

                correct = data['predicted'] == data['is_iceberg']
                axes[row*channels, col].set_title(str(data['is_iceberg']) +
                                                  ', p=' + str(data['prob']))

                if show_metadata:
                    axes[row*channels+1, col].set_title(data.index)
                    axes[row*channels+2, col].set_title(' '.join(
                        [str(c) for c in data.highlight] + [str(data.inc_angle)]))

                # full contrast images for display
                img = np.multiply(arr - arr.min(),
                                  255/(arr.max() - arr.min()))

                for ch in range(channels):
                    # loc is place in grid
                    loc = ch + row*channels, col

                    axes[loc].axis('off')
                    axes[loc].title.set_fontsize(8)
                    if ch < img.shape[0]:
                        axes[loc].imshow(Image.fromarray(img[ch]))

                        # identify incorrect predictions
                        if ch == 0:
                            if not correct:
                                axes[loc].plot(3, 3, 'rx')

        plt.show()


def set_rate(rate):
    # BUILD NEW OPTIMIZER (for variable-rate training)
    global optimizer
    optimizer = optim.SGD(net.parameters(), lr=rate, momentum=0.9)
    return True


# noinspection PyArgumentList,PyUnresolvedReferences
def run_model_epoch(model, loader, opt, kind):
    """
    run 1 epoch of model as type training, xval, or test;
    return appropriate metrics plus a dictionary containing prediction probabilities
    """
    running_loss = 0.0
    items_processed = 0
    correct = 0
    results_dict = {}  # image id: probability

    if kind == 'test':
        probabilities = {}

    # turn dropout on/off
    if kind == 'training':
        net.training = True
    else:
        net.training = False

    for i, data in enumerate(loader, 0):
        # get the input (loader.batch_size examples)
        images, data_dict = data

        images = images.cuda().float()
        image_ids = data_dict['id.1']
        targets = data_dict['is_iceberg']
        # print('targets:', targets, 'images', images.size())

        # wrap them in Variable
        images, targets = Variable(images.cuda()), Variable(targets.cuda())

        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs = model(images, v=0)
        items_processed += len(image_ids)

        # write probabilities to dictionary - prob = e^softmax_out[1]
        for j in range(loader.batch_size):
            results_dict[image_ids[j]] = np.exp(outputs.data[j, 1])

        if kind == 'training':
            # print(outputs.size())
            loss = criterion(outputs, targets)
            loss.backward()
            opt.step()

            # statistics
            running_loss += loss.data[0]

        elif kind == 'xval':
            loss = criterion(outputs, targets)
            _, ps = torch.max(outputs, dim=1)

            correct += (ps == targets).sum().float()

            # statistics
            running_loss += loss.data[0]

        elif kind == 'test':
            prob = outputs[1].cpu().data.numpy()
            # noinspection PyUnboundLocalVariable
            probabilities[image_ids] = prob

        else:
            raise ValueError('invalid use: not train, test or xval')

    # write probs to dataset dataframe:
    for image_id, prob in results_dict.items():
        loader.dataset.df['prob'].loc[image_id] = prob

    if kind == 'training':
        return running_loss, items_processed
    elif kind == 'xval':
        return running_loss, items_processed, correct
    else:
        return probabilities, items_processed


def train(epochs, model, _train_loader, _val_loader, opt):

    for epoch in range(epochs):  # loop over the full dataset
        loss, items_processed = run_model_epoch(model,
                                                _train_loader,
                                                opt,
                                                kind='training')
        print(loss, items_processed, loss/items_processed)

        if epoch % 10 == 9:
            loss, items_processed, correct = run_model_epoch(model,
                                                             _val_loader,
                                                             opt,
                                                             kind='xval')
            print(loss/items_processed, correct[0], 'of', items_processed)

    print('Finished Training {} Epochs'.format(epochs))


if __name__ == '__main__':

    print('loading X, y pickle')
    training_path = 'data/train_dataset.pkl'
    if os.path.exists(training_path):
        with open(training_path, 'rb') as f:
            X = pickle.load(f)
    else:
        raise IOError('training dataset does not exist; run prep.py')

    df_path = 'data/train_normalized_stats.csv'
    print('loading stats dataframe from csv...')
    data_f = pd.read_csv(df_path)
    data_f.set_index('id', inplace=True)
    assert len(data_f) == X.shape[0]
    print('dataframe loaded')
    split = 32 * 48

    X_train = X[:split]
    df_train = data_f.iloc[:split]

    X_test = X[split:]
    df_test = data_f.iloc[split:]

    # dtype = torch.FloatTensor
    dtype = torch.cuda.FloatTensor  # to run on GPU

    print('initializing IceNet')
    net = IceNet(3)  # this is dumb, you must set channels (3 != 2)
    net.float()  # floating point
    net.cuda()  # gpu

    # negative log loss; there is an output softmax layer in model
    criterion = nn.NLLLoss(size_average=False)  # return sum of losses

    # optimizer = optim.Adam(net.parameters(), lr=.00003, weight_decay=1e-5)
    optimizer = optim.SGD(net.parameters(), lr=0.0003, momentum=0.9)

    # DEFINE TRANSFORMATIONS

    train_trs = af.RandomAffine(rotation_range=12.0,
                                translation_range=.08,
                                zoom_range=[.7, 1.3])

    xval_trs = None

    # DEFINE DATASETS
    print('building datasets...')
    train_dataset = IcebergDataset(X_train,
                                   df_train,
                                   transform=train_trs)

    xval_dataset = IcebergDataset(X_test,
                                  df_test,
                                  kind='xval',
                                  transform=xval_trs)

    xval2_dataset = IcebergDataset(X_test,
                                   df_test,
                                   kind='training',
                                   transform=train_trs)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                               shuffle=True, num_workers=4)

    xval_loader = torch.utils.data.DataLoader(xval_dataset, batch_size=32,
                                              shuffle=False, num_workers=4)

    xval2_loader = torch.utils.data.DataLoader(xval2_dataset, batch_size=32,
                                               shuffle=False, num_workers=4)

    print('showing samples')
    train_dataset.show(8)

    print('training model...')

    train(200, net, train_loader, xval_loader, optimizer)

    train_dataset.show(8)
    xval_dataset.show(8)
