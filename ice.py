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
import torch.nn as nn
import torch.optim as optim

# import torchvision
import torchvision.transforms as transforms
import affine_transforms as af
from net_parameters import IcebergDataset, IceNet
from ice_transforms import norm1, norm2, clip_low_except_center
from ice_transforms import center_crop, contrast_background
import pickle


# global dictionary that stores predictions to avoid altering dfs
# {index: pred}
_last_pred = {}
_last_prob = {}


# LOAD TRAINING DATA

imdata = pd.read_json('data/train.json')
imdata.sort_values('id', inplace=True)
imdata.set_index('id', inplace=True)

# IMAGE NORMS / STATS

for b in 'band_1', 'band_2':
    imdata[b] = imdata[b].apply(lambda x: np.array(x).reshape(75, 75))

x1 = np.stack(imdata.band_1)
x2 = np.stack(imdata.band_2)


# pickle stats here, OR load normalize from ice_transforms
pkl = 'data/stats.pkl'
if not os.path.exists(pkl):
    min1 = np.min(x1)
    max1 = np.max(x1)
    min2 = np.min(x2)
    max2 = np.max(x2)
    with open(pkl, 'wb') as f:
        pickle.dump(min1, f)
        pickle.dump(max1, f)
        pickle.dump(min2, f)
        pickle.dump(max2, f)
else:
    with open(pkl, 'rb') as f:
        min1 = pickle.load(f)
        max1 = pickle.load(f)
        min2 = pickle.load(f)
        max2 = pickle.load(f)


# store normalized data in df? yes

imdata['band_1'] = imdata['band_1'].apply(norm1)
imdata['band_2'] = imdata['band_2'].apply(norm2)


def make_stats_frame(save=False):
    df = pd.DataFrame()
    df['id'] = imdata.index
    df.set_index('id', inplace=True)
    df['is_iceberg'] = imdata.is_iceberg
    df['angle'] = imdata.inc_angle
    df['pred'] = [None] * len(df)

    for b in 'band_1', 'band_2':
        df[b + '_max'] = imdata[b].apply(np.max)
        df[b + '_min'] = imdata[b].apply(np.min)
        df[b + '_mean'] = imdata[b].apply(np.mean)
        df[b + '_median'] = imdata[b].apply(np.median)
        df[b + '_coords'] = imdata[b].apply(
            lambda x: np.unravel_index(np.argmax(x), x.shape))
        df[b + '_x'] = df[b + '_coords'].apply(lambda x: x[0])
        df[b + '_y'] = df[b + '_coords'].apply(lambda x: x[1])

    if save:
        df.to_csv(save)

    return df


df_path = 'data/stat_frame.csv'

if os.path.exists(df_path):
    df = pd.read_csv(df_path, index_col=0)
else:
    print('making new stats dataframe! \n')
    df = make_stats_frame(save=df_path)
    df.to_csv(df_path)

y = df.is_iceberg.values

# prep image data for neural net
# rescale to +/- 1

print('normalizing / building X')
# X = normalize(np.stack([x1, x2], axis=1))
x1 = norm1(x1)
x2 = norm2(x2)
X = np.stack([x1, x2], axis=1)
X = np.add(np.multiply(X, 2), -1)
print(X.shape)

# >>> X.shape
# (1604, 3, 75, 75)
y.reshape(-1, 1)

split = 32 * 36

X_train = X[:split]
y_train = y[:split]
df_train = df.iloc[:split]

X_test = X[split:]
y_test = y[split:]
df_test = df.iloc[split:]


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
'''
optimizer = optim.Adam(net.parameters(), lr=.001)

'''
for later on:
optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9, weight_decay=1e-3)   # noqa
'''

# DEFINE TRANSFORMATIONS


def set_rate(rate):
    # BUILD NEW OPIMIZER (for variable-rate training)
    global optimizer
    optimizer = optim.SGD(net.parameters(), lr=rate, momentum=0.9)
    return True


# DEFINE DATASETS

train_trs = transforms.Compose([af.RandomZoom([.7, 1.3]),
                                clip_low_except_center,
                                af.RandomTranslate(.10),
                                # af.RandomRotate(180),
                                af.RandomChoiceRotate([0, 5, 85, 90, 95, 175,
                                                       180, 185, 265, 270]),
                                # contrast_background,
                                center_crop])

# scale_to_angle happens in loader; ok because images only get expanded
xval_trs = transforms.Compose([center_crop])

train_dataset = IcebergDataset(X_train,
                               y_train,
                               df_train,
                               transform=train_trs)

xval_dataset = IcebergDataset(X_test,
                              y_test,
                              df_test,
                              transform=xval_trs)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                           shuffle=True, num_workers=4)

xval_loader = torch.utils.data.DataLoader(xval_dataset, batch_size=32,
                                          shuffle=False, num_workers=4)


def train(n, path='model/full_color_nov_5.torch'):

    for epoch in range(n):  # loop over the dataset multiple times

        running_loss = 0.0
        net.training = True
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            image, target, ID = data

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
            if i == 16:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / i))

        val_loss = 0.0
        net.training = False

        if epoch % 10 == 9:
            print('\n## validation ## ')
            correct, total = 0, 0
            for i, data in enumerate(xval_loader, 0):
                # get the inputs
                image, target, ID = data

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

            print('validation loss: %.3f' %
                  (val_loss / i))
            print('accuracy:', correct.data.cpu(), 'of', total)

            if val_loss / i < net.best_xval_loss:
                print('New best! Saving... \n')
                net.best_xval_loss = val_loss / i
                torch.save(net, path)

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
        image, target, ID = data

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

    print('done')
    pred_df.set_index('id', inplace=True)
    for i in pred_df.index:
        loader.dataset.latest_pred[i] = pred_df.pred[i]
        loader.dataset.latest_prob[i] = pred_df.val1[i]
    return pred_df
