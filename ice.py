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
from PIL import Image
import matplotlib.pyplot as plt  # noqa

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import torchvision
import torchvision.transforms as transforms
import affine_transforms as af


# global dictionary that stores predictions to avoid altering dfs
# {index: pred}
_last_pred = {}


# LOAD TRAINING DATA

imdata = pd.read_json('data/train.json')
imdata.set_index('id', inplace=True)

output_df = pd.read_json('data/test.json')
output_df.set_index('id', inplace=True)

# IMAGE NORMS / STATS

for b in 'band_1', 'band_2':
    imdata[b] = imdata[b].apply(lambda x: np.array(x).reshape(75, 75))
    output_df[b] = output_df[b].apply(lambda x: np.array(x).reshape(75, 75))

x1 = np.stack(imdata.band_1)
x2 = np.stack(imdata.band_2)
minimum = np.min([x1, x2])
maximum = np.max([x1, x2])
difference = (maximum - minimum)


def normalize(arr):
    return np.divide(np.subtract(arr, minimum), difference)


# store normalized data in df? yes
for b in 'band_1', 'band_2':
    imdata[b] = imdata[b].apply(normalize)
    output_df[b] = output_df[b].apply(normalize)


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

print('normalizing')
# X = normalize(np.stack([x1, x2], axis=1))
X = normalize(np.stack([x1, x2, np.subtract(x2, x1)], axis=1))
X = np.add(np.multiply(X, 2), -1)
print(X.shape)

# >>> X.shape
# (1604, 3, 75, 75)
y.reshape(-1, 1)

split = 32 * 48

X_train = X[:split]
y_train = y[:split]
df_train = df.iloc[:split]

X_test = X[split:]
y_test = y[split:]
df_test = df.iloc[split:]


# prep test data
x1 = np.stack(output_df.band_1)
x2 = np.stack(output_df.band_2)
output_data = normalize(np.stack([x1, x2, np.subtract(x2, x1)], axis=1))
output_data = np.add(np.multiply(X, 2), -1)
# output_df
print('test data', output_data.shape)


# dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor  # Uncomment this to run on GPU


class IcebergDataset(torch.utils.data.Dataset):
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
        self.transform = transform
        assert len(self.X) == len(self.df)
        self.n = len(self.X)
        global _last_pred

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if not self.training:  # for test set:
            x = self.X[i]
            x = self.transform(x)
            return x, self.df.index[i]  # data, id

        x, y = self.X[i], self.y[i].reshape(1, 1)
        x, y = torch.from_numpy(x).float(), y

        if self.transform:
            x = self.transform(x)

        return x, y, self.df.index[i]   # iloc to avoid non-0 index

    def show(self, n, rando=True):
        # show approximately n samples
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

                p = str(_last_pred.get(ID))
                arr = arr.numpy()
                angle = str(self.df.angle.iloc[index])
                coords = ' '.join([str(self.df.band_1_x[index]),
                                   str(self.df.band_1_y[index]),
                                   str(self.df.band_2_x[index]),
                                   str(self.df.band_2_y[index])])
                img = np.multiply(np.add(arr, 1), 127.5)
                axes[row*channels, col].set_title(str(label)+' - pred: '+p)
                axes[row*channels+1, col].set_title(angle)
                axes[row*channels+2, col].set_title(coords)

                for ch in range(channels):
                    # Image.fromarray(img[ch]).show()   # for PIL
                    loc = ch + row*channels, col
                    axes[loc].axis('off')
                    axes[loc].title.set_fontsize(8)
                    if ch < img.shape[0]:
                        # print image if applicable
                        axes[loc].imshow(Image.fromarray(img[ch]))
        plt.show()


def drop_low(x):
    '''clip an image to values above median'''
    median = torch.median(x)
    return torch.clamp(x, min=median)


trs = transforms.Compose([drop_low, af.RandomRotate(60)])
train_set = IcebergDataset(X_train,
                           y_train,
                           df_train,
                           transform=trs)

xval_set = IcebergDataset(X_test,
                          y_test,
                          df_test,
                          transform=drop_low)

output_dataset = IcebergDataset(output_data,
                                None,
                                output_df,
                                training=False,
                                transform=drop_low)
# dataset.show(5)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                           shuffle=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(xval_set, batch_size=32,
                                          shuffle=False, num_workers=4)
output_loader = torch.utils.data.DataLoader(output_dataset, batch_size=32,
                                            shuffle=False, num_workers=4)

'''
eval_set = IcebergDataset(X, y, df, transform=None)
eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=32,
                                          shuffle=False, num_workers=4)
'''


class Net(nn.Module):
    def __init__(self, channels):
        super(Net, self).__init__()
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


net = Net(X.shape[1])
net.float()
net.cuda()


# criterion = nn.LogSoftmax()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.015, momentum=0.9)


def set_rate(rate):
    global optimizer
    optimizer = optim.SGD(net.parameters(), lr=rate, momentum=0.9)
    return True


def train(n):

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
            outputs = net(image)
            # print(outputs.size())
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i == 35:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / i))

        val_loss = 0.0
        net.training = False

        if epoch % 10 == 9:
            print('\n## validation ## ')
            correct, total = 0, 0
            for i, data in enumerate(test_loader, 0):
                # get the inputs
                image, target, ID = data

                image = image.cuda().float()
                target = target.view(-1)
                # print('target:', target.size(), 'image', image.size())

                # wrap them in Variable
                image, target = Variable(image.cuda()), Variable(target.cuda())
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
                  (epoch + 1, i + 1, val_loss / i))
            print('accuracy:', correct[0], 'of', total)

    print('Finished Training')


def write_preds(loader=test_loader):
    '''
    write predictions to _last_pred
    '''
    global _last_pred
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
        _last_pred[i] = pred_df.pred[i]
    return pred_df
