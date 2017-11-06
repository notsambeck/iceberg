import pandas as pd
import torch
from torch.autograd import Variable
import numpy as np
from net_parameters import IcebergDataset, IceNet
from ice_transforms import clip_low_except_center, norm1, norm2


df = pd.read_json('data/test.json')
df.set_index('id', inplace=True)

for band in 'band_1', 'band_2':
    df[band] = df[band].apply(lambda x: np.array(x).reshape(75, 75))


df['band_1'] = df['band_1'].apply(norm1)
df['band_2'] = df['band_2'].apply(norm2)

n = len(df)
batch_size = 64
batches = n // batch_size + 1

net = IceNet(2)
net = torch.load('model/contrast_center_nov_5.torch')

_preds = {}
_probs = {}


def predict(loader):
    '''
    write predictions, probs to preds, probs
    '''
    cs = ['id',  'val0', 'val1', 'pred']
    pred_df = pd.DataFrame(columns=cs)
    for i, data in enumerate(loader, 0):
        # get the inputs
        image, ID = data

        image = image.cuda().float()
        # wrap them in Variable
        image = Variable(image.cuda())

        outputs = net(image)
        values = torch.nn.Softmax()(outputs)
        # print(outputs)
        vals, ps = torch.max(outputs, 1)   # value, loc of value (argmax)
        tdf = pd.DataFrame(columns=cs)
        tdf['id'] = ID
        tdf['val0'] = values[:, 0].cpu().data.numpy()
        tdf['val1'] = values[:, 1].cpu().data.numpy()
        tdf['pred'] = ps.cpu().data.numpy()
        # print(tdf.head())
        pred_df = pred_df.append(tdf)

    print('batch done')
    pred_df.set_index('id', inplace=True)
    return pred_df


cs = ['val0', 'val1', 'pred']
out = pd.DataFrame(columns=cs)

for b in range(batches):
    start = b * batch_size
    end = (b + 1) * batch_size
    end = min(end, n)
    x1 = np.stack(df.band_1.iloc[start: end])
    x2 = np.stack(df.band_2.iloc[start: end])
    X = np.stack([x1, x2], axis=1)
    X = np.add(np.multiply(X, 2), -1)
    batch_dataset = IcebergDataset(X,
                                   None,
                                   df.iloc[start: end],
                                   transform=clip_low_except_center,
                                   training=False)
    batch_loader = torch.utils.data.DataLoader(batch_dataset,
                                               batch_size=32,
                                               shuffle=False,
                                               num_workers=4)
    out = out.append(predict(batch_loader))
