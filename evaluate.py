'''
NOTE - input dimension is manual
'''

import pandas as pd
import torch
from torch.autograd import Variable
import numpy as np   # noqa
from ice import IcebergDataset
from net_parameters import IceNet
import pickle

ids = pd.read_csv('data/test_ids.csv')
ids.set_index('id', inplace=True)

with open('data/test_dataset_X.pkl', 'rb') as f:
    X = pickle.load(f)

net = IceNet(3)
net = torch.load('model/nov9.torch')

_preds = {}
_probs = {}


def predict(loader):
    '''
    write predictions, probs
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

    # print('batch done')
    pred_df.set_index('id', inplace=True)
    return pred_df


cs = ['val0', 'val1', 'pred']
out = pd.DataFrame(columns=cs)


dataset = IcebergDataset(X,
                         None,
                         ids,
                         transform=None,
                         kind='test')

loader = torch.utils.data.DataLoader(dataset,
                                     batch_size=32,
                                     shuffle=False,
                                     num_workers=4)

out = out.append(predict(loader))

final = pd.DataFrame(out['val1'])

lo = final < .002
final[lo] = .002

hi = final > .998
final[hi] = .998

final.to_csv('preds/nov9_2pm.csv')
