'''transforms for images / pytorch tensors'''
import torch
import pickle
import numpy as np
import affine_transforms as af

# pickle stats for access in test_loader
pkl = 'data/stats.pkl'
with open(pkl, 'rb') as f:
    min1 = pickle.load(f)
    max1 = pickle.load(f)
    min2 = pickle.load(f)
    max2 = pickle.load(f)

diff1 = max1 - min1
diff2 = max2 - min2


def norm1(arr):
    return np.divide(np.subtract(arr, min1), diff1)


def norm2(arr):
    return np.divide(np.subtract(arr, min2), diff2)


def clip_low(x):
    '''clip an image to values above median'''
    median = torch.median(x)
    return torch.clamp(x, min=median)


def clip_low_except_center(x, size=10):
    '''clip an image to values above median,
    but keep entire image in square of +/- size px
    about center of image'''

    median = torch.median(x)
    out = torch.clamp(x, min=median)
    cntr = x.size()[1] // 2
    im_cntr = x[:, cntr-size:cntr+size, cntr-size:cntr+size]
    out[:, cntr-size:cntr+size, cntr-size:cntr+size] = im_cntr
    return out


def center_crop(x, c=12):
    '''crop center of square images by c'''
    return x[:, c:1-c, c:1-c]


def scale_to_angle(x, angle=.6):
    '''don't do this it's broken?'''
    return af.Zoom(1.6 - .75 * angle)(x)