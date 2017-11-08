'''
prep training data & write to file to reduce load times/improve repeatability
stores df as .csv and X, y as .pkl
run 1x
'''

import pandas as pd
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import pickle

import scipy.ndimage.filters as filters
# from scipy.ndimage import uniform_filter, gaussian_filter

# LOAD TRAINING DATA

imdata = pd.read_json('data/train.json')

# id is random; sort by id to randomize data
imdata.sort_values('id', inplace=True)
imdata.set_index('id', inplace=True)


# make ndarrays from arrays
for b in 'band_1', 'band_2':
    imdata[b] = imdata[b].apply(lambda x: np.array(x).reshape(75, 75))

# stack each channel into  n*h*w
x1 = np.stack(imdata.band_1)
x2 = np.stack(imdata.band_2)


def standardize(arr):
    '''subtract mean and divide by standard dev.'''
    return np.divide(np.subtract(arr, arr.mean()), arr.std())

x1 = standardize(x1)
x2 = standardize(x2)

# normalize df
imdata['band_1'] = imdata['band_1'].apply(standardize)
imdata['band_2'] = imdata['band_2'].apply(standardize)


def find_brightest_region(im, n=5):
    '''
    args:
        im: image channel
        n:  number of points to check
    returns:
        x, y: element-wise medians of brightest points
    '''
    shp = im.shape
    xs = []
    ys = []
    pts = []
    for pt in range(n):
        x, y = np.unravel_index(np.argmax(im), shp)
        im[x, y] = -100
        xs.append(x)
        ys.append(y)
        pts.append([x, y])
    mx, my = np.median(xs), np.median(ys)
    return mx, my, ([mx, my] in pts)


def make_stats_frame():
    df = pd.DataFrame()
    df['id'] = imdata.index
    df.set_index('id', inplace=True)
    df['is_iceberg'] = imdata.is_iceberg
    df['angle'] = imdata.inc_angle
    df['pred'] = [None] * len(df)

    if False:
        # extra per-image stats, unused
        for b in 'band_1', 'band_2':
            df[b + '_max'] = imdata[b].apply(np.max)
            df[b + '_min'] = imdata[b].apply(np.min)
            df[b + '_mean'] = imdata[b].apply(np.mean)
            df[b + '_median'] = imdata[b].apply(np.median)

    df['coords'] = imdata.band_1.apply(find_brightest_region)
    df['x'] = df['coords'].apply(lambda x: int(x[0]))
    df['y'] = df['coords'].apply(lambda x: int(x[1]))
    safe = df['coords'].apply(lambda x: x[2])
    print('of 1604, points that were selected:', sum(safe))

    return df


df = make_stats_frame()

# prep image data for neural net
# rescale to +/- 1


print('normalizing / blurring /  building X...')
# X = normalize(np.stack([x1, x2], axis=1))


foot = np.array([[[0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 0]]])


def blur_keep_highlight(im_stack):
    # blur image
    print(im_stack.min())
    blur = filters.median_filter(im_stack, footprint=foot)
    blur = np.maximum(im_stack, blur)
    print(blur.min())
    im_stack = standardize(blur)
    return im_stack


x3 = x1 - x2
x3 = standardize(x3)
X = np.stack([x1, x2, x3], axis=1)

# b1 = blur_keep_highlight(x1)
# b2 = blur_keep_highlight(x2)
# b3 = blur_keep_highlight(x3)
# X = np.stack([b1, b2, b3], axis=1)

for i in range(1, 4):
    offset = 25
    im1 = x1[i+offset]
    plt.subplot('33' + str(3*i - 2))
    plt.imshow(Image.fromarray((im1 - im1.min()) * 255
                               / (im1.max() - im1.min())))

    im2a = x3[i+offset]
    plt.subplot('33' + str(3*i - 1))
    plt.imshow(Image.fromarray((im2a - im2a.min()) * 255
                               / (im2a.max() - im2a.min())))

    im2 = X[i + offset, 0]
    plt.subplot('33' + str(3*i))
    plt.imshow(Image.fromarray((im2 - im2.min()) * 255
                               / (im2.max() - im2.min())))

plt.show()

print('dataset init:', X.min(), X.max(), X.shape)
for ch in range(3):
    print('channel:', ch, '-', X[:, ch].min(), X[:, ch].max())

# >>> X.shape
# (1604, 3, 75, 75)

print('making, reshaping y...')
y = df.is_iceberg.values
y.reshape(-1, 1)

assert X.shape[0] == y.shape[0]


if True:
    print('saving X, y to pickle')
    train = 'data/train_dataset.pkl'
    with open(train, 'wb') as f:
        pickle.dump(X, f)
        pickle.dump(y, f)
    df_path = 'data/train_normalized_stats.csv'
    print('writing stats dataframe to csv')
    df.to_csv(df_path)
