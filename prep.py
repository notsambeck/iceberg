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

test = True

# LOAD DATA
if test:
    filepath = 'data/test.json'
else:
    filepath = 'data/train.json'

imdata = pd.read_json(filepath)
print(imdata.head())

# id is random; sort by id to randomize data
imdata.sort_values('id', inplace=True)
imdata.set_index('id', inplace=True)


# make ndarrays from arrays
for b in 'band_1', 'band_2':
    imdata[b] = imdata[b].apply(lambda x: np.array(x).reshape(75, 75))


def standardize(arr):
    '''subtract mean and divide by standard dev.'''
    return np.divide(np.subtract(arr, arr.mean()), arr.std())


class Standardizer():
    '''subtract mu and divide by sigma - provided separately.'''
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.epsilon = 1e-3

    def standardize(self, arr):
        out = np.divide(arr - self.mu, self.sigma)
        # print('pseudo-standardized; mean is {}'.format(out.mean()))
        return out


if not test:
    # save values for test set
    with open('data/mu_sigma.pkl', 'wb') as f:
        pickle.dump(np.stack(imdata.band_1.values.mean()), f)
        pickle.dump(np.stack(imdata.band_2.values.std()), f)
        pickle.dump(np.stack(imdata.band_1.values.mean()), f)
        pickle.dump(np.stack(imdata.band_2.values.std()), f)


# load values from training set
with open('data/mu_sigma.pkl', 'rb') as f:
    mu1 = pickle.load(f)
    sigma1 = pickle.load(f)
    mu2 = pickle.load(f)
    sigma2 = pickle.load(f)

# standardize
s1 = Standardizer(mu1, sigma1)
s2 = Standardizer(mu2, sigma2)
imdata['band_1'] = imdata['band_1'].apply(s1.standardize)
imdata['band_2'] = imdata['band_2'].apply(s2.standardize)


def find_brightest_region(image, n=7):
    '''
    this function has side effects if you do not replace im with a copy!
    locate bright spots as reference points for other transforms
    args:
        im: image channel
        n:  number of points to check
    returns:
        x, y: element-wise medians of brightest points
    '''
    im = image.copy()
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


def make_stats_frame(imdata=imdata, test=test):
    df = pd.DataFrame()
    df['id'] = imdata.index
    df.set_index('id', inplace=True)
    if not test:
        print('not test, including y in df')
        df['is_iceberg'] = imdata.is_iceberg
    df['angle'] = imdata.inc_angle
    # df['pred'] = [None] * len(df)

    if True:
        # extra per-image stats, unused
        for b in 'band_1', 'band_2':
            df[b + '_max'] = imdata[b].apply(np.max)
            df[b + '_min'] = imdata[b].apply(np.min)
            df[b + '_mean'] = imdata[b].apply(np.mean)
            df[b + '_median'] = imdata[b].apply(np.median)

    df['coords'] = imdata.band_1.apply(find_brightest_region)
    df['x'] = df['coords'].apply(lambda x: int(x[0]))
    df['y'] = df['coords'].apply(lambda x: int(x[1]))
    # safe = df['coords'].apply(lambda x: x[2])

    return df


print('normalizing / blurring /  building X...')
# X = normalize(np.stack([x1, x2], axis=1))


def blur_keep_highlight(im_stack, h_size=1, filt=[0, 1, 1]):
    # blur image
    print(im_stack.min())
    # CAUTION: dim0 is across stacked images
    blur = filters.gaussian_filter(im_stack, filt)
    for i in range(len(im_stack)):
        ix, iy = df.x.iloc[i], df.y.iloc[i]
        a, b, c, d = ix-h_size, ix+h_size, iy-h_size, iy+h_size
        blur[i, a:b, c:d] = np.maximum(im_stack[i, a:b, c:d],
                                       blur[i, a:b, c:d])
    # blur = np.maximum(im_stack, blur)
    print(blur.min())
    im_stack = standardize(blur)
    return im_stack


def my_blur(im_stack, filt=[0, 2, 2]):
    # apply gaussian_filter within image, standardize
    return standardize(filters.gaussian_filter(im_stack, filt))


def make_image_stacks(start, stop, imdata=imdata, test=test):
    partial_df = imdata.iloc[start: stop]
    x1 = np.stack(partial_df.band_1.values)
    x2 = np.stack(partial_df.band_2.values)
    # b1 = my_blur(x1.copy(), filt=[0, 1, 1])
    # b2 = my_blur(x2.copy(), filt=[0, 1, 1])
    # x3 = (b1 - b1.min()) * (b2 - b2.min())
    x3 = (x1 - x1.min()) * (x2 - x2.min())
    x3 = standardize(x3)
    if not test:
        stats_df = make_stats_frame(partial_df)
        return np.stack([x1, x2, x3], axis=1), stats_df
    else:
        return np.stack([x1, x2, x3], axis=1)


# run program:

if not test:
    X, df = make_image_stacks(0, len(imdata))

    print('built train dataset', X.min(), X.max(), X.shape)

    for ch in range(3):
        print('channel:', ch, '-', X[:, ch].min(), X[:, ch].max())

    # >>> X.shape
    # (1604, 3, 75, 75)

    print('making, reshaping y...')
    y = df.is_iceberg.values
    y.reshape(-1, 1)

    assert X.shape[0] == y.shape[0]

    print('saving X, y to pickle')
    train = 'data/train_dataset.pkl'
    with open(train, 'wb') as f:
        pickle.dump(X, f)
        pickle.dump(y, f)
    df_path = 'data/train_normalized_stats.csv'
    print('writing stats dataframe to csv')
    df.to_csv(df_path)

else:
    batches = 1
    pickle_size = len(imdata) // batches

    for batch in range(batches):
        X = make_image_stacks(batch * pickle_size,
                              (batch + 1) * pickle_size)

        print('built partial test dataset', X.min(), X.max(), X.shape)

        for ch in range(3):
            print('channel:', ch, '-', X[:, ch].min(), X[:, ch].max())

        print('saving X to pickle')
        test_batch = 'data/test_dataset_{}.pkl'.format(batch)
        with open(test_batch, 'wb') as f:
            pickle.dump(X, f)
        print('done')

        ids = pd.DataFrame(imdata.index)
        ids.to_csv('data/test_ids_{}.csv'.format(batch))


# show sample of X
for i in range(1, 4):
    offset = 5
    im1 = X[i+offset, 0]
    plt.subplot('33' + str(3*i - 2))
    plt.imshow(Image.fromarray((im1 - im1.min()) * 255
                               / (im1.max() - im1.min())))

    im2a = X[i+offset, 1]
    plt.subplot('33' + str(3*i - 1))
    plt.imshow(Image.fromarray((im2a - im2a.min()) * 255
                               / (im2a.max() - im2a.min())))

    im2 = X[i + offset, 2]
    plt.subplot('33' + str(3*i))
    plt.imshow(Image.fromarray((im2 - im2.min()) * 255
                               / (im2.max() - im2.min())))

plt.show()
