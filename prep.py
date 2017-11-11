"""
prep training data & write to file to reduce load times/improve repeatability
stores df as .csv and X, y as .pkl

run 1x
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.ndimage.filters as filters
from PIL import Image


def reformat_dataframe(df):
    """take dataframe; index to id; sort by id; reshape images as ndarray"""

    # id is random; sort by id to pseudo-randomize data
    input_df.sort_values('id', inplace=True)
    input_df.set_index('id', inplace=True)

    # make ndarrays from arrays
    for band in 'band_1', 'band_2':
        input_df[band] = input_df[band].apply(lambda x: np.array(x).reshape(75, 75))

    return df


def standardize(arr):
    """subtract mean and divide by standard dev."""
    return np.divide(np.subtract(arr, arr.mean()), arr.std())


class Standardizer:
    """subtract mu and divide by sigma - provided separately."""
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def standardize(self, arr):
        out = np.divide(arr - self.mu, self.sigma)
        # print('pseudo-standardized; mean is {}'.format(out.mean()))
        return out


def find_brightest_region(image, n=7):
    """
    this function has side effects if you do not replace im with a copy!
    locate bright spots as reference points for other transforms
    args:
        im: image channel
        n:  number of points to check
    returns:
        x, y: element-wise medians of brightest points
    """
    im = image.copy()
    shp = im.shape
    xs = []
    ys = []
    pts = []
    for pt in range(n):
        bright_x, bright_y = np.unravel_index(np.argmax(im), shp)
        im[bright_x, bright_y] = -100
        xs.append(bright_x)
        ys.append(bright_y)
        pts.append([bright_x, bright_y])
    mx, my = np.median(xs), np.median(ys)

    return mx, my, ([mx, my] in pts)


def make_stats_frame(data, is_test_set=False, include_all_data=False):
    """
    make a stats dataframe; includes index, inc_angle, coordinates of highlight
    if is_test_set, also includes y as column is_iceberg
    if include_all_data, includes stats on each image
    """
    stats_df = pd.DataFrame()
    stats_df['id'] = data.index
    stats_df.set_index('id', inplace=True)
    if not is_test_set:
        print('not test_set, therefore including y in stats_df')
        stats_df['is_iceberg'] = data.is_iceberg
    stats_df['inc_angle'] = data.inc_angle
    # df['predicted'] = [None] * len(df)

    if include_all_data:
        # extra per-image stats, unused
        for band in 'band_1', 'band_2':
            stats_df[band + '_max'] = data[band].apply(np.max)
            stats_df[band + '_min'] = data[band].apply(np.min)
            stats_df[band + '_mean'] = data[band].apply(np.mean)
            stats_df[band + '_median'] = data[band].apply(np.median)

    stats_df['highlight'] = data.band_1.apply(find_brightest_region)
    stats_df['hl_x'] = stats_df['highlight'].apply(lambda x: int(x[0]))
    stats_df['hl_y'] = stats_df['highlight'].apply(lambda x: int(x[1]))

    return stats_df


def blur_keep_highlight(im_stack, h_size=1, fltr=None):
    # blur image
    if fltr is None:
        fltr = [0, 1, 1]
    print("old min:", im_stack.min())

    # CAUTION: dim0 is across stacked images
    blur = filters.gaussian_filter(im_stack, fltr)
    for i in range(len(im_stack)):
        ix, iy = df.x.iloc[i], df.y.iloc[i]
        a, b, c, d = ix-h_size, ix+h_size, iy-h_size, iy+h_size
        blur[i, a:b, c:d] = np.maximum(im_stack[i, a:b, c:d],
                                       blur[i, a:b, c:d])
    # blur = np.maximum(im_stack, blur)
    print("new min:", blur.min())
    im_stack = standardize(blur)
    return im_stack


def my_blur(im_stack, fltr=None):
    # apply gaussian_filter within image, standardize
    if fltr is None:
        fltr = [0, 2, 2]
    return standardize(filters.gaussian_filter(im_stack, fltr))


def make_image_stacks(data, start=None, stop=None):
    if start or stop:
        partial_df = data.iloc[start: stop]
    else:
        partial_df = data
    x1 = np.stack(partial_df.band_1.values)
    x2 = np.stack(partial_df.band_2.values)
    x3 = (x1 - x1.min()) * (x2 - x2.min())
    x3 = standardize(x3)
    image_stack = np.stack([x1, x2, x3], axis=1)
    return image_stack


# show sample of X
def show_sample(image_stack):
    for i in range(1, 4):
        offset = 5
        im1 = image_stack[i + offset, 0]
        plt.subplot('33' + str(3*i - 2))
        plt.imshow(Image.fromarray((im1 - im1.min()) * 255
                                   / (im1.max() - im1.min())))

        im2a = image_stack[i + offset, 1]
        plt.subplot('33' + str(3*i - 1))
        plt.imshow(Image.fromarray((im2a - im2a.min()) * 255
                                   / (im2a.max() - im2a.min())))

        im2 = image_stack[i + offset, 2]
        plt.subplot('33' + str(3*i))
        plt.imshow(Image.fromarray((im2 - im2.min()) * 255
                                   / (im2.max() - im2.min())))

    plt.show()


if __name__ == '__main__':

    # use test or training file
    test = True

    # load data
    if test:
        filepath = 'data/test.json'
    else:
        filepath = 'data/train.json'

    input_df = pd.read_json(filepath)

    input_df = reformat_dataframe(input_df)
    print('reformatted input dataframe:')
    print(input_df.head())

    print('normalizing / prepping / building X...')

    if not test:
        # save values for test set
        with open('data/mu_sigma.pkl', 'wb') as f:
            pickle.dump(np.stack(input_df.band_1.values.mean()), f)
            pickle.dump(np.stack(input_df.band_2.values.std()), f)
            pickle.dump(np.stack(input_df.band_1.values.mean()), f)
            pickle.dump(np.stack(input_df.band_2.values.std()), f)

    # load values from training set
    with open('data/mu_sigma.pkl', 'rb') as f:
        mu1 = pickle.load(f)
        sigma1 = pickle.load(f)
        mu2 = pickle.load(f)
        sigma2 = pickle.load(f)

    # standardize / stack
    s1 = Standardizer(mu1, sigma1)
    s2 = Standardizer(mu2, sigma2)
    input_df['band_1'] = input_df['band_1'].apply(s1.standardize)
    input_df['band_2'] = input_df['band_2'].apply(s2.standardize)

    X = make_image_stacks(input_df)

    print('built dataset; is_test_set =', test, X.min(), X.max(), X.shape)

    for ch in range(3):
        print('channel', ch, ':: ', X[:, ch].min(), X[:, ch].max())

    # save x (and optionally y for training) to pickle
    if test:
        file = 'data/test_dataset.pkl'
    else:
        file = 'data/train_dataset.pkl'

    with open(file, 'wb') as f:
        print('writing X to pickle...')
        pickle.dump(X, f)

        if test:
            print('training: saving y to pickle...')
            y = df.is_iceberg.values
            y.reshape(-1, 1)
            assert X.shape[0] == y.shape[0]
            pickle.dump(y, f)
        else:
            print('test, not saving y')

    if test:
        df_path = 'data/test_normalized_stats.csv'
    else:
        df_path = 'data/train_normalized_stats.csv'

    print('writing stats dataframe to csv')
    stats_about_X = make_stats_frame(input_df, is_test_set=test)
    stats_about_X.to_csv(df_path)

    show_sample(X)
