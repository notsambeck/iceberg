import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt  # noqa


# LOAD TRAINING DATA

df = pd.read_json('data/train.json')

for b in 'band_1', 'band_2':
    df[b] = df[b].apply(lambda x: np.array(x).reshape(75, 75))

x1 = np.stack(df.band_1)
x2 = np.stack(df.band_2)
minimum = np.min([x1, x2])
maximum = np.max([x1, x2])
difference = (maximum - minimum)


# IMAGE NORMS / STATS

def normalize(arr):
    return np.divide(np.subtract(arr, minimum), difference)


for b in 'band_1', 'band_2':
    df[b] = df[b].apply(normalize)
    df[b + '_max'] = df[b].apply(np.max)
    df[b + '_min'] = df[b].apply(np.min)
    df[b + '_mean'] = df[b].apply(np.mean)
    df[b + '_median'] = df[b].apply(np.median)


# logistic regression - baseline predictions:

cs = ['band_1_max', 'band_1_min', 'band_1_median', 'band_1_mean',
      'band_2_max', 'band_2_min', 'band_2_median', 'band_2_mean']

X = df[cs].values
y = df.is_iceberg.values

lr = LogisticRegression()

lr.fit(X, y)
print('logistic regression baseline score w/o inc_angle:')
print(lr.score(X, y))
print()

'''
does it make sense to use an interolated inc_angle? no.
also, note that 100% of images w/ inc_angle == 'na' are class 0


logistic regression with mean fill on inc_angle:
0.6963840399

logistic regression without inc_angle:
0.6963840399

logistic regression with LinearRegression fill:
0.695137157107
'''

# prep image data for neural net:

X = normalize(np.stack([x1, x2, np.subtract(x2, x1)], axis=1))

# >>> X.shape
# (1604, 3, 75, 75)
y.reshape(-1, 1)


# show some samples
for i in range(1):
    img1 = Image.fromarray(np.multiply(X[i][0], 255))
    img1.show()
    img2 = Image.fromarray(np.multiply(X[i][1], 255))
    img2.show()
    img3 = Image.fromarray(np.multiply(X[i][2], 255))
    img3.show()

