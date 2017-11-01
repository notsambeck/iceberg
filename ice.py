import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt


# LOAD TRAINING DATA

df = pd.read_json('data/train.json')

for b in 'band_1', 'band_2':
    df[b] = df[b].apply(lambda x: np.array(x).reshape(75, 75))

x1 = np.stack(df.band_1)
x2 = np.stack(df.band_2)
minimum = np.min([x1, x2])
maximum = np.max([x1, x2])
difference = (maximum - minimum) / 255.


# IMAGE NORMS / STATS

def normalize(arr):
    return np.divide(np.subtract(arr, minimum), difference)


for b in 'band_1', 'band_2':
    df[b] = df[b].apply(normalize)
    df[b + '_max'] = df[b].apply(np.max)
    df[b + '_min'] = df[b].apply(np.min)
    df[b + '_mean'] = df[b].apply(np.mean)
    df[b + '_median'] = df[b].apply(np.median)


for i in range(0):
    arr1 = df.band_1[i]
    arr2 = df.band_2[i]

    img1 = Image.fromarray(arr1)
    img1.show()

    img2 = Image.fromarray(arr2)
    img2.show()


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
does it make sense to use an implied inc_angle? NO!


logistic regression with mean fill on inc_angle:
0.6963840399

logistic regression without inc_angle:
0.6963840399

fitting inc_angle by linear regression on these columns:
['band_1_max', 'band_1_min', 'band_1_median', 'band_2_max',
 'band_2_min', 'band_2_median', 'band_1_mean', 'band_2_mean']
filled 133 values out of 1603
double check len(p): 133

logistic regression with LinearRegression fill:
0.695137157107
'''

# prep image data for neural net:

X = np.stack([x1, x2], axis=1)
