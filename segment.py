'''
image segmentation based on 1-channel brightness
problem is...it's hard
'''

import numpy as np
from sklearn import cluster
import pickle
import matplotlib.pyplot as plt


def find_brightest_region(image, n=5):
    '''
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
        im[x, y] = 0
        xs.append(x)
        ys.append(y)
        pts.append([x, y])
    mx, my = np.median(xs), np.median(ys)
    return int(mx), int(my), ([mx, my] in pts)


with open('data/train_dataset.pkl', 'rb') as f:
    X = pickle.load(f)
    is_iceberg = pickle.load(f)

img = X[2]
img.shape

im1 = img[0]

brightest = find_brightest_region(im1)
xi, yi = brightest[0], brightest[1]

start = np.array([[xi, yi, im1[xi, yi]],   # BRIGHT
                  [0, 0, im1[0, 0]]])      # EDGE

points = []
'''im1 is 2d matrix of x3 indexed by x1, x2'''
for x in range(im1.shape[0]):
    for y in range(im1.shape[0]):
        points.append((x, y, im1[x, y] * 50))

# kmeans = cluster.KMeans(n_clusters=2, init=start, n_init=1)
# kmeans.fit(points)

agg = cluster.AgglomerativeClustering(n_clusters=6)
agg.fit(points)

mask = agg.labels_.reshape(75, 75)

plt.subplot(212)
plt.imshow(mask)
plt.subplot(211)
plt.imshow((im1 - im1.min()) * 255 / (im1.max() - im1.min()))
plt.scatter(xi, yi, s=1, c='r')
plt.show()
