import sys
from os import listdir
from os.path import isfile, join, basename
import scipy.misc
import math
import matplotlib.pyplot as plt
import numpy as np

image_path = "C:/Datasets/Fisheries/train/ALB/img_00085.jpg"
img = scipy.misc.imread(image_path)

plt.imshow(img)
plt.show()
# plt.imshow(img[150:275, 540:900])
# plt.show()
# print (np.shape(img))
# image_parts = []
# labels = []
# for x in range(0,np.shape(img)[0]-40):
#     for y in range(0,np.shape(img)[1]-40):
#         image_parts.append(img[x:x+40,y:y+40])
#         i1 = max(x,150)
#         j1 = max(y,540)
#         i2 = min(x+40,275)
#         j2 = min(y + 40, 900)
#         labels.append(max(i2-i1,0)*max(j2-j1,0))
# labels = [a > 800 for a in labels]

fish_window = ((150, 540), (275, 900))
win_size = (40, 40)
min_xs = np.arange(img.shape[0]-win_size[0])
min_ys = np.arange(img.shape[1]-win_size[1])
max_xs = min_xs + win_size[0]
max_ys = min_ys + win_size[1]
int_min_xs = np.maximum(min_xs, fish_window[0][0])
int_min_ys = np.maximum(min_ys, fish_window[0][1])
int_max_xs = np.minimum(max_xs, fish_window[1][0])
int_max_ys = np.minimum(max_ys, fish_window[1][1])
int_ws = np.maximum(int_max_xs - int_min_xs, 0)
int_hs = np.maximum(int_max_ys - int_min_ys, 0)
int_areas = int_ws[:, np.newaxis] * int_hs
int_percents = int_areas / float(win_size[0]*win_size[1])
plt.imshow(int_percents)
plt.show()


