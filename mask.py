import scipy.misc
import Imagescale as sc
import matplotlib.pyplot as plt
import numpy as np
import json
from os.path import join

_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft']
_json_path = 'C:/Datasets/Fisheries/train/Annotations'
_image_path = 'C:/Datasets/Fisheries/train/ALL'
win_size = (10, 10)
image_size = sc.size

def read_json():
    json_dic = {}
    for _class in _classes:
        with open(join(_json_path, _class+'.json')) as data_file:
            data = json.load(data_file)
            for x in range(len(data)):
                json_dic[data[x]['filename']] = data[x]
    return json_dic

json_dic = read_json()

# name of the image
# i the number of bounding boxes
def create_mask(name):
    h = []
    w = []
    x = []
    y = []
    for i in range(len(json_dic[name]['annotations'])):
        h.append(json_dic[name]['annotations'][i]['height'])
        w.append(json_dic[name]['annotations'][i]['width'])
        x.append(json_dic[name]['annotations'][i]['x'])
        y.append(json_dic[name]['annotations'][i]['y'])
    return preprocessing(h, w, x, y, name)

def masking(height, width, fish_x, fish_y, name):
    scale = sc.dict_scale[name]
    y = fish_y / scale[0]
    x = fish_x / scale[1]
    h = height / scale[0]
    w = width / scale[1]
    fish_window = ((y, x), (y+h, x+w))
    min_xs = np.arange(image_size[0] - win_size[0])
    min_ys = np.arange(image_size[1] - win_size[1])
    max_xs = min_xs + win_size[0]
    max_ys = min_ys + win_size[1]
    int_min_xs = np.maximum(min_xs, fish_window[0][0])
    int_min_ys = np.maximum(min_ys, fish_window[0][1])
    int_max_xs = np.minimum(max_xs, fish_window[1][0])
    int_max_ys = np.minimum(max_ys, fish_window[1][1])
    int_ws = np.maximum(int_max_xs - int_min_xs, 0)
    int_hs = np.maximum(int_max_ys - int_min_ys, 0)
    int_areas = int_ws[:, np.newaxis] * int_hs
    int_percents = np.floor(int_areas / float(win_size[0] * win_size[1]))
    return int_percents

def preprocessing(height, width, fish_x, fish_y, name):
    if not fish_x:
        return np.zeros(shape=(image_size[0]-win_size[0], image_size[1]-win_size[1]), dtype=float)
    img_mask = np.zeros(shape=(image_size[0]-win_size[0], image_size[1]-win_size[1]), dtype=float)
    for i in range(len(height)):
        img_mask += masking(height[i], width[i], fish_x[i], fish_y[i], name)
    #np.pad(img_mask, ((20, 20), (20, 20)), 'constant', constant_values=((0, 0), (0, 0)))
    return img_mask.astype(bool).astype(float)

def mask_decode(mask):
    mask_image = np.zeros(shape=image_size, dtype=np.int32)
    delta_shape = [image_size[i]-mask.shape[i] for i in range(2)]
    mask_image[:mask.shape[0], :mask.shape[1]] = mask
    mask_image = np.cumsum(mask_image, 0)
    mask_image = np.cumsum(mask_image, 1)
    final_mask = np.copy(mask_image)
    final_mask[delta_shape[0]:, delta_shape[1]:] += mask_image[:mask.shape[0], :mask.shape[1]]
    final_mask[delta_shape[0]:, :] -= mask_image[:mask.shape[0], :]
    final_mask[:, delta_shape[1]:] -= mask_image[:, :mask.shape[1]]
    # for x in range(mask.shape[0]):
    #     for y in range(mask.shape[1]):
    #         if mask[x, y]:
    #             mask_image[x:x+win_size[0], y:y+win_size[1]] = 1
    return final_mask != 0

def image_segment(image,mask):
    my_mask = mask_decode(mask)
    return image * my_mask[:, :, None]


my_mask = create_mask('img_00003.jpg')
img = scipy.misc.imread(join(_image_path, 'img_00003.jpg'))
img = scipy.misc.imresize(img, image_size)
# mask_image = mask_decode(my_mask)
# plt.imshow(img)
# plt.imshow(mask_image, alpha=0.5)
# plt.show()
seg = image_segment(img, my_mask)
plt.imshow(seg[:,:,:])
plt.show()
