from os import listdir
from os.path import isfile, join, basename
import scipy.misc
import numpy as np

size = (360,480)
image_path = "C:/Datasets/Fisheries/train/ALL"
scale_path = "C:/Users/Yassin/PycharmProjects/FishyandTheGradDemons/scale.npy"

def save_scale():
    files = [file_name for file_name in listdir(image_path)]
    scale = {}
    for img in files:
        image = scipy.misc.imread(join(image_path, img))
        scale[img] = (image.shape[0]/size[0] , image.shape[1]/size[1])
    np.save(scale_path, scale)

dict_scale = np.load(scale_path).item()
print(dict_scale['img_06061.jpg'])