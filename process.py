from os import listdir
from os.path import isfile, join, basename
import numpy as np
import Imagescale as sc
import scipy.misc
import mask


main_folder = "C:/Users/Yassin/PycharmProjects/FishyandTheGradDemons/"
_classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
_path = 'C:/Datasets/Fisheries/'
_train_folder = 'train/'


def get_dataset_filenames():
    train_path = join(_path, _train_folder)
    file_names = []
    file_classes = []
    val_file_names = []
    val_file_classes = []
    for idx, current_class in enumerate(_classes):
        class_path = join(train_path, current_class)
        class_files = [join(class_path, file_name) for file_name in listdir(class_path)]
        class_files = [file_name for file_name in class_files if isfile(file_name) and file_name.endswith('.jpg')]
        train_slice = int(len(class_files)*0.7)
        file_names += class_files[:train_slice]
        file_classes += [idx] * train_slice
        val_file_names += class_files[train_slice:]
        val_file_classes += [idx] * (len(class_files) - train_slice)
    return file_names, np.array(file_classes, np.int64), val_file_names, np.array(val_file_classes,np.int64)

def load_images_to_memory(file_names, shape=sc.size):
    batch_shape = (len(file_names),) + shape + (3,)
    mask_batch_shape = (len(file_names),) + (shape[0]-mask.win_size[0],shape[1]-mask.win_size[1])
    data = np.empty(batch_shape, dtype=np.uint8)
    masks = np.empty(mask_batch_shape, dtype=bool)
    for idx, file_name in enumerate(file_names):
        img = scipy.misc.imread(file_name)
        img = scipy.misc.imresize(img, shape)
        data[idx] = img
        masks[idx] = mask.create_mask(basename(file_name.replace('\\', '/')))
        print("Loaded %i/%i: %s" % (idx+1, len(file_names), file_name), end='\r')
    return data, masks


files, classes, val_files, val_classes = get_dataset_filenames()
img, masks = load_images_to_memory(files)
val_img, val_masks = load_images_to_memory(val_files)
np.save(main_folder +'Train_classes.npy', classes)
np.save(main_folder +'val_classes.npy', val_classes)
np.save(main_folder +'Train_images.npy', img)
np.save(main_folder +'Train_masks.npy', masks)
np.save(main_folder +'val_images.npy', val_img)
np.save(main_folder +'val_masks.npy', val_masks)
