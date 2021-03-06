# Rather than importing everything manually, we'll make things easy
#   and load them all in utils.py, and just import them from there.
import utils;
from utils import *
import time
# from __future__ import division, print_function
import sys
sys.path.insert(1, '/home/mh/opencv-master/build/lib/python3')
import cv2
import glob
from keras.optimizers import Adadelta

data_path = '/home/mh/ws/fish_challenge/input/'
model_path = '/home/mh/ws/fish_challenge/input/models/'
batch_size=32

batches = get_batches(data_path+'train', batch_size=batch_size)
val_batches = get_batches(data_path+'valid', batch_size=batch_size*2, shuffle=False)
test_filenames = get_batches(data_path+'test', batch_size=batch_size).filenames

(val_classes, trn_classes, val_labels, trn_labels, 
    val_filenames, filenames, test_filenames) = get_classes(data_path)

raw_filenames = [f.split('/')[-1] for f in filenames]
raw_test_filenames = [f.split('/')[-1] for f in test_filenames]
raw_val_filenames = [f.split('/')[-1] for f in val_filenames]

## Load data.
trn_data = load_array(model_path+'train_data.bc')
val_data = load_array(model_path+'valid_data.bc')
trn_labels = load_array(model_path+'trn_labels.bc')
val_labels = load_array(model_path+'val_labels.bc')
val_labels = onehot(val_labels)
trn_labels = onehot(trn_labels)

test_data = load_array(model_path+'test_data.bc')
test_data = test_data.transpose((0,3,1,2))

from vgg16bn import Vgg16BN
model = vgg_ft_bn(8)

model.compile(optimizer=Adam(1e-3),
       loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights(data_path+'results/ft1.h5')

conv_layers,fc_layers = split_at(model, Convolution2D)
conv_model = Sequential(conv_layers)
#load features
conv_feat = load_array(data_path+'results/conv_feat.dat')
conv_val_feat = load_array(data_path+'results/conv_val_feat.dat')
conv_test_feat = load_array(data_path+'results/conv_test_feat.dat')

def get_bn_layers(p):
    return [
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        BatchNormalization(axis=1),
        Dropout(p/4),
        Flatten(),
        Dense(512,activation='relu'),
        BatchNormalization(),
        Dropout(p/2),
        Dense(512,activation='relu'),
        BatchNormalization(),
        Dropout(p/2),
        Dense(8,activation='softmax')
    ]

p=0.6
bn_model = Sequential(get_bn_layers(p))
bn_model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

bn_model.load_weights(data_path+'models/conv_512_6.h5')