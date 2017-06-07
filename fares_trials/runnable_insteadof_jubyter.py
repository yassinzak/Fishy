# Rather than importing everything manually, we'll make things easy
#   and load them all in utils.py, and just import them from there.
# %matplotlib inline
import utils;
from utils import *
import time
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

import ujson as json
anno_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft']
bb_json = {}
for c in anno_classes:
    j = json.load(open('{}/boxes/{}.json'.format(data_path, c), 'r'))
    for l in j:
        if 'annotations' in l.keys() and len(l['annotations'])>0:
            bb_json[l['filename'].split('/')[-1]] = sorted(
                l['annotations'], key=lambda x: x['height']*x['width'])[-1]

sizes = [PIL.Image.open(data_path+'train/'+f).size for f in filenames]
id2size = list(set(sizes))
size2id = {o:i for i,o in enumerate(id2size)}
raw_val_sizes = [PIL.Image.open(data_path+'valid/'+f).size for f in val_filenames]

file2idx = {o:i for i,o in enumerate(raw_filenames)}
val_file2idx = {o:i for i,o in enumerate(raw_val_filenames)}
empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}
for f in raw_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox
for f in raw_val_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox

bb_params = ['height', 'width', 'x', 'y']
def convert_bb(bb, size):
    bb = [bb[p] for p in bb_params]
    bb[0]= bb[0]*(224/size[0])
    bb[1]= bb[1]*(224/size[1])
    bb[2]= max(bb[2]*(224/size[0]),0)
    bb[3]= max(bb[3]*(224/size[1]),0)
    return bb
val_bbox = np.stack([convert_bb(bb_json[f], s) 
                   for f,s in zip(raw_val_filenames, raw_val_sizes)]).astype(np.float32)

trn_bbox = np.stack([convert_bb(bb_json[f],s) for f,s in zip(raw_filenames, sizes)]).astype(np.float32)

# val = get_data(data_path+'valid', (360,640))

def create_rect(bb, color='red'):
    plot.Rectangle((bb[3],bb[2]),bb[1],bb[0],color=color, fill=False, lw=3)
    
def show_bb(i):
    bb = val_bbox[i]
    print(cv2.resize(val,(360,640)).shape)
    img = val[i]
    plot(img)
    plt.gca().add_patch(create_rect(bb))

# Model
inp = Input(conv_layers[-1].output_shape[1:])
x = MaxPooling2D()(inp)
x = BatchNormalization(axis=1)(x)
x = Dropout(p/4)(x)
x = Flatten()(x)
i = x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p/2)(x)
x_bb = Dense(4, name='bb')(x)
x_class = Dense(8, activation='softmax', name='class')(x)

model = Model([inp], [x_bb, x_class])
model.compile(Adam(lr=0.00001), loss=['mse', 'categorical_crossentropy'], metrics=['accuracy'],
             loss_weights=[.001, 1.])

# trn_data=trn_data.transpose((0,3,1,2))
print(conv_feat.shape)

# model.fit(conv_feat, [trn_bbox, trn_labels], batch_size=16, nb_epoch=3, 
#              validation_data=(conv_val_feat, [val_bbox, val_labels]))

# model.save_weights(data_path+'models/bn_anno.h5')

# model.load_weights(data_path+'models/bn_anno.h5')

# model.fit(conv_feat, [trn_bbox, trn_labels], batch_size=16, nb_epoch=8, 
#              validation_data=(conv_val_feat, [val_bbox, val_labels]))

# model.save_weights(data_path+'models/bn_anno_11epochs.h5')

# model.load_weights(data_path+'models/bn_anno_11epochs.h5')

# model.fit(conv_feat, [trn_bbox, trn_labels], batch_size=16, nb_epoch=5, 
#              validation_data=(conv_val_feat, [val_bbox, val_labels]))

# model.save_weights(data_path+'models/bn_anno_16epochs.h5')

model.load_weights(data_path+'models/bn_anno_16epochs.h5')

model.fit(conv_feat, [trn_bbox, trn_labels], batch_size=16, nb_epoch=8, 
             validation_data=(conv_val_feat, [val_bbox, val_labels]))

model.save_weights(data_path+'models/bn_anno_24epochs.h5')

