# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

np.random.seed(1989)

import sys
sys.path.insert(1, '/home/mh/opencv-master/build/lib/python3')

import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings

import bcolz
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]

warnings.filterwarnings("ignore")

from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adagrad, Adam
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import np_utils
from keras.constraints import maxnorm
from sklearn.metrics import log_loss
from keras import __version__ as keras_version

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model


data_path='../input'

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (139, 139), cv2.INTER_LINEAR)
    return resized


def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('../input', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def load_test():
    path = os.path.join('../input', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


def read_and_normalize_train_data():
    train_data = load_array(data_path+'/train_data.bc')
    train_target = load_array(data_path+'/train_target.bc')
    train_id = load_array(data_path+'/train_id.bc')
    
#     train_data, train_target, train_id = load_train()

#     print(train_data[:10])
#     print('Convert to numpy...')
#     train_data = np.array(train_data, dtype=np.uint8)
#     train_target = np.array(train_target, dtype=np.uint8)

#     save_array(data_path+ '/train_data.bc', train_data)
#     save_array(data_path+ '/train_target.bc', train_target)
#     save_array(data_path+ '/train_id.bc', train_id)


    
    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, 8)
    
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id

def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()


    test_data = np.array(test_data, dtype=np.uint8)
    # test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    #     test_data = load_array(data_path+'test_data.bc')
#     test_id = load_array(data_path+'test_id.bc')

    save_array(data_path+ '/test_data.bc', test_data)
    save_array(data_path+ '/test_id.bc', test_id)

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id

def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def create_model():
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(139,139,3))

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(8, activation='softmax')(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for x in range(len(base_model.layers)):
        base_model.layers[x-3].trainable = False
        # print(layer)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv

def create_model():
    
    # create the base pre-trained model
    base_model = VGG16(weights='imagenet', include_top=False)
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(256, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(8, activation='softmax')(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable=False
    
    print(model.summary())


    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


def create_model_vgg16():
    # create the base pre-trained model
    base_model = VGG16(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer

    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(8, activation='softmax')(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    for layer in model.layers:
        layer.trainable = False

    # Get input
    new_input = model.input
    # Find the layer to connect
    hidden_layer = model.layers[-2].output
    # Connect a new layer on it
    new_output = Dense(256) (hidden_layer)
    new_output = Dense(8) (new_output)
    # Build a new model
    model2 = Model(new_input, new_output)
    # print(model.summary())
    print(model2.summary())

    # compile the model (should be done *after* setting layers to non-trainable)
    model2.compile(optimizer='adam', loss='categorical_crossentropy')
    return model2


def run_cross_validation_create_models(nfolds=10):
    # input image dimensions
    batch_size = 16
    nb_epoch = 1
    random_state = 51
    first_rl = 96

    train_data, train_target, train_id = read_and_normalize_train_data()

    yfull_train = dict()
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in kf:
        model = create_model()
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
            TensorBoard(log_dir='/tmp/tflogs', histogram_freq=0, write_graph=True, write_images=True)
        ]

        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
              callbacks=callbacks)

        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]
#         models[kf].load_weights(model_path+'Vgg16finetune1'+kf+'.h5')

        model.save_weights(data_path+'Vgg16finetune1'+str(num_fold)+'.h5')
        models.append(model)

    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)

#     cm = confusion_matrix(Y_valid, predictions_valid)
#     plot_confusion_matrix(cm, val_batches.class_indices)
#     print(cm)
    info_string = '_' + str(np.round(score,3)) + '_flds_' + str(nfolds) + '_eps_' + str(nb_epoch) + '_fl_' + str(first_rl)
    return info_string, models


def run_cross_validation_process_test(info_string, models):
    batch_size = 24
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_data, test_id = read_and_normalize_test_data()
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string \
                + '_folds_' + str(nfolds)
    create_submission(test_res, test_id, info_string)


if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    # model = create_model()
    num_folds = 3
    info_string, models = run_cross_validation_create_models(num_folds)
    run_cross_validation_process_test(info_string, models)
    # create_model_vgg16()