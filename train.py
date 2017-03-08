from os import listdir
from os.path import isfile, join, basename
import scipy.misc
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

_classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
_path = 'C:/Datasets/Fisheries/'
_train_folder = 'train/'
_test_folder = "C:/Datasets/Fisheries/test_stg1"


def get_test_filenames():
    _test_file_names = [join(_test_folder, file_name) for file_name in listdir(_test_folder)]
    _test_file_names = [file_name for file_name in _test_file_names if isfile(file_name) and file_name.endswith('.jpg')]
    return _test_file_names

def get_dataset_filenames():
    train_path = join(_path, _train_folder)
    _train_file_names = []
    _train_file_classes = []
    _val_file_names = []
    _val_file_classes = []
    for idx, current_class in enumerate(_classes):
        class_path = join(train_path, current_class)
        class_files = [join(class_path, file_name) for file_name in listdir(class_path)]
        class_files = [file_name for file_name in class_files if isfile(file_name) and file_name.endswith('.jpg')]
        train_slice = int(len(class_files))
        _train_file_names += class_files[:train_slice]
        _train_file_classes += [idx] * train_slice
        _val_file_names += class_files[train_slice:]
        _val_file_classes += [idx] * (len(class_files) - train_slice)
        print("Found %i images of class %s" % (len(class_files), current_class))
    print("Total %i images found (%i train/ %i val)" % (len(_train_file_names) + len(_val_file_names), len(_train_file_classes), len(_val_file_names)))
    return _train_file_names, np.array(_train_file_classes, np.int64), _val_file_names, np.array(_val_file_classes, np.int64)

def load_images_to_memory(file_names, shape=(224, 336)):
    batch_shape = (len(file_names),) + shape + (3,)
    data = np.empty(batch_shape, dtype=np.uint8)
    for idx, file_name in enumerate(file_names):
        img = scipy.misc.imread(file_name)
        img = scipy.misc.imresize(img, shape)
        data[idx] = img
        print("Loaded %i/%i: %s" % (idx+1, len(file_names), file_name), end='\r')
    print()
    return data

@static_vars(count=0)
def conv_layer(input_tensor, shape, output_channels, strides=None, padding='VALID', activation=None, dropout=None, name_or_scope=None):
    conv_layer.count += 1
    with tf.variable_scope(name_or_scope or ("Conv_Layer_%i" % conv_layer.count)):
        input_channels = input_tensor.get_shape().as_list()[-1]
        filter = tf.get_variable('filter_weights', shape=shape+(input_channels, output_channels))
        bias = tf.get_variable('filter_bias', shape=(output_channels,), initializer=tf.constant_initializer(0))
        result = tf.nn.conv2d(input_tensor, filter, strides or ([1]*4), padding) + bias
        if activation is not None:
            result = activation(result)
        if dropout is not None:
            result = tf.nn.dropout(result, dropout)
    return result

@static_vars(count=0)
def linear_layer(input_tensor, output_size, activation=None, dropout=None, name_or_scope=None):
    linear_layer.count += 1
    with tf.variable_scope(name_or_scope or ("Linear_Layer_%i" % linear_layer.count)):
        input_size = input_tensor.get_shape().as_list()[-1]
        weights = tf.get_variable('layer_weights', shape=(input_size, output_size))
        bias = tf.get_variable('layer_bias', shape=(output_size,), initializer=tf.constant_initializer(0))
        result = tf.matmul(input_tensor, weights) + bias
        if activation is not None:
            result = activation(result)
        if dropout is not None:
            result = tf.nn.dropout(result, dropout)
    return result


class Model(object):
    def __init__(self):
        self.image = tf.placeholder(tf.float32, (None, 224, 336, 3), 'image')
        self.true_class = tf.placeholder(tf.int64, (None,), 'true_class')
        self.is_training = tf.placeholder(tf.bool, (), 'is_training')

        keep_prob = tf.cond(self.is_training, lambda: [tf.constant(0.5)], lambda: [tf.constant(1.0)])

        x = self.image

        x = conv_layer(x, (7, 7), 64, strides=[1, 2, 2, 1], activation=tf.nn.relu, dropout=keep_prob)
        x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

        for _ in range(3):
            x = conv_layer(x, (3, 3), 64, activation=tf.nn.relu, dropout=keep_prob)
            x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        features = tf.reduce_mean(x, reduction_indices=[1, 2])
        self.scores = linear_layer(features, len(_classes))

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.true_class, logits=self.scores))


def test():
    test_filenames = get_test_filenames()
    test_images = load_images_to_memory(test_filenames)
    image_mean = np.load("C:/Users/Yassin/PycharmProjects/FishyandTheGradDemons/npfish.npy")
    print(image_mean)
    images_data = test_images - image_mean
    result = np.empty((len(test_filenames), len(_classes)))
    model = Model()
    probability = tf.nn.softmax(model.scores)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "C:/Users/Yassin/PycharmProjects/FishyandTheGradDemons/model")
        i = 0
        for imdata in range(len(images_data)):
            feed_dict = {model.image: [images_data[imdata]],
                         model.is_training: False}
            result[i] = sess.run(probability, feed_dict)
            print("classify %d out of %d" % (i, len(images_data)), end='\r')
            i=i+1
    print(result)
    with open('results.csv', 'w') as out_file:
        out_file.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
        for idx, file_name in enumerate(test_filenames):
            file_name = basename(file_name.replace('\\', '/'))
            out_file.write(file_name+','+','.join([str(x) for x in result[idx]])+'\n')

def train():
    train_file_names, train_true_classes, val_file_names, val_true_classes = get_dataset_filenames()
    #random_batch = np.random.choice(len(file_names), 100, False)
    #file_names = [file_names[i] for i in random_batch]
    #true_classes = true_classes[random_batch]
    print('Loading Training Images')
    train_images = load_images_to_memory(train_file_names)
    print('Loading Validation Images')
    val_images = load_images_to_memory(val_file_names)
    # for i in range(10):
    #     print(file_names[i])
    #     plt.imshow(images[i])
    #     plt.show()

    image_mean = np.mean(np.mean(np.mean(train_images, 2), 1), 0)
    np.save("C:/Users/Yassin/PycharmProjects/FishyandTheGradDemons/npfish.npy", image_mean)
    model = Model()

    train_ops = tf.train.AdamOptimizer().minimize(model.loss)

    model.answer = tf.arg_max(model.scores, 1)
    model.accuracy = tf.reduce_mean(tf.to_float(tf.equal(model.true_class, model.answer)))

    batch_size = 32

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "C:/Users/Yassin/PycharmProjects/FishyandTheGradDemons/model")
        #sess.run(init_op)
        for epoch in range(100):
            batch_order = np.arange(len(train_images))
            np.random.shuffle(batch_order)
            total_loss, total_accuracy = 0, 0
            for i in range(math.ceil(len(train_images)/batch_size)):
                batch_start = i * batch_size
                batch_end = min(len(train_images), batch_size + batch_start)
                batch_indices = batch_order[batch_start: batch_end]
                feed_dict = {model.image: train_images[batch_indices] - image_mean,
                             model.true_class: train_true_classes[batch_indices],
                             model.is_training: True}
                loss, accuracy, _ = sess.run([model.loss, model.accuracy, train_ops], feed_dict)
                print('epoch %i iteration %i, loss=%f, accuracy=%f' % (epoch, i, float(loss), float(accuracy)), end='\r')
                total_loss += float(loss) * (batch_end - batch_start)
                total_accuracy += float(accuracy) * (batch_end - batch_start)
            print('val epoch %i total results, loss=%f, accuracy=%f' % (epoch, total_loss / len(train_images), total_accuracy / len(train_images)))

           # for i in range(math.ceil(len(val_images)/batch_size)):
            #    batch_start = i * batch_size
             #   batch_end = min(len(val_images), batch_size + batch_start)
              #  feed_dict = {model.image: val_images[batch_start:batch_end] - image_mean,
               #              model.true_class: val_true_classes[batch_start:batch_end],
                #             model.is_training: False}
                #loss, accuracy = sess.run([model.loss, model.accuracy], feed_dict)
               # print('val epoch %i iteration %i, loss=%f, accuracy=%f' % (epoch, i, float(loss), float(accuracy)), end='\r')
               # total_loss += float(loss) * (batch_end - batch_start)
               # total_accuracy += float(accuracy) * (batch_end - batch_start)
            #print('val epoch %i total results, loss=%f, accuracy=%f' % (epoch, total_loss/len(val_images), total_accuracy/len(val_images)))
            saver_path = saver.save(sess, "C:/Users/Yassin/PycharmProjects/FishyandTheGradDemons/model")


if __name__ == '__main__':
    operation = sys.argv[1]
    if operation == "train":
        train()
    else:
        test()


