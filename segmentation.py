from os import listdir
from os.path import isfile, join, basename
import scipy.misc
import math
import numpy as np
import tensorflow as tf


_classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def get_dataset():
    train_true_classes = np.load("Train_masks.npy")
    train_images = np.load("Train_images.npy")
    print("Training dataset has been loaded")
    val_true_classes = np.load("val_masks.npy")
    val_images = np.load("val_images.npy")
    print("validation dataset has been loaded")
    return train_images, train_true_classes, val_images, val_true_classes


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
        self.image = tf.placeholder(tf.float32, (None, 360, 480, 3), 'image')
        self.true_class = tf.placeholder(tf.float32, (None, 341, 461), 'true_class')
        self.is_training = tf.placeholder(tf.bool, (), 'is_training')

        keep_prob = tf.cond(self.is_training, lambda: [tf.constant(0.8)], lambda: [tf.constant(1.0)])

        x = self.image

        for _ in range(3):
            x = conv_layer(x, (3, 3), 16, strides=[1, 1, 1, 1], activation=tf.nn.relu, dropout=keep_prob)
            x = conv_layer(x, (3, 3), 16, strides=[1, 1, 1, 1], activation=tf.nn.relu, dropout=keep_prob)
            x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 1, 1, 1], 'VALID')

        x = conv_layer(x, (3, 3), 16, activation=tf.nn.relu, dropout=keep_prob)
        x = conv_layer(x, (3, 3), 16, activation=tf.nn.relu, dropout=keep_prob)
        x = conv_layer(x, (1, 1), 1, activation=None, dropout=None)
        x = tf.squeeze(x, axis=-1)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.true_class, logits=x, name="Loss")
        self.loss = tf.reduce_mean(loss, reduction_indices=[0, 1, 2])
        self.score = tf.nn.sigmoid(x)


def train():
    train_images, train_true_classes, val_images, val_true_classes = get_dataset()

    image_mean = np.mean(np.mean(np.mean(train_images, 2), 1), 0)
    np.save("npfish.npy", image_mean)
    model = Model()

    train_ops = tf.train.AdamOptimizer().minimize(model.loss)

    batch_size = 4

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #saver.restore(sess, "logs/model")
        sess.run(init_op)
        for epoch in range(100):
            batch_order = np.arange(len(train_images))
            np.random.shuffle(batch_order)
            total_loss = 0
            for i in range(math.ceil(len(train_images)/batch_size)):
                batch_start = i * batch_size
                batch_end = min(len(train_images), batch_size + batch_start)
                batch_indices = batch_order[batch_start: batch_end]
                feed_dict = {model.image: train_images[batch_indices] - image_mean,
                             model.true_class: train_true_classes[batch_indices],
                             model.is_training: True}
                loss,  _ = sess.run([model.loss,  train_ops], feed_dict)
                print('epoch %i iteration %i, loss=%f' % (epoch, i, float(loss)), end='\r')
                total_loss += float(loss) * (batch_end - batch_start)
            print('train epoch %i total results, loss=%f' % (epoch, total_loss / len(train_images)))
            total_loss = 0
            for i in range(math.ceil(len(val_images)/batch_size)):
                batch_start = i * batch_size
                batch_end = min(len(val_images), batch_size + batch_start)
                feed_dict = {model.image: val_images[batch_start:batch_end] - image_mean,
                             model.true_class: val_true_classes[batch_start:batch_end],
                             model.is_training: False}
                loss = sess.run([model.loss], feed_dict)
                print('val epoch %i iteration %i, loss=%f' % (epoch, i, float(loss)), end='\r')
                total_loss += float(loss) * (batch_end - batch_start)
            print('val epoch %i total results, loss=%f' % (epoch, total_loss/len(val_images)))
            saver_path = saver.save(sess, "logs/model")


if __name__ == '__main__':
    train()



