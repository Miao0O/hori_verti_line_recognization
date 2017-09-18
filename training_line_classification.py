# This program aims to use tensorflow to train the model
# In order to classify the line
#!/usr/bin/python
# -*- coding:utf8 -*-

import cv2
import glob
import numpy as np
import tensorflow as tf
import scipy
from skimage.exposure import rescale_intensity
import skimage.io as io
from skimage import data_dir
from glob import glob
from matplotlib import pyplot as plt
from scipy import ndimage
sess = tf.InteractiveSession()

# Initialize the required data
IMG_SIZE = 100
LABEL_CNT = 2
P_KEEP_INPUT = 0.8 #imput dropout layer rate
P_KEEP_HIDDEN = 0.5 #hidden layer droput layer rate

x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE])

# A Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations.
# It can be used and even modified by the computation.
# For machine learning applications, one generally has the model parameters be Variables.
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, dtype=tf.float32)

# get and initialize the weight
def weight_variable(shape):  # Weight Initialization
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, dtype=tf.float32)

# Convolution and Pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 3, 3, 1], padding='SAME')

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 2]) #label, 0-horizontal line, 1-vertical line

# First Convolution Layer
W_conv1 = weight_variable([3, 3, 1, 8]) #[filter_height, filter_width, in_channels, out_channels]
b_conv1 = bias_variable([8])
x_image = tf.reshape(x, [1,28,28, 1]) #[batch, in_height, in_width, in_channels]

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Densely connected Layer
W_fc1 = weight_variable([10 * 10 * 8, 16])
b_fc1 = bias_variable([16])

h_pool_flat = tf.reshape(h_pool1, [-1, 10 * 10 * 8])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([16, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  #Minimize error using cross entropy
  # Train and evaluate the modul
cross_entropy = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #is this gradient decent method?
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# read in the test image
img1 = cv2.imread("/Users/miaoyan/PycharmProjects/line/Training/00/t1.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
img1 = 255-img1
img1 = img1.astype(float)
img1 = img1/255
cv2.imshow('image', img1)

paths = glob( "/Users/miaoyan/PycharmProjects/line/Training/00/*")
imgs = []
for path in paths:
    img = cv2.imread(path)
    imgs.append(img)
y_label1 = [0,0]*len(imgs)

path = glob( "/Users/miaoyan/PycharmProjects/line/Training/01/*")
for path in paths:
    img = cv2.imread(path)
    imgs.append(img)
y_label2 = [0,1]*len(imgs)


sess.run(tf.global_variables_initializer())
for i in range(100):
    batch = tf.train.batch(x, batch_size=1)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
        x: batch[0], y_: batch[1], keep_prob: 1.0})
      print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


