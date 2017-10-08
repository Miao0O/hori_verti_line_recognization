import os
import tensorflow as tf
from PIL import Image

# This is another model for image classification
# 1.convolution -> 1. ReLu -> 1. Maxpooling ->1. Dropout
# 2.convolution -> 2. ReLu -> 2. Maxpooling ->2. Dropout
# 3.convolution -> 3. ReLu -> 3. Maxpooling ->3. Dropout
# 4.convolution -> 4. ReLu -> 4. Maxpooling ->reshape

IMG_SIZE = 100
LABEL_CNT = 2
P_KEEP_INPUT = 0.8 #imput dropout layer rate
P_KEEP_HIDDEN = 0.5 #hidden layer droput layer rate

# get and initialize the weight
def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

X = tf.placeholder("float", [None, IMG_SIZE, IMG_SIZE])
Y = tf.placeholder("float", [None, 2])

w = init_weights([3, 3, 3, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([3, 3, 128, 128])
w5 = init_weights([4 * 4 * 128, 625])
w_o = init_weights([625, 2])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

#convolution model
def simple_model(X, w, w_2, w_3, w_4, w_5, w_o, p_keep_input, p_keep_hidden):
    # batchsize * 128 * 128 * 3
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    # 2x2 max_pooling
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # dropout
    l1 = tf.nn.dropout(l1, p_keep_input)  # 64 * 64 * 32

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w_2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_hidden)  # 32 * 32 * 64

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w_3, strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.nn.dropout(l3, p_keep_hidden)  # 16 * 16 * 128

    l4a = tf.nn.relu(tf.nn.conv2d(l3, w_4, strides=[1, 1, 1, 1], padding='SAME'))
    l4 = tf.nn.max_pool(l4a, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')  # 4 * 4 * 128
    l4 = tf.reshape(l4, [-1, w_5.get_shape().as_list()[0]])

    l5 = tf.nn.relu(tf.matmul(l4, w_5))
    l5 = tf.nn.dropout(l5, p_keep_hidden)

    return tf.matmul(l5, w_o)

#optimization
y_pred = simple_model(X, w, w2, w3, w4, w5, w_o, p_keep_input, p_keep_hidden) #y_pred is predicted tensor
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_pred))
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(y_pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

