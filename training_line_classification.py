import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tempfile
from skimage import transform
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from load_data import load_data
sess = tf.InteractiveSession()

def weight_variable(shape):
    """weight variable generates a weight variable of a given shape"""
    initial = tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)
    return tf.Variable(initial)

def classify_model(x):
    """classify_model builds the graph for a deep net for classifying digits
    Args:
        x: an input tensorf with the dimensions (20, 100), where 100 is the number of pixels in a training image

    Returns:
        y is a tensor of shape (20, 2), with values equal to the label of the image
        (horizontal line is 1, vertical line is 0)
    """

    # Reshape to use within a convolutional neural net.
    # The last dimension is for "features", there is only one here, since images a re grayscale
    x_image = tf.reshape(x, [-1, 10, 10, 1])

    # convolutional layer
    w_hor = weight_variable([3, 3, 1, 1])
    w_ver = weight_variable([3, 3, 1, 1])
    # convolution with horizontal weight
    conv_hkernel = tf.nn.conv2d(
        input=x_image,
        filter=w_hor,
        strides=[1, 1, 1, 1],
        padding='VALID')  # horizontal kernel

    conv_vkernel = tf.nn.conv2d(
        input=x_image,
        filter=w_ver,
        strides=[1, 1, 1, 1],
        padding='VALID')
    # pooling layer
    pool_hkernel = tf.nn.max_pool(tf.to_float(conv_hkernel, name='ToFloat'), ksize=[1, 8, 8, 1],
                                  strides=[1, 1, 1, 1], padding='VALID')
    pool_vkernel = tf.nn.max_pool(tf.to_float(conv_vkernel, name='ToFloat'), ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1],
                                  padding='VALID')

    result_hkernel = tf.squeeze(pool_hkernel)
    result_vkernel = tf.squeeze(pool_vkernel)
    y_result = tf.reshape(tf.div(result_hkernel, tf.add(result_hkernel, result_vkernel)), [-1])

    return y_result, w_hor, w_ver

def main(unused_argv):
    ROOT_PATH =  "/Users/miaoyan/Dropbox/Git/hori_verti_line_recognization/short_line/"  # load training data
    train_data_dir = os.path.join(ROOT_PATH, "train")
    train_data_index, train_labels = load_data(train_data_dir) #read vertical-0 first, then read horizontal-1
    # pre-process training data
    train_data = np.asarray(train_data_index)
    train_data = 1 - train_data
    train_data = train_data.astype('float32')
    train_labels = np.asarray(train_labels, dtype=np.int32)  # 0-vertical 1-horizontal

    # load test data
    test_data_dir = os.path.join(ROOT_PATH, "test")
    eval_data_index, eval_labels = load_data(test_data_dir)
    # pre-process test data
    eval_data = np.asarray(eval_data_index)
    eval_data = 1 - eval_data
    eval_data = eval_data.astype('float32')
    eval_labels =  np.asarray(eval_labels, dtype=np.int32)

    # create the model
    x = tf.placeholder(tf.float32, [None, 10, 10])

    # define loss and optimizer
    y_label = tf.placeholder(tf.float32, [None])

    # BUild the graph for the convolutional neural network
    y_result, w_hor, w_ver = classify_model(x)

    # Define Loss
    loss = tf.losses.mean_squared_error(labels=y_label, predictions=y_result)

    # Define training
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss) # 1e-3 is learning rate

    #calculate accuracy
    # if y_result > 0.5, the image is classified as horizontal line, if y_result < 0.5, the image is classified as vertical line
    estimate_constant = tf.constant(0.5, dtype=tf.float32)
    correct_prediction = tf.equal(tf.cast(tf.cast(tf.add(y_result, estimate_constant), tf.int32), tf.float32), y_label)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # run graph by using session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        horizontal_kernel = np.squeeze(w_hor.eval())  # squeeze to one 2 dimensional matrix
        vertical_kernel = np.squeeze(w_ver.eval())  # squeeze to one 2 dimensional matrix
        plt.title('initial horizontal kernel')
        plt.imshow(horizontal_kernel, cmap='gray')
        plt.show()
        plt.title('initial vertical kernel')
        plt.imshow(vertical_kernel, cmap='gray')
        plt.show()
        for i in range (20000):
            train_step.run(feed_dict={x: train_data, y_label: train_labels})
            if i% 100 == 0:
                train_accuracy = accuracy.eval(feed_dict = {x: train_data, y_label: train_labels})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                # print('result:')
                # print y_result.eval()
                print('horizontal kernel:')
                print (w_hor.eval())
                print('vertical kernel:')
                print (w_ver.eval())

            if i == 19999:
                horizontal_kernel = np.squeeze(w_hor.eval()) # squeeze to one 2 dimensional matrix
                vertical_kernel = np.squeeze(w_ver.eval()) # squeeze to one 2 dimensional matrix
                plt.title('trained horizontal kernel')
                plt.imshow(horizontal_kernel, cmap='gray')
                plt.show()
                plt.title('trained vertical kernel')
                plt.imshow(vertical_kernel, cmap='gray')
                plt.show()

        #print the accuracy
        print ('test accuracy %g' % accuracy.eval(feed_dict= {x: eval_data, y_label: eval_labels}))


if __name__ == "__main__":
    tf.app.run()
