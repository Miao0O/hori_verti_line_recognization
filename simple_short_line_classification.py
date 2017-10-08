# First program of the line classification
# This program aims to classify the imported images into 2 classes,
# that is horizontal line or vertical line
# We use CNN models but with constant hkernel and vkernel to do 2D convolution with features of the imported figure
# After pooling of horizontal filter and vertical filter  , we compare the result of 2 kinds pooling result
# if the result of horizontal is largert than the vertical result, then the image is horizontal line, and vice versa.

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
sess = tf.InteractiveSession()

# Define the CNN model
def cnn_model_line(features):
    # Define the 2 kinds of convolutional kernel
    hkernel = np.array((
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]), dtype="float32")

    vkernel = np.array((
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]), dtype="float32")

    input_layer = tf.reshape(features, [-1, 10, 10, 1])

    conv_hkernel = tf.nn.conv2d(
        input=input_layer,
        filter = tf.reshape(hkernel, [3,3,1,1]),
        strides=[1, 1, 1, 1],
        padding='VALID')  # horizontal kernel

    #print(conv_hkernel)

    pool_hkernel =  tf.nn.max_pool(tf.to_float(conv_hkernel,name='ToFloat'), ksize=[1,8,8,1],  strides=[1, 1, 1, 1], padding='VALID')

    conv_vkernel = tf.nn.conv2d(
        input=input_layer,
        filter = tf.reshape(vkernel, [3,3,1,1]),  # vertical kernel
        strides=[1, 1, 1, 1],
        padding='VALID')

    pool_vkernel = tf.nn.max_pool(tf.to_float(conv_vkernel, name='ToFloat'), ksize=[1,8,8,1],  strides=[1, 1, 1, 1], padding='VALID')

    # Classification of the image, horizontal line or vertical line
        # reshape the data of pool_hkernel and pool_vkernel
    result_hkernel = tf.squeeze(pool_hkernel)
    result_vkernel = tf.squeeze(pool_vkernel)

    result = tf.cond(tf.greater(result_hkernel, result_vkernel), lambda: tf.ones([]), lambda: tf.zeros([]))
    return result

def main(unused_argv):
    # read 2 images, the one is horizontal line, another is vertical line
    # img1 is horizontal line
    # This folder just include 2 images, a horizontal line image and a vertical line image.
    img1 = cv2.imread("/Users/miaoyan/Git/hori_verti_line_recognization/short_line/test/1/hor1.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img1 = 255 - img1
    img1 = img1.astype('float32')
    img1 = img1 / 255

    # img2 is vertical line
    img2 = cv2.imread("/Users/miaoyan/Git/hori_verti_line_recognization/short_line/test/0/ver1.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img2 = 255 - img2
    img2 = img2.astype('float32')
    img2 = img2 / 255

    test_image = img1
    cv2.imshow('input_image', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    result = sess.run(cnn_model_line(test_image)) #input weather img1 or img2

    if result ==1:
        print("The input image is a horizontal line")
    elif result ==0:
        print("The input image is a vertical line")


if __name__ == "__main__":
  tf.app.run()