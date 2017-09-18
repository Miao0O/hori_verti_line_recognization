import cv2
import numpy as np
import os
from skimage import transform
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from load_data import load_data
from tensorflow.contrib import learn
sess = tf.InteractiveSession()


def cnn_model_line(features, variable_line):
    # Define the 2 kinds of convolutional kernel
    hkernel = np.array((
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]), dtype="float32")

    vkernel = np.array((
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]), dtype="float32")

    # instead of fixed filter, we user variable filter
    filter_hor =  tf.reshape(hkernel, [3,3,1,1]) * variable_line
    filter_ver = tf.reshape(vkernel, [3,3,1,1]) * variable_line

    print(sess.run(filter_hor))
    print(sess.run(filter_ver))

    input_layer = tf.reshape(features, [-1, 10, 10, 1])

    conv_hkernel = tf.nn.conv2d(
        input=input_layer,
        filter= filter_hor,
        strides=[1, 1, 1, 1],
        padding='VALID')  # horizontal kernel

    # print(conv_hkernel)

    pool_hkernel = tf.nn.max_pool(tf.to_float(conv_hkernel, name='ToFloat'), ksize=[1, 8, 8, 1],
                                  strides=[1, 1, 1, 1], padding='VALID')

    conv_vkernel = tf.nn.conv2d(
        input=input_layer,
        filter=filter_ver,  # vertical kernel
        strides=[1, 1, 1, 1],
        padding='VALID')

    pool_vkernel = tf.nn.max_pool(tf.to_float(conv_vkernel, name='ToFloat'), ksize=[1, 8, 8, 1],
                                  strides=[1, 1, 1, 1], padding='VALID')

    # Classification of the image, horizontal line or vertical line
    # reshape the data of pool_hkernel and pool_vkernel
    result_hkernel = tf.squeeze(pool_hkernel)
    result_vkernel = tf.squeeze(pool_vkernel)

    softmax_result_hkernel = tf.cast(tf.div(result_hkernel, result_vkernel), dtype= tf.int32)
    softmax_result_vkernel = tf.cast(tf.div(result_vkernel, result_hkernel), dtype= tf.int32)

    print(sess.run(result_hkernel))
    print(sess.run(result_vkernel))

    print(sess.run(softmax_result_hkernel))
    print(sess.run(softmax_result_vkernel))

    result = tf.div(softmax_result_hkernel, tf.add(softmax_result_hkernel, softmax_result_vkernel))
    print(sess.run(result))

    #test_comparision = tf.greater(pool_hkernel, pool_vkernel)

    # vertical line has logits 0, horizontal line has logits 1
    # softmax of result_hkernel and result_vkernel


    # result = tf.cast(tf.greater(result_hkernel, result_vkernel), dtype=tf.float32)
    #probability_hor = tf.div(result_hkernel, tf.add(result_hkernel, result_vkernel))
    #probability_ver = tf.div(result_vkernel, tf.add(result_hkernel, result_vkernel))
    #print(sess.run(result))

    return result

def main(unused_argv):
    # Input data
    ROOT_PATH = "/Users/miaoyan/PycharmProjects/short_line/"
    train_data_dir = os.path.join(ROOT_PATH, "train")
    train_data_index, train_labels = load_data(train_data_dir) #read vertical-01 first, then read horizontal-10

    train_data = np.asarray(train_data_index)
    train_data = 1 - train_data
    train_data = train_data.astype('float32')

    train_labels = np.asarray(train_labels, dtype=np.float32) #01-vertical 10-horizontal

    # create variable for horizontal and vertical filters
    variable_line = tf.Variable(tf.random_uniform(shape=[3, 3, 1, 1], minval=1, maxval=2, dtype=tf.float32))
    sess.run(variable_line.initializer)

    # Use cnn_training_model
    y = cnn_model_line(train_data, variable_line) #output of CNNs

    # initialize and reshape the original labels
    y_= tf.placeholder(tf.float32, [None, 1]) #Initialization of the labels
    y_ = tf.reshape(train_labels, [-1]) # 01 is vertical line, 10 is horizontal line

    # Define loss and optimizer
    loss = None
    train_op =  None

    # mean square error
    onehot_labels_prediction = tf.one_hot(indices=y, depth=2)
    onehot_labels_label = tf.one_hot(indices=tf.cast(y_, dtype=tf.int32), depth=2)
    print(sess.run(onehot_labels_prediction))
    print(sess.run(onehot_labels_label))
    mse_loss = tf.losses.mean_squared_error(labels=onehot_labels_label, predictions=onehot_labels_prediction)
    print(sess.run(mse_loss))

    # Train the model
    # Create the Estimator
    line_classifier = learn.Estimator(model_fn=cnn_model_line,
        model_dir="/Users/miaoyan/PycharmProjects/hori_verti_line_recognization")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}

    # Prints the given tensors once every N local steps
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=2)

    # Fit the data for training the model
    line_classifier.fit(x=train_data, y=train_labels, batch_size=20, steps=1000, monitors=[logging_hook])

    # Configure the accuracy metric for evaluation
    metrics = {"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"), }

    train_op = tf.contrib.layers.optimize_loss(
        loss=mse_loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer='SGD') # use stochastic gradient descent as optimization algorithm
    print(sess.run(variable_line))


if __name__ == "__main__":
  tf.app.run()







