import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import transform
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from load_data import load_data
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
sess = tf.InteractiveSession()

# Description:
# This program is using tensorflow to train the function cnn_model_line in order to classify,
# if the image is horizontal line or vertical line.
# The training images and testing images are saved in the short_line folder and loaded by the function load_data

# Algorithm:
# We use tf.Variable to generate a variable matrix, which has the shape of [3, 3, 1, 1]
# For using of horizontal kernel, we multiply the h_kernel with the generated variable
# For using of vertical kernel, we multiply the v_kernel with the generated variable
# So these 2 filter kernels all have the shape [3, 3, 1, 1]

# Then we use these 2 kinds of filters to convolute the features
# After that we use the same size of kernel [1, 8, 8, 1] to do pooling of the to convoluted features
# Due the the size of the feature [-1, 10, 10, 1], after the convolution and pooling we can get the two results which have the size [-1, 1, 1, 1]

# We do squeeze to have a single value for both horizontal result and vertical result
# Then we can calculate the probability of the horizontal line by using of the equation horizontal result/ (vertical result + horizontal result)

# Because of the feature matrix is preprocessed into the numbers between 0 and 1, the bigger the value, the darker the color
# So the pixels, which demonstrate the line have more than 0.5 values.
# After the pooling, for horizontal line, we can get a single value bigger thant 0.5, for vertical line, we can get a single value smaller thant 0.5
# Labels of the horizontal line are 1, labels of the vertical line are 0

# We calculate the loss, which is mean square error between the result and the labels
# Then, during each training, we use "SGD" optimization method to reduce the loss

# After the training, we input the testing data to classify the images by using of the pre-trained cnn model
# For each imported image, we can get the results and print the classification result, weather horizontal line or vertical line

def cnn_model_line(features, labels, mode):
    # Define the 2 kinds of convolutional kernel
    hkernel = np.array((
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]), dtype="float32")
    vkernel = np.array((
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]), dtype="float32")
    # create variable for horizontal and vertical filters
    variable_line = tf.Variable(tf.random_uniform(shape=[3, 3, 1, 1], minval=1, maxval=2, dtype=tf.float32))
    # sess.run(variable_line.initializer)
    # instead of fixed filter, we user variable filter
    filter_hor =  tf.reshape(hkernel, [3,3,1,1]) * variable_line
    filter_ver = tf.reshape(vkernel, [3,3,1,1]) * variable_line
    # print(sess.run(filter_hor))
    # print(sess.run(filter_ver))
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
    result = tf.reshape(tf.div(result_hkernel,tf.add(result_hkernel, result_vkernel)), [-1])

    labels = tf.reshape(labels, [-1])

    # Define loss and optimizer
    loss = None
    train_op =  None

 # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels=labels, predictions=result)


    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer='SGD'
        )  # use stochastic gradient descent as optimization algorithm

  # Generate Predictions
    predictions = {
      "probability": result}

  # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op= train_op)


def main(unused_argv):
    # Input data
    ROOT_PATH = "/Users/miaoyan/Git/hori_verti_line_recognization/short_line/"
    train_data_dir = os.path.join(ROOT_PATH, "train")
    train_data_index, train_labels = load_data(train_data_dir) #read vertical-01 first, then read horizontal-10

    train_data = np.asarray(train_data_index)
    train_data = 1 - train_data
    train_data = train_data.astype('float32')
    train_labels = np.asarray(train_labels, dtype=np.int32) #01-vertical 10-horizontal

    # test data
    test_data_dir = os.path.join(ROOT_PATH, "test")
    eval_data_index, eval_labels = load_data(test_data_dir)

    eval_data = np.asarray(eval_data_index)
    eval_data = 1 - eval_data
    eval_data = eval_data.astype('float32')
    eval_labels =  np.asarray(eval_labels, dtype=np.int32)

    # Use cnn_training_model
    # y = cnn_model_line(train_data, variable_line) #output of CNNs

    # initialize and reshape the original labels
    # y_= tf.placeholder(tf.float32, [None, 1]) #Initialization of the labels
    # y_ = tf.reshape(train_labels, [-1]) # 01 is vertical line, 10 is horizontal line

    # Train the model
    # Create the Estimator
    line_classifier = learn.Estimator(model_fn=cnn_model_line,
        model_dir="/Users/miaoyan/PycharmProjects/hori_verti_line_recognization/training_data")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    #tensors_to_log = {"probabilities"}

    # Prints the given tensors once every N local steps
    #logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

    # Fit the data for training the model
    line_classifier.fit(x=train_data, y=train_labels , batch_size=20, steps=1000)



    # Score Accuracy
    ev=line_classifier.evaluate(x=eval_data, y=eval_labels, steps=1)
    loss_score = ev["loss"]
    print("Loss: %s" % loss_score)

    # Print out predictions
    predictions = line_classifier.predict(x=eval_data, as_iterable=True)
    for i, p in enumerate(predictions):
        print("Prediction %s: %s" % (i+1, p["probability"]))
        if p["probability"] >=0.5:
            print ("horizontal line")
        else:
            print("vertical line")

if __name__ == "__main__":
  tf.app.run()
