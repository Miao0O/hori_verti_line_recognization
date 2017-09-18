import cv2
import numpy as np
from numpy import array
import os
from skimage import transform
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from load_data import load_data
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)
sess = tf.InteractiveSession()


def cnn_model_line(features, train_labels, mode):
    input_layer = tf.reshape(features, [-1, 10, 10, 1])

    hkernel = np.array(([0, 0, 0], [1, 1, 1], [0, 0, 0]), dtype="float32")

    vkernel = np.array(([0, 1, 0], [0, 1, 0], [0, 1, 0]), dtype="float32")

    # create variable for horizontal and vertical filters
    variable_line = tf.Variable(tf.random_uniform(shape=[3, 3, 1, 1], minval=1, maxval=2, dtype=tf.float32))
    # sess.run(variable_line.initializer)

    # instead of fixed filter, we user variable filter
    filter_hor = tf.reshape(hkernel, [3, 3, 1, 1]) * variable_line
    filter_ver = tf.reshape(vkernel, [3, 3, 1, 1]) * variable_line

    conv_hkernel = tf.nn.conv2d(
        input=input_layer,
        filter=filter_hor,
        strides=[1, 1, 1, 1],
        padding='VALID')  # horizontal kernel

    # print(conv_hkernel)
    pool_hkernel = tf.nn.max_pool(tf.to_float(conv_hkernel, name='ToFloat'), ksize=[1, 8, 8, 1],
                                  strides=[1, 1, 1, 1], padding='VALID')

    conv_vkernel = tf.nn.conv2d(
        input=input_layer,
        filter=filter_ver,
        strides=[1, 1, 1, 1],
        padding='VALID')

    pool_vkernel = tf.nn.max_pool(tf.to_float(conv_vkernel, name='ToFloat'), ksize=[1, 8, 8, 1],
                                  strides=[1, 1, 1, 1], padding='VALID')

    # Classification of the image, horizontal line or vertical line
    # reshape the data of pool_hkernel and pool_vkernel
    result_hkernel = tf.squeeze(pool_hkernel)
    result_vkernel = tf.squeeze(pool_vkernel)

    softmax_result_hkernel = tf.div(result_hkernel, tf.add(result_hkernel, result_vkernel))
    softmax_result_vkernel = tf.div(result_vkernel, tf.add(result_hkernel, result_vkernel))

    result = tf.div(softmax_result_hkernel, tf.add(softmax_result_hkernel, softmax_result_vkernel))


    y_ = tf.reshape(train_labels, [-1, 20]) #labels
    # y = tf.reshape(result, [-1, 20]) #logits

    loss = None
    train_op = None

    if mode != learn.ModeKeys.INFER:
        onehot_labels_prediction = tf.one_hot(indices=tf.cast(result, dtype=tf.int32), depth=2)
        onehot_labels_label = tf.one_hot(indices=tf.cast(y_, dtype=tf.int32), depth=2)
        mse_loss = tf.losses.mean_squared_error(labels=onehot_labels_label, predictions=onehot_labels_prediction)
        # mse_loss = tf.losses.mean_squared_error(labels=y_, predictions=y)
        # print(sess.run(mse_loss))

    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss = mse_loss,
            global_step = tf.contrib.framework.get_global_step(),
            learning_rate = 0.001,
            optimizer = "SGD")

    # Generate Predictions
    predictions = {"classes": tf.argmax(  # argmax can extract the predicted class
        input=result, axis=1), "probabilities": softmax_result_hkernel}

  # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=mse_loss, train_op=train_op)


def main(unused_argv):
    # Input data
    ROOT_PATH = "/Users/miaoyan/PycharmProjects/short_line/"
    train_data_dir = os.path.join(ROOT_PATH, "train")
    train_data_index, train_labels = load_data(train_data_dir)

    train_data = np.asarray(train_data_index)
    train_data = 1 - train_data
    train_data = train_data.astype('float32')

    train_labels = np.asarray(train_labels, dtype=np.float32) #01 ist vertical, 10 ist horizontal

    test_data_dir = os.path.join(ROOT_PATH, "test")
    eval_data_index, eval_labels = load_data(test_data_dir)

    eval_data = np.asarray(eval_data_index)
    eval_data = 1 - eval_data
    eval_data = eval_data.astype('float32')

    eval_labels = np.asarray(eval_labels, dtype=np.float32)
    # initialize and reshape the original labels

    y_ = tf.placeholder(tf.float32, [None, 1])  # Initialization of the labels
    y_ = tf.reshape(train_labels, [-1])  # 01 is vertical line, 10 is horizontal line


    x = tf.placeholder(tf.float32, shape = [None, 10, 10, 1])
    x = tf.reshape(train_data, [-1, 10, 10, 1])

    # Use cnn_training_model
    # cnn_model_line(train_data, train_labels) #output of CNNs

    # y_= tf.placeholder(tf.float32, [None, 20]) #Initialization of the labels
     # 0 is vertical line, 1 is horizontal line
    # sess.run(train_op, feed_dict={x: train_data, y_: train_labels})
    # print(sess.run(mse_loss, feed_dict={x: train_data, y_: train_labels}))

    # Create the Estimator
    line_classifier = learn.Estimator(model_fn=cnn_model_line, model_dir="/Users/miaoyan/PycharmProjects/hori_verti_line_recognization")
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    # tensors_to_log = {"probabilities": "softmax_tensor"}

    # Prints the given tensors once every N local steps
    # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=2)

    # Fit the data for training the model
    line_classifier.fit(x, y_, batch_size=20, steps=10000)

    # Configure the accuracy metric for evaluation
    metrics = {"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"), }

    # Evaluate the model and print results
    eval_results = line_classifier.evaluate(train_data, train_labels, metrics=metrics)
    print(eval_results)


if __name__ == "__main__":
  tf.app.run()
