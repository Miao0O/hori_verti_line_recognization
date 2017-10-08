
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from hori_verti_line_recognization.load_data import load_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from skimage import transform

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

# This is a very good example of MNIST, we can imitate this function to generate our own CNN model

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features, [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32, # number of the filter kernel
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  # Flatten tensor into a batch of vectors, so that the tensor has only 2 dimensions
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])


  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu) #the number of neurons in the dense layer (1024)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN) #40% of the elements will be randomly dropped out during training.

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 2]
  logits = tf.layers.dense(inputs=dropout, units=2) #one for each target 2 classes

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
      onehot_labels = tf.one_hot(indices=labels, depth=2) #10-hor 01-ver
      loss = tf.losses.softmax_cross_entropy(
          onehot_labels=onehot_labels, logits=logits) # takes onehot_labels and logits as arguments, performs softmax activation on logits
      #returns our loss as a scalar Tensor

######################################################
  # Configure the Training Op (for TRAIN mode)
  # configure our model to optimize loss value during training
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD") #use stochastic gradient descent as optimization algorithm

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(              # argmax can extract the predicted class
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):


  # load training and eval datasets.
  ROOT_PATH = "/Users/miaoyan/PycharmProjects/line"
  train_data_dir = os.path.join(ROOT_PATH, "Training")
  test_data_dir = os.path.join(ROOT_PATH, "Testing")

  train_data_index  , train_labels = load_data(train_data_dir)
  train_data_index = [transform.resize(image, (28, 28)) for image in train_data_index]
  train_data_index = [image.astype(np.float32)for image in train_data_index]
  train_labels = np.asarray(train_labels, dtype=np.int32)
  train_data = np.asarray(train_data_index)

  eval_data_index, eval_labels = load_data(test_data_dir)

  eval_labels = np.asarray(eval_labels, dtype=np.int32)

  eval_data_index = [transform.resize(image, (28, 28)) for image in eval_data_index]
  eval_data_index = [image.astype(np.float32)for image in eval_data_index]
  eval_data = np.asarray(eval_data_index)

###############################################################################
  # Create the Estimator
  line_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="/Users/miaoyan/PycharmProjects/hori_verti_line_recognization")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}

  # Prints the given tensors once every N local steps
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=2)

  # Fit the data for training the model
  line_classifier.fit(
      x=train_data,
      y=train_labels,
      batch_size=20,
      steps=1000,
      monitors=[logging_hook])

  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }

  # Evaluate the model and print results
  eval_results = line_classifier.evaluate(
      x=eval_data, y=eval_labels, metrics=metrics)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()