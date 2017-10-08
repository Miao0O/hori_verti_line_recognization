from keras.layers import Convolution2D, MaxPooling2D, Activation
from keras.models import Sequential
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import cv2  # only used for loading the image, you can use anything that returns the image as a np.ndarray

def visualize_cnns(model, features):
    # Keras expects batches of images, so we have to add a dimension to trick it into being nice
    # features_batch = np.expand_dims(features,axis=0)
    conv_features = model.predict(features_batch)
    conv_features = np.squeeze(conv_features, axis=0)
    print conv_features.shape
    plt.imshow(conv_features)
    plt.show()
