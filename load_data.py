import os
import numpy as np
import matplotlib as plt
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import skimage
from skimage import transform
import skimage.io as io
import cv2

def load_data(data_dir):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".jpg")]
        for f in file_names:
            images.append(io.imread(f,as_grey=True))

            labels.append(int(d))



    return images, labels

#load training and testing datasets.

# ROOT_PATH = "/Users/miaoyan/PycharmProjects/line"
# train_data_dir = os.path.join(ROOT_PATH,"Training")
# test_data_dir = os.path.join(ROOT_PATH,"Testing")
#
#
# images, labels = load_data(train_data_dir)
#
#
#
#
# images32 = [transform.resize(image, (32, 32)) for image in images]
#
# for image in images[:5]:
#     print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))
#



