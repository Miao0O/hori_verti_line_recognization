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
#



