import cv2  # only used for loading the image, you can use anything that returns the image as a np.ndarray
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.models import Sequential

# This is a test function for visualization of cats through keras

cat = cv2.imread('cat.png') # don't need the path, if it is in the same path of the code
plt.imshow(cat)
plt.show()
print(cat.shape) #320, 400, 3

def visualize_cat(model, cat):
    # Keras expects batches of images, so we have to add a dimension to trick it into being nice
    cat_batch = np.expand_dims(cat,axis=0)
    conv_cat = model.predict(cat_batch)
    conv_cat = np.squeeze(conv_cat, axis=0)
    print conv_cat.shape
    plt.imshow(conv_cat)
    plt.show()

def nice_cat_printer(model, cat):
    '''prints the cat as a 2d array'''
    cat_batch = np.expand_dims(cat,axis=0)
    conv_cat2 = model.predict(cat_batch)

    conv_cat2 = np.squeeze(conv_cat2, axis=0)
    print conv_cat2.shape # 318, 398, 1

    conv_cat2 = conv_cat2.reshape(conv_cat2.shape[:2])
    print conv_cat2.shape # 318, 398
    plt.imshow(conv_cat2)
    plt.show()  

model = Sequential()

model.add(Conv2D(3, (3, 3), input_shape=cat.shape))
# model.add(Conv2D(3, (10, 10), input_shape=cat.shape))
# model.add(Conv2D(3, (3, 3), input_shape=cat.shape))
# model.add(Conv2D(3, (3, 3), activation='relu', padding='same', input_shape=cat.shape))

# model.add(Conv2D(1, (3, 3), input_shape=cat.shape))
# model.add(Conv2D(1, (15, 15), input_shape=cat.shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(1, (3, 3), input_shape=cat.shape))
# Lets activate then pool!
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

nice_cat_printer(model, cat)
# visualize_cat(model, cat)
print(cat.shape)  #320, 400, 3

