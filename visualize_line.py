import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
sess = tf.InteractiveSession()

def visualize_layer(tensor, shape):
    squeeze_tensor = tf.squeeze(tensor) # change the dimension of the tensor into 3
    visualize_image = sess.run(squeeze_tensor)
    reshape_image = np.reshape(visualize_image, shape)
    plt.imshow(np.squeeze(reshape_image), cmap='gray')

# read the images into Greyscale image, so the channel of the image is 1
img = cv2.imread("/Users/miaoyan/Git/hori_verti_line_recognization/short_line/test/1/hor1.jpg", cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.title('Original Greyscale Image')
plt.show()

# Invert colors or image and change the intensity of the inverted greyscale image between 0 and 1
img = img.astype('float32')
img = (255-img)/255
plt.imshow(img, cmap='gray')
plt.title('Inverted Color')
plt.show()

# Define horizontal and vertical kernel for horizontal and vertical filter
hkernel = np.array(([0, 0, 0],
                    [1, 1, 1],
                    [0, 0, 0]), dtype="float32")
vkernel = np.array((
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]), dtype="float32")

# Use tensorflow to do the convolution
#   - horizontal kernel
input_layer = tf.reshape(img, [-1, 10, 10, 1])
conv_hkernel = tf.nn.conv2d(input=input_layer, filter=tf.reshape(hkernel, [3, 3, 1, 1]), strides=[1, 1, 1, 1],
    padding='VALID') # shape: [3, 8, 8, 1]
img_shape = tf.squeeze(conv_hkernel, axis=0).shape #get the shape of the to be visualized image
visualize_layer(conv_hkernel, img_shape)
plt.title('Convolution with Horizontal Kernel')
plt.show()
#   - vertical kernel
conv_vkernel = tf.nn.conv2d(input=input_layer, filter=tf.reshape(vkernel, [3, 3, 1, 1]),  # vertical kernel
    strides=[1, 1, 1, 1], padding='VALID')
img_shape = tf.squeeze(conv_vkernel, axis=0).shape #get the shape of the to be visualized image
visualize_layer(conv_vkernel, img_shape)
plt.title('Convolution with Vertical Kernel')
plt.show()

# Use tensorflow to do the max pooling
pool_hkernel = tf.nn.max_pool(tf.to_float(conv_hkernel, name='ToFloat'), ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                              padding='VALID')
img_shape = tf.squeeze(pool_hkernel, axis=0).shape #get the shape of the to be visualized image
visualize_layer(pool_hkernel, img_shape)
plt.title('Max Pooling after Convolution with Horizontal Kernel')
plt.show()

pool_vkernel = tf.nn.max_pool(tf.to_float(conv_vkernel, name='ToFloat'), ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                              padding='VALID')
img_shape = tf.squeeze(pool_vkernel, axis=0).shape #get the shape of the to be visualized image
visualize_layer(pool_vkernel, [6, 6, 1])
plt.title('Max Pooling after Convolution with Vertical Kernel')
plt.show()







