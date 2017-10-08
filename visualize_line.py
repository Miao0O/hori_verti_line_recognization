import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
sess = tf.InteractiveSession()

def visualize_layer(tensor, shape):
    squeeze_tensor = tf.squeeze(tensor) # change the dimension of the tensor into 3
    visualize_image = sess.run(squeeze_tensor)
    reshape_image = np.reshape(visualize_image, shape)
    plt.imshow(reshape_image)
    plt.show()

# read the images by using unchanges method, so the channel will be 3
img = cv2.imread("/Users/miaoyan/Git/hori_verti_line_recognization/short_line/test/1/hor1.jpg", cv2.CV_LOAD_IMAGE_UNCHANGED)
plt.imshow(img)
plt.show()

# convert the number of image matrix into the range of 0 to 1 and related the ratio of the color
img = img.astype('float32')
img = (255-img)/255
plt.imshow(img)
plt.show()

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
visualize_layer(conv_hkernel, [8, 8, 3])

#   - vertical kernel
conv_vkernel = tf.nn.conv2d(input=input_layer, filter=tf.reshape(vkernel, [3, 3, 1, 1]),  # vertical kernel
    strides=[1, 1, 1, 1], padding='VALID')
visualize_layer(conv_vkernel, [8, 8, 3])

# Use tensorflow to do the max pooling
pool_hkernel = tf.nn.max_pool(tf.to_float(conv_hkernel, name='ToFloat'), ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                              padding='VALID')
visualize_layer(pool_hkernel, [6, 6, 3])

pool_vkernel = tf.nn.max_pool(tf.to_float(conv_vkernel, name='ToFloat'), ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                              padding='VALID')
visualize_layer(pool_vkernel, [6, 6, 3])

# add a convolution layer
conv_hkernel2 = tf.nn.conv2d(input=pool_hkernel, filter=tf.reshape(hkernel, [3, 3, 1, 1]), strides=[1, 1, 1, 1],
    padding='VALID') # shape: [3, 4, 4, 1]
visualize_layer(conv_hkernel2, [4, 4, 3])

conv_vkernel2 = tf.nn.conv2d(input=pool_vkernel, filter=tf.reshape(hkernel, [3, 3, 1, 1]), strides=[1, 1, 1, 1],
    padding='VALID') # shape: [3, 4, 4, 1]
visualize_layer(conv_vkernel2, [4, 4, 3])

# add a max pooling layer
pool_hkernel2 = tf.nn.max_pool(tf.to_float(conv_hkernel2, name='ToFloat'), ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1],
                              padding='VALID')
visualize_layer(pool_hkernel2, [1, 1, 3])

pool_vkernel2 = tf.nn.max_pool(tf.to_float(conv_vkernel2, name='ToFloat'), ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1],
                              padding='VALID')
visualize_layer(pool_vkernel2, [1, 1, 3])







