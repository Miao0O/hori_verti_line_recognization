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