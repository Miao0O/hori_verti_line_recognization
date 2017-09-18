# make TFRecord file, label each class of images

import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cwd = '/Users/miaoyan/PycharmProjects/hori_verti_line_recognization/line/'
classes={'horizontal','vertical'}
writer= tf.python_io.TFRecordWriter("line_train.tfrecords")

for index,name in enumerate(classes):
    class_path = cwd + name+  '/'
    for img_name in os.listdir(class_path):
        img_path=class_path+img_name

        img=Image.open(img_path)

        img= img.resize((50,50))
        img_raw=img.tobytes() #make images into raw data
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())


writer.close()

''''' Read the TFRecord
for serialized_example in tf.python_io.tf_record_iterator("line_train.tfrecords"):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    image = example.features.feature['image'].bytes_list.value
    label = example.features.feature['label'].int64_list.value
    print image, label
'''

# Use queue read the TFRecord to conviniently used in TF
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read (filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8) #each R,G,B has 8 bit
    img = tf.reshape(img, [100, 100, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #normalize
    label = tf.cast(features['label'], tf.int32)

    return img, label


img, label = read_and_decode("line_train.tfrecords")

img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=30, capacity=2000,
                                                min_after_dequeue=1000)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(3):
        val, l= sess.run([img_batch, label_batch])

        print(val.shape, l)