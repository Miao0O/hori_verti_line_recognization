import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.constant(2)
y = tf.constant(5)


def f1(): return tf.multiply(x, 17)


def f2(): return tf.add(y, 23)


r = tf.cond(tf.less(x, y), f1, f2)

print(sess.run(r))
