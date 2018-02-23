import tensorflow as tf
import numpy as np

# dropout 激活函数
# a = tf.constant([[-1., 2., 3., 4.]])
# with tf.Session() as sess:
#     b = tf.nn.dropout(a, .5, noise_shape=[1, 4])
#     print(sess.run(b))
#     b = tf.nn.dropout(a, .5, noise_shape=[1, 1])
#     print(sess.run(b))
# conv2d 卷积函数
input_data = tf.Variable(np.random.rand(1, 5, 5, 1), dtype=tf.float32)
filter_data = tf.Variable(np.random.rand(3, 3, 1, 3), dtype=tf.float32)
y = tf.nn.atrous_conv2d(input_data, filter_data, 2, padding='SAME')
print(tf.shape(y))
print(y.shape)
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print('input_data',sess.run(input_data))
#     print('filter_data',sess.run(filter_data))
#     print('y',sess.run(y))