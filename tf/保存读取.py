import tensorflow as tf
import numpy as np

# remember to define the same dtype and shape when restore
# W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
# b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     save_path = saver.save(sess, 'net/save_net.ckpt')
#     print('save_path：', save_path)

W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'net/save_net.ckpt')
    print('Weights:', sess.run(W))
    print('biases:', sess.run(b))
