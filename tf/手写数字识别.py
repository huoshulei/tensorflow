import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

activation = tf.nn.tanh
h_layers = 7
h_hidden_units = 64
n_inputs = 28
n_steps = 28
n_classes = 10


def build_net():
    def add_layer(inputs, input_size, output_size, on_train=tf.Variable(False), activation_function=None,
                  norm=True):
        with tf.name_scope('layer'):
            weights = tf.Variable(tf.random_normal(tf.float32, [input_size, output_size], name='weights'))
            biases = tf.Variable(tf.zeros([1, output_size]) + 0.1, name='biases')
        wx_plus_b = tf.add(tf.matmul(input_data, weights), biases)
        if norm:
            if on_train.initial_value():
                fc_mean, fc_var = tf.nn.moments(wx_plus_b, axes=[0])
            scale = tf.Variable(tf.ones([output_size]))
            shift = tf.Variable(tf.zeros([output_size]))
            epsilon = 0.01
            ema = tf.train.ExponentialMovingAverage(decay=.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            mean, var = tf.cond(on_train,
                                mean_var_with_update,
                                lambda: (
                                    ema.average(fc_mean),
                                    ema.average(fc_var)
                                ))
            wx_plus_b = tf.nn.batch_normalization(wx_plus_b, mean, var, shift, scale, epsilon)
        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)
