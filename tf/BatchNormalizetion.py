"""
https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf23_BN/tf23_BN.py
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

ACTIVATION = tf.nn.tanh
N_LAYERS = 5
N_HIDDEN_UNITS = 99


def fix_seed(seed=1):
    """伪随机数"""
    np.random.seed(seed)
    tf.set_random_seed(seed)


def plot_his(inputs, inputs_norm):
    for j, all_inputs in enumerate([inputs, inputs_norm]):
        for i, input in enumerate(all_inputs):
            plt.subplot(2, len(all_inputs), j * len(all_inputs) + i + 1)
            plt.cla()
            if not i:
                the_range = (-7, 10)
            else:
                the_range = (-1, 1)
            plt.hist(input.ravel(), bins=15, range=the_range, color='#FF5733')
            plt.yticks(())
            if j == 1:
                plt.xticks(the_range)
            else:
                plt.xticks(())
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
        plt.title('{} normalizing'.format('Without' if not j else 'With'))
    plt.draw()
    plt.pause(0.1)


def built_net(x, y, norm, on_train):
    def add_layer(inputs, in_size, out_size, activation_function=None, norm=False):
        weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=1.))
        biases = tf.Variable(tf.zeros([1, out_size]) + .1)
        wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)
        if norm:
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.01
            ema = tf.train.ExponentialMovingAverage(decay=.5)
            fc_mean, fc_var = 0, 0
            def mean_var_with_update():
                nonlocal fc_mean, fc_var
                fc_mean, fc_var = tf.nn.moments(wx_plus_b, axes=[0])
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
        return outputs

    fix_seed(1)
    if norm:
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
        epsilon = 0.001
        ema = tf.train.ExponentialMovingAverage(decay=.5)
        fc_mean, fc_var = 0, 0

        def mean_var_with_update():
            nonlocal fc_mean, fc_var
            fc_mean, fc_var = tf.nn.moments(x, axes=[0])
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)

        mean, var = tf.cond(on_train,
                            mean_var_with_update,
                            lambda: (
                                ema.average(fc_mean),
                                ema.average(fc_var)
                            ))
        x = tf.nn.batch_normalization(x, mean, var, shift, scale, epsilon)

    layers_inputs = [x]

    for l_n in range(N_LAYERS):
        layer_input = layers_inputs[l_n]
        in_size = layers_inputs[l_n].get_shape()[1].value

        output = add_layer(layer_input, in_size, N_HIDDEN_UNITS, ACTIVATION, norm)

        layers_inputs.append(output)
    """30 or 50"""
    prediction = add_layer(layers_inputs[-1], N_HIDDEN_UNITS, 1)

    cost = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), reduction_indices=[1]))
    train = tf.train.AdamOptimizer(0.001).minimize(cost)
    return [train, cost, layers_inputs, prediction]


fix_seed(1)
x_data = np.linspace(-7, 10, 3333)[:, np.newaxis]
np.random.shuffle(x_data)
noise = np.random.normal(0, 8, x_data.shape)
y_data = np.square(x_data) - 5 + noise

plt.scatter(x_data, y_data, s=1, c='r')
plt.show()

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
on_train = tf.placeholder(tf.bool)
train_op, cost, layers_inputs, prediction = built_net(xs, ys, norm=False, on_train=on_train)
train_op_norm, cost_norm, layers_inputs_norm, prediction_norm = built_net(xs, ys, norm=True, on_train=on_train)
cost_his = []
cost_norm_his = []
record_step = 5
plt.ion()
plt.figure(figsize=(7, 3))
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(3333):
        if not i % 33:
            all_inputs, all_inputs_norm = sess.run([layers_inputs, layers_inputs_norm],
                                                   feed_dict={xs: x_data, ys: y_data, on_train: True})
            plot_his(all_inputs, all_inputs_norm)
        sess.run([train_op, train_op_norm],
                 feed_dict={xs: x_data, ys: y_data, on_train: True})

        if not i % record_step:
            cost_his.append(sess.run(cost, feed_dict={xs: x_data, ys: y_data, on_train: True}))
            cost_norm_his.append(sess.run(cost_norm, feed_dict={xs: x_data, ys: y_data, on_train: True}))

    plt.ioff()
    plt.figure()
    plt.plot(np.arange(len(cost_his)) + record_step, np.array(cost_his), label="no BN")
    plt.plot(np.arange(len(cost_norm_his)) + record_step, np.array(cost_norm_his), label="BN")
    plt.legend()
    plt.show()

    pred, pred_norm = sess.run([prediction, prediction_norm], feed_dict={xs: x_data, ys: y_data, on_train: True})
    plt.scatter(x_data, pred, s=1, c='r')
    plt.show()
    plt.scatter(x_data, pred_norm, s=1, c='b')
    plt.show()
