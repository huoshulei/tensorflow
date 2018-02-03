import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    layer_name = 'layer'
    with tf.name_scope('layer'):
        with tf.name_scope('W'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name="W")
            tf.summary.histogram(layer_name + '/w', Weights)
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        with tf.name_scope('Wx_puls_b'):
            Wx_puls_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            output = Wx_puls_b
        else:
            output = activation_function(Wx_puls_b)
        return output


x_data = np.linspace(-2, 2, 666)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
with tf.name_scope('in_put'):
    xs = tf.placeholder(tf.float32, [None, 1], name='xs')
    ys = tf.placeholder(tf.float32, [None, 1], name='ys')

l1 = add_layer(xs, 1, 20, tf.nn.relu)
# l2 = add_layer(l1, 20, 20, tf.nn.relu)
# l3 = add_layer(l2, 40, 20, tf.nn.relu)
prediction = add_layer(l1, 20, 1)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data, 0.5, 'g')
plt.ion()
plt.show()

with tf.Session() as sess:
    merge_all = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(999):
        _, l = sess.run([train, loss], feed_dict={xs: x_data, ys: y_data})
        if not i % 5:
            run = sess.run(merge_all, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(run, i)
            print('训练损失0.05：{:>10.9f}'.format(l))
            try:
                ax.lines.remove(plot[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            plot = ax.plot(x_data, prediction_value, 'r-', lw=1)
            plt.pause(0.005)
            # if not i % 30000:
            #     print(prediction_value)
    train = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
    for i in range(999):
        _, l = sess.run([train, loss], feed_dict={xs: x_data, ys: y_data})
        if not i % 5:
            run = sess.run(merge_all, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(run, i + 1000)
            print('训练损失0.05：{:>10.9f}'.format(l))
            try:
                ax.lines.remove(plot[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            plot = ax.plot(x_data, prediction_value, 'r-', lw=1)
            plt.pause(0.005)
            # if not i % 30000:
            #     print(prediction_value)
    train = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)
    for i in range(999):
        _, l = sess.run([train, loss], feed_dict={xs: x_data, ys: y_data})
        if not i % 5:
            run = sess.run(merge_all, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(run, i + 2000)
            print('训练损失0.0005：{:>10.9f}'.format(l))
            try:
                ax.lines.remove(plot[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            plot = ax.plot(x_data, prediction_value, 'r-', lw=1)
            plt.pause(0.005)
    train = tf.train.GradientDescentOptimizer(0.00005).minimize(loss)
    for i in range(999):
        _, l = sess.run([train, loss], feed_dict={xs: x_data, ys: y_data})
        if not i % 5:
            run = sess.run(merge_all, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(run, i + 3000)
            print('训练损失0.000005：{:>10.9f}'.format(l))
            try:
                ax.lines.remove(plot[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            plot = ax.plot(x_data, prediction_value, 'r-', lw=1)
            plt.pause(0.005)
    train = tf.train.GradientDescentOptimizer(0.000005).minimize(loss)
    for i in range(999):
        _, l = sess.run([train, loss], feed_dict={xs: x_data, ys: y_data})
        if not i % 5:
            run = sess.run(merge_all, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(run, i + 4000)
            print('训练损失0.00000005：{:>10.9f}'.format(l))
            try:
                ax.lines.remove(plot[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            plot = ax.plot(x_data, prediction_value, 'r-', lw=1)
            plt.pause(0.005)
