import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

learning_rate = .01
real_params = [1.2, 2.5]
init_params = [[5, 4],
               [5, 1],
               [2, 4.5],
               [-0.3,4.6]][3]

x = np.linspace(-1, 1, 555, dtype=np.float32)

y_fun = lambda a, b: np.sin(b * np.cos(a * x))
tf_y_fun = lambda a, b: tf.sin(b * tf.cos(a * x))

noise = np.random.randn(555) / 10

y = y_fun(*real_params) + noise

a, b = [tf.Variable(initial_value=p, dtype=tf.float32) for p in init_params]

prediction = tf_y_fun(a, b)

mse = tf.reduce_mean(tf.square(y - prediction))

train = tf.train.AdamOptimizer(learning_rate).minimize(mse)

a_list, b_list, cost_list = [], [], []

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(666):
        a_, b_, mse_ = sess.run([a, b, mse])
        a_list.append(a_)
        b_list.append(b_)
        cost_list.append(mse_)
        result, _ = sess.run([prediction, train])
        if not i % 33:
            print('a={:<.6f},b={:<.6f}'.format(a_, b_))

    plt.figure(1)
    plt.scatter(x, y, s=1, c='b')
    plt.plot(x, result, 'r-', lw=2)

    fig = plt.figure(2)
    ax = Axes3D(fig)
    a3D, b3D = np.meshgrid(np.linspace(-2, 7, 30), np.linspace(-2, 7, 30))
    cost3D = np.array([np.mean(np.square(y_fun(a_, b_) - y)) for a_, b_ in zip(a3D.flatten(), b3D.flatten())]).reshape(
        a3D.shape)
    ax.plot_surface(a3D, b3D, cost3D, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), alpha=.5)
    ax.scatter(a_list[0], b_list[0], zs=cost_list[0], s=300, c='r')
    ax.set_xlabel('a')
    ax.set_ylabel('y')
    ax.plot(a_list, b_list, zs=cost_list, zdir='z', c='r', lw=2)
    plt.show()
