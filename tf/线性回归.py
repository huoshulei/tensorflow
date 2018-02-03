print('线性回归')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 产生数据
def linear(x):
    return 1.25 * x + 3.75


line_space = np.linspace(-5., 5., 5555)
plt.plot(line_space, linear(line_space))
plt.grid()
plt.show()


# 添加噪声
def linear_with_n(x):
    return [linear(t) + np.random.normal(0, 0.5) for t in x]


plt.plot(line_space, linear_with_n(line_space))
plt.grid()
plt.show()

# 采样
sampled_x = np.random.choice(line_space, size=2048)
sampled_y = linear_with_n(sampled_x)
plt.scatter(sampled_x, sampled_y)
plt.grid()
plt.show()

# 对数据进行预处理
whole = np.transpose(np.array([sampled_x, sampled_y]))
training_set = whole[:-64]
test_set = whole[-256:]


def gen_batch(data):
    for i in range(len(data) // 64):
        pos = 64 * i
        yield data[pos:pos + 64]


# 计算图
graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='x')
    y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='y')
    a = tf.Variable(0.)
    b = tf.Variable(0.)
    linear_model = a * x + b
    loss = tf.reduce_mean(tf.square(linear_model - y))
    opt = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

# 运行计算图
with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    for epoch in range(666):
        for minibatch in gen_batch(training_set):
            _, l = sess.run([opt, loss],
                            feed_dict={x: np.reshape(minibatch[:, 0], (-1, 1)),
                                       y: np.reshape(minibatch[:, 1], (-1, 1))})
            step += 1
            if step % 100 == 0:
                print('训练损失：{:>10.4f}'.format(l))
    print("训练结束")
    res, l = sess.run([(a, b), loss],
                      feed_dict={x: np.reshape(test_set[:, 0], (-1, 1)),
                                 y: np.reshape(test_set[:, 1], (-1, 1))})
    print('测试损失：{:>10.4f}'.format(l))
    print(res)
    print(step)


def pre(x, res):
    return x * res[0] + res[1]


plt.plot(line_space, pre(line_space, res), 'salmon')
plt.scatter(sampled_x[:256], sampled_y[:256])
plt.grid()
plt.show()
