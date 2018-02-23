import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# from tensorflow.contrib.keras.api.keras.models import Sequential

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

learning_rate = .0005
training_iters = 200000
batch_size = 128
display_step = 10

n_input = 784
n_classes = 10
dropout = .75

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prod = tf.placeholder(tf.float32)


# 卷积层
def conv2d(name, x, w, b, strides=1):
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name)


# 池化层
def max_pool(name, x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


# 规范化操作
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1., alpha=.0001 / 9., beta=.75, name=name)


# 网络参数
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 96]), dtype=tf.float32),  # patch 11*11 输入 1 输出96
    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256]), dtype=tf.float32),  # patch 5*5 input 96 output 256
    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384]), dtype=tf.float32),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384]), dtype=tf.float32),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256]), dtype=tf.float32),
    'wd1': tf.Variable(tf.random_normal([2 * 2 * 256, 4096]), dtype=tf.float32),
    # 全连接层 输入维度 4*4*256 为上一层的输出数据由三维转换成一维 输出维度4096
    'wd2': tf.Variable(tf.random_normal([4096, 4096]), dtype=tf.float32),
    'out': tf.Variable(tf.random_normal([4096, n_classes]), dtype=tf.float32)
}

biases = {
    'bc1': tf.Variable(tf.random_normal([96]), dtype=tf.float32),
    'bc2': tf.Variable(tf.random_normal([256]), dtype=tf.float32),
    'bc3': tf.Variable(tf.random_normal([384]), dtype=tf.float32),
    'bc4': tf.Variable(tf.random_normal([384]), dtype=tf.float32),
    'bc5': tf.Variable(tf.random_normal([256]), dtype=tf.float32),
    'bd1': tf.Variable(tf.random_normal([4096]), dtype=tf.float32),
    'bd2': tf.Variable(tf.random_normal([4096]), dtype=tf.float32),
    'out': tf.Variable(tf.random_normal([n_classes]), dtype=tf.float32)
}


def alex_net(x, weights, biases, dropout):
    # reshape batch -1 宽度28 高度 28 单色通道
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # 卷积
    conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'])
    # 池化
    pool1 = max_pool('pool1', conv1, k=2)
    # 采样
    norm1 = norm('norm1', pool1, lsize=4)
    conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
    pool2 = max_pool('pool2', conv2, k=2)
    norm2 = norm('norm2', pool2, lsize=4)
    conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
    pool3 = max_pool('pool3', conv3, k=2)
    norm3 = norm('norm3', pool3, lsize=4)
    conv4 = conv2d('conv4', norm3, weights['wc4'], biases['bc4'])
    conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'])
    pool5 = max_pool('pool5', conv5, k=2)
    norm5 = norm('norm5', pool5, lsize=4)
    # 全连接层
    fc1 = tf.reshape(norm5, [-1, weights['wd1'].get_shape().as_list()[0]])
    # fc1 = tf.reshape(norm5, [-1, weights['wd1'].shape[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)

    # 输出层
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out


# 网络模型
prediction = alex_net(x, weights, biases, dropout)

# 损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
# 优化器
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# 评估函数
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 评估和训练模型
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    step = 0
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # pred = sess.run(prediction, feed_dict={x: batch_x, keep_prod: dropout})
        # print(len(pred))
        sess.run(train_op, feed_dict={
            x: batch_x,
            y: batch_y,
            keep_prod: dropout
        })
        # 计算训练集损失值和准确度
        if not step % display_step:
            c_, a_ = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prod: dropout})
            print('Loss:{:.8f} Accuracy:{:.6f}'.format(c_, a_))
        step += 1
    print('测试集数据准确度：{:.6f}'.format(sess.run(accuracy, feed_dict={
        x: mnist.test.images[:512],
        y: mnist.test.labels[:512],
        keep_prod: 1.
    }) * 100))
