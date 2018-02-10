import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

lr = 0.00001
training_iters = 9999999
batch_size = 128

n_inputs = 28  # 行输入数据
n_steps = 28  # 总行数
n_hidden_units = 128
n_classes = 10  # 输出分类数量

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    'in': tf.Variable(tf.constant(.1, shape=[n_hidden_units])),
    'out': tf.Variable(tf.constant(.1, shape=[n_classes]))
}


def RNN(x, w, b):
    # x[128,28,28]
    # ==>{128*28,28]
    x = tf.reshape(x, [-1, n_inputs])
    # ==>[128*28,128]
    x_in = tf.matmul(x, w['in']) + b['in']
    # ==>[128,28,128]
    x_in = tf.reshape(x_in, [-1, n_steps, n_hidden_units])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1., state_is_tuple=True)
    # lstm cell is divided into two pars(c_state,m_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=_init_state, time_major=False)

    results = tf.matmul(states[1], w['out']) + b['out']
    # or
    # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results


prediction = RNN(x, weights, biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
train = tf.train.AdamOptimizer(lr).minimize(loss)
correct_prod = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prod, tf.float32))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    step = 0
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)  # batch_x shape=128*[28,28] batch_y shape=128*[10]
        batch_x = batch_x.reshape([batch_size, n_steps, n_inputs])  # 1*[128, 28, 28]
        sess.run(train, feed_dict={x: batch_x, y: batch_y})
        if not step % 20:
            print(sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
        step += 1
