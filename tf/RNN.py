import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

lr = 0.001
training_iters = 99999
batch_size = 128

n_inputs = 28  # 行输入数据
n_steps = 28  # 总行数
n_hidden_unis = 128
n_classes = 10  # 输出分类数量

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_unis])),
    'out': tf.Variable(tf.random_normal([n_hidden_unis, n_classes]))
}

biases = {
    'in': tf.Variable(tf.constant(.1, shape=[n_hidden_unis, ])),
    'out': tf.Variable(tf.constant(.1, shape=[n_classes, ]))
}


def RNN(x, w, b):

    pass


prediction = RNN(x, weights, biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction))
train = tf.train.AdamOptimizer(lr).minimize(loss)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    step = 0
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshcpe([batch_size, n_steps, n_inputs])
        sess.run(train, feed_dict={x: batch_x, y: batch_y})
        if not step % 20:
            print(sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
        step += 1
