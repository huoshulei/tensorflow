import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# try:
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# except Exception:
#     pass


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) - biases
    if activation_function is None:
        output = Wx_plus_b
    else:
        output = activation_function(Wx_plus_b)
    return output


xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)


def compute_accuracy(images, labels):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: images})
    equal = tf.equal(tf.argmax(y_pre, 1), tf.argmax(labels, 1))
    mean = tf.reduce_mean(tf.cast(equal, tf.float32))
    result = sess.run(mean, feed_dict={xs: images, ys: labels})
    return result * 100


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(9999):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={xs: batch_xs, ys: batch_ys})
        if not i % 99:
            print('准确度：{:>10.2f}%'.format(compute_accuracy(mnist.test.images, mnist.test.labels)))
