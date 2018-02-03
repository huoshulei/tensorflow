import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, ):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_drop)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name + '/output', outputs)
    return outputs


keep_drop = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])

l1 = add_layer(xs, 64, 50, 'l1', tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', tf.nn.softmax)

loss = tf.reduce_mean(-tf.reduce_sum(ys + tf.log(prediction), reduction_indices=[1]))

tf.summary.scalar('loss', loss)
train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
merge_all = tf.summary.merge_all()
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('logs/test', sess.graph)
    sess.run(tf.initialize_all_variables())
    for i in range(9999):
        _, l = sess.run([train, loss], feed_dict={xs: X_train, ys: y_train, keep_drop: 0.5})
        if not i % 50:
            print('损失：{:>10.5f}'.format(l))
            train_result = sess.run(merge_all, feed_dict={xs: X_train, ys: y_train, keep_drop: 1})
            test_result = sess.run(merge_all, feed_dict={xs: X_test, ys: y_test, keep_drop: 1})
            train_writer.add_summary(train_result, i)
            test_writer.add_summary(test_result, i)
