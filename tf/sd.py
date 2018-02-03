import tensorflow as tf

sess = tf.Session()
# n1 = tf.constant(3.0)
# n2 = tf.constant(4.0)
# print(sess.run([n1,n2]))
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# ad = tf.add(a, b)
# print(sess.run(ad, {a: 2, b: 3.2}))
# print(sess.run(ad, {a: [1, 2.3, 56], b: [5, 6.3, 88]}))
w = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = w * x + b
init = tf.global_variables_initializer()
sess.run(init)
# print(sess.run(linear_model, {x: [1, 2, 3, 4 ]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_mean(squared_deltas)
# print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# fw = tf.assign(w, [-1.])
# fb = tf.assign(b, [1.0])
# sess.run([fw, fb])
# print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# sess.run(init)
for i in range(6666):
    sess.run(train, feed_dict={x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
print(sess.run([w, b]))
