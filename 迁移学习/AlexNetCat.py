import tensorflow as tf
import numpy as np
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import os


def load_img(path):
    img = skimage.io.imread(path)
    img = img / 255.
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    return skimage.transform.resize(crop_img, (224, 224))[None, :, :, :]


def load_data():
    imgs = {'tiger': [], 'kittycat': []}
    y = {'tiger': [], 'kittycat': []}
    for k in imgs.keys():
        dir = './for_transfer_learning/data/' + k
        for file in os.listdir(dir):
            if not file.lower().endswith('.jpg'):
                continue
            try:
                resized_img = load_img(os.path.join(dir, file))
                imgs[k].append(resized_img)
                y[k].append([0, 1] if k == 'tiger' else [1, 0])
            except OSError:
                continue
    return imgs['tiger'], y['tiger'], imgs['kittycat'], y['kittycat']


class AlexNet(object):
    def __init__(self, keep_prod=1., restore_path=None):
        self.x = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.y = tf.placeholder(tf.float32, [None, 2])
        r1, g, b = tf.split(axis=3, num_or_size_splits=3, value=self.x * 255.)
        bgr = tf.concat(axis=3, values=[
            b - 103.939,
            g - 116.779,
            r1 - 123.68
        ])
        weights = {
            'wc1': tf.Variable(tf.random_normal([5, 5, 3, 64]), dtype=tf.float32, name='wc1'),  # 224*224
            'wc2': tf.Variable(tf.random_normal([5, 5, 64, 128]), dtype=tf.float32, name='wc2'),  # 122*122
            'wc3': tf.Variable(tf.random_normal([5, 5, 128, 256]), dtype=tf.float32, name='wc3'),  # 61*61
            'wc4': tf.Variable(tf.random_normal([5, 5, 256, 384]), dtype=tf.float32, name='wc4'),  # 31*31
            'wc5': tf.Variable(tf.random_normal([5, 5, 384, 512]), dtype=tf.float32, name='wc5'),  # 16*16
            'wc6': tf.Variable(tf.random_normal([5, 5, 512, 768]), dtype=tf.float32, name='wc6'),  # 16*16
            'wc7': tf.Variable(tf.random_normal([5, 5, 768, 512]), dtype=tf.float32, name='wc7'),  # 8*8
            'wc8': tf.Variable(tf.random_normal([5, 5, 512, 384]), dtype=tf.float32, name='wc8'),  # 8*8
            'wc9': tf.Variable(tf.random_normal([5, 5, 384, 256]), dtype=tf.float32, name='wc9'),  # 4*4
            'wd1': tf.Variable(tf.random_normal([2 * 2 * 256, 4096]), dtype=tf.float32, name='wd1'),  # 4*4*256
            'wd2': tf.Variable(tf.random_normal([4096, 2048]), dtype=tf.float32, name='wd2'),  # 4*4
            'out': tf.Variable(tf.random_normal([2048, 2]), dtype=tf.float32, name='w_out')  # 4*4
        }
        biases = {
            'bc1': tf.Variable(tf.random_normal([64]), dtype=tf.float32, name='bc1'),
            'bc2': tf.Variable(tf.random_normal([128]), dtype=tf.float32, name='bc2'),
            'bc3': tf.Variable(tf.random_normal([256]), dtype=tf.float32, name='bc3'),
            'bc4': tf.Variable(tf.random_normal([384]), dtype=tf.float32, name='bc4'),
            'bc5': tf.Variable(tf.random_normal([512]), dtype=tf.float32, name='bc5'),
            'bc6': tf.Variable(tf.random_normal([768]), dtype=tf.float32, name='bc6'),
            'bc7': tf.Variable(tf.random_normal([512]), dtype=tf.float32, name='bc7'),
            'bc8': tf.Variable(tf.random_normal([384]), dtype=tf.float32, name='bc8'),
            'bc9': tf.Variable(tf.random_normal([256]), dtype=tf.float32, name='bc9'),
            'bd1': tf.Variable(tf.random_normal([4096]), dtype=tf.float32, name='bd1'),
            'bd2': tf.Variable(tf.random_normal([2048]), dtype=tf.float32, name='bd2'),
            'out': tf.Variable(tf.random_normal([2]), dtype=tf.float32, name='b_out')

        }
        conv1 = self.conv2d('conv1', bgr, w=weights['wc1'], b=biases['bc1'])
        pool1 = self.max_pool('pool1', conv1)  # 122*122
        norm1 = self.norm('norm1', pool1)

        conv2 = self.conv2d('conv2', norm1, w=weights['wc2'], b=biases['bc2'])
        pool2 = self.max_pool('pool2', conv2)  # 61*61
        norm2 = self.norm('norm2', pool2)

        conv3 = self.conv2d('conv3', norm2, w=weights['wc3'], b=biases['bc3'])
        pool3 = self.max_pool('pool3', conv3)  # 31*31
        norm3 = self.norm('norm3', pool3)

        conv4 = self.conv2d('conv4', norm3, w=weights['wc4'], b=biases['bc4'])
        pool4 = self.max_pool('pool4', conv4)  # 16*16
        norm4 = self.norm('norm4', pool4)

        conv5 = self.conv2d('conv5', norm4, w=weights['wc5'], b=biases['bc5'])
        # pool5 = self.max_pool('pool5', conv5)
        # norm5 = self.norm('norm5', pool5)

        conv6 = self.conv2d('conv6', conv5, w=weights['wc6'], b=biases['bc6'])
        pool6 = self.max_pool('pool6', conv6)  # 8*8
        norm6 = self.norm('norm6', pool6)

        conv7 = self.conv2d('conv7', norm6, w=weights['wc7'], b=biases['bc7'])

        conv8 = self.conv2d('conv8', conv7, w=weights['wc8'], b=biases['bc8'])
        pool8 = self.max_pool('pool8', conv8)  # 4*4
        norm8 = self.norm('norm8', pool8)

        conv9 = self.conv2d('conv9', norm8, w=weights['wc9'], b=biases['bc9'])
        pool9 = self.max_pool('pool9', conv9)  # 2*2
        norm9 = self.norm('norm9', pool9)
        # 全连接层
        fc1 = self.add_layer('fc1', biases['bd1'], keep_prod, norm9, weights['wd1'])
        fc2 = self.add_layer('fc2', biases['bd2'], keep_prod, fc1, weights['wd2'])
        # 输出层
        self.out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
        self.sess = tf.Session()
        if restore_path:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_path)
        else:
            # self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.out)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.out))
            # self.train_op = tf.train.RMSPropOptimizer(.0001).minimize(self.loss)
            self.train_op = tf.train.AdamOptimizer(.0005).minimize(self.loss)
            try:
                saver = tf.train.Saver()
                saver.restore(self.sess, './for_transfer_learning/model/transfer_kitty')
            except:
                self.sess.run(tf.global_variables_initializer())

    def add_layer(self, name, biases, keep_prod, input, weights):
        with tf.variable_scope(name):
            fc = tf.reshape(input, [-1, weights.shape[0]])
            fc = tf.add(tf.matmul(fc, weights), biases)
            fc = tf.nn.relu(fc)
            return tf.nn.dropout(fc, keep_prod)

    # 卷积
    def conv2d(self, name, x, w, b, strides=1):
        with tf.variable_scope(name):
            conv = tf.nn.conv2d(x, w, [1, strides, strides, 1], padding='SAME')
            x = tf.nn.bias_add(conv, b)
            return tf.nn.relu(x, name=name)

    # 池化
    def max_pool(self, name, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

    def norm(self, name, input, lsize=4):
        return tf.nn.lrn(input, lsize, bias=1., alpha=9.00 / 9., beta=.75, name=name)

    # 训练模型
    def train(self, x, y):
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
            self.x: x,
            self.y: y
        })
        return loss

    # 预测
    def predict(self, paths):
        fig, axs = plt.subplots(1, 2)
        for i, path in enumerate(paths):
            img = load_img(path)
            output = self.sess.run(self.out, feed_dict={self.x: img})
            a = tf.argmax(output, 1)
            axs[i].imshow(img[0])
            classes = self.sess.run(a)
            axs[i].set_title('tiger' if classes[0] else 'cat')
            axs[i].set_xticks(())
            axs[i].set_yticks(())
        plt.show()

    # 保存训练参数
    def save(self, path='./for_transfer_learning/model/transfer_kitty'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)


# 训练
def train():
    tiger_x, tiger_y, cat_x, cat_y = load_data()
    x = np.concatenate(tiger_x + cat_x, axis=0)
    y = np.concatenate((tiger_y, cat_y), axis=0)
    net = AlexNet(keep_prod=.5)
    for i in range(331):
        idx = np.random.randint(0, len(x), 128)
        loss = net.train(x[idx], y[idx])
        print(i, 'Loss:{:>.6f}'.format(loss))
        if not i % 33:
            net.save()


# 预测
def eval():
    net = AlexNet(restore_path='./for_transfer_learning/model/transfer_kitty')
    net.predict(['./for_transfer_learning/data/kittycat/59381086_fca6bcee81.jpg',
                 './for_transfer_learning/data/tiger/2236557600_04d0b7197f.jpg'])


if __name__ == '__main__':
    # train()
    eval()
