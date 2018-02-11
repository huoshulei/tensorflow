"""
transfer_learning
没有梯子
拿不到数据
娶不到参数
等待梯子
家里不需梯子就可以下载
"""
import os
from urllib.request import urlretrieve
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt


def download():
    categories = ['tiger', 'kittycat']
    for category in categories:
        os.makedirs('./for_transfer_learning/data/{}'.format(category), exist_ok=True)
        with open('./imagenet_{}'.format(category), 'r') as file:
            urls = file.readlines()
            n_urls = len(urls)
            for i, url in enumerate(urls):
                try:
                    listdir = os.listdir('./for_transfer_learning/data/{}'.format(category))
                    if url.strip().split('/')[-1] in listdir:
                        continue
                    urlretrieve(url.strip(),
                                './for_transfer_learning/data/{}/{}'.format(category, url.strip().split('/')[-1]))
                    print('{} {}/{}'.format(url.strip(), i, n_urls))
                except:
                    print('{} {}/{}'.format(url.strip(), i, n_urls), '图片加载错误！')


def load_img(path):
    img = skimage.io.imread(path)
    print('图片img.shape：', img.shape)
    img = img / 255.
    print('图片img.shape/225.：', img.shape)
    short_edge = min(img.shape[:2])  # 根据中心裁剪图片
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    print('图片crop_img.shape：', crop_img.shape)
    resized_img = skimage.transform.resize(crop_img, (224, 224))[None, :, :, :]
    print('图片resized_img.shape：', resized_img.shape)
    return resized_img


def load_data():
    imgs = {'tiger': [], 'kittycat': []}
    for k in imgs.keys():
        dir = '.for_transfer_learning/data/' + k
        for file in os.listdir(dir):
            if not file.lower().endswith('.jpg'):
                continue
            try:
                resized_img = load_img(os.path.join(dir, file))
                imgs[k].append(resized_img)
            except OSError:
                continue

    tigers_y = np.maximum(20, np.random.randn(len(imgs['tiger']), 1) * 30 + 100)
    cat_y = np.maximum(10, np.random.randn(len(imgs['kittycat']), 1) * 8 + 40)
    return imgs['tiger'], imgs['kittycat'], tigers_y, cat_y


class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None, restore_path=None):
        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        except FileNotFoundError:
            print('Please download VGG16 parameters at here {}'.format(
                "https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM"))
        self.tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.tfy = tf.placeholder(tf.float32, [None, 1])
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.)  # RGB=>BGR
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2]
        ])

        # 添加预先训练好的vgg图层
        conv1_1 = self.conv_layer(bgr, 'conv1_1')
        conv1_2 = self.conv_layer(conv1_1, 'conv1_2')
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, 'conv2_1')
        conv2_2 = self.conv_layer(conv2_1, 'conv2_2')
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, 'conv3_1')
        conv3_2 = self.conv_layer(conv3_1, 'conv3_2')
        conv3_3 = self.conv_layer(conv3_2, 'conv3_3')
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, 'conv4_1')
        conv4_2 = self.conv_layer(conv4_1, 'conv4_2')
        conv4_3 = self.conv_layer(conv4_2, 'conv4_3')
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, 'conv5_1')
        conv5_2 = self.conv_layer(conv5_1, 'conv5_2')
        conv5_3 = self.conv_layer(conv5_2, 'conv5_3')
        pool5 = self.max_pool(conv5_3, 'pool5')

        # detach original VGG fc layers and
        # reconstruct yours own fc layers serve for your own purpose

        self.flatten = tf.reshape(pool5, [-1, 7 * 7 * 512])
        self.fc6 = tf.layers.dense(self.flatten, 256, tf.nn.relu, name='fc6')
        self.out = tf.layers.dense(self.fc6, 1, name='out')
        self.sess = tf.Session()
        if restore_path:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_path)
        else:
            self.loss = tf.losses.mean_squared_error(labels=self.tfy, predictions=self.out)
            self.train_op = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

    def conv_layer(self, x, name):
        with tf.variable_scope(name):
            conv = tf.nn.conv2d(x, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def max_pool(self, x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def train(self, x, y):
        loss, _ = self.sess.run([self.loss, self.train_op], {self.tfx: x, self.tfy: y})
        return loss

    def predict(self, paths):
        fig, axs = plt.subplots(1, 2)
        for i, path in enumerate(paths):
            x = load_img(path)
            length = self.sess.run(self.out, {self.tfx: x})
            axs[i].imshow(x[0])
            axs[i].set_title('Len:[:<.2f]cm'.format(length))
            axs[i].set_xticks(())
            axs[i].set_xticks(())
        plt.show()

    def save(self, path='./for_transfer_learning/model/transfer_learn'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)


if __name__ == '__main__':
    download()
