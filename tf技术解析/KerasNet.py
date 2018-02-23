from tensorflow.contrib.keras.api.keras.datasets import mnist
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Activation, Convolution2D
# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras._impl.keras.utils import np_utils

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 128
n_classes = 10
n_epoch = 12  # 训练轮数

img_rows, img_cols = 28, 28
n_filters = 32
poll_size = (2, 2)  # 池化大小
kernel_size = (3, 3)  # 卷积核大小
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x.reshape(train_x.shape[0], img_rows, img_cols, 1)
test_x = test_x.reshape(test_x.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
train_x /= 255
test_x /= 255
# 将类向量转换成二进制矩阵
train_y = np_utils.to_categorical(train_y, n_classes)
test_y = np_utils.to_categorical(test_y, n_classes)
# 构建训练模型
model = Sequential()
# 添加卷积层
model.add(Convolution2D(n_filters, kernel_size[0],
                        kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
