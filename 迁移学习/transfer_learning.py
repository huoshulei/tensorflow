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
                    urlretrieve(url.strip(),
                                './for_transfer_learning/data/{}/{}'.format(category, url.strip().split('/')[-1]))
                    print('{} {}/{}'.format(url.strip(), i, n_urls))
                except:
                    print('{} {}/{}'.format(url.strip(), i, n_urls), '图片加载错误！')


if __name__ == '__main__':
    download()
