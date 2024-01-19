import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.pyplot import MultipleLocator

def plot_val_loss(n):
    y = np.load("loss_val.npy")
    x = range(1,len(y)+1,1)  #这里采用的是range变量作为x的表示,其中开始的是序列是1,结尾是len(y)+1。此外步长为1
    x = x[:n]   #这里表示的是x的序列的从0开始到第n个
    y = y[:n]
    plt.plot(x, y, '.-')
    plt_title = 'BATCH_SIZE = 1; LEARNING_RATE:0.001 ;'
    plt.title(plt_title)
    plt.xlabel('epoach {} times'.format(n))
    plt.ylabel('VAL-LOSS')
    # plt.savefig(file_name)
    plt.show()

#这个函数的目的是用于绘制验证集的数据的图片，其中横坐标表示的是epoach的次数，纵坐标表示的是验证集的损失函数





plot_val_loss(200)









