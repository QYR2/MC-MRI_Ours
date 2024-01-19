import math
import utilize.Tools as Tools  #打开项目，打开MRI文件夹，不要打开备胎文件夹
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

def to_single(x):
    x = Tools.complex2double(x)
    x_real = sos(x[:, 0, ...].squeeze())
    x_imag = sos(x[:, 1, ...].squeeze())
    output = torch.stack((x_real, x_imag), dim=1)
    return output

def sos(x, dim):
    pnorm = 2
    out = torch.sqrt(torch.sum(abs(x ** pnorm), axis=dim))
    return out.float()
#sos被to_single使用，to_single是用于多通道的多对比度图像重建


def RLNE(im_ori, im_rec):
    L2_error = im_ori - im_rec
    out = torch.norm(L2_error, p=2) / torch.norm(im_ori, p=2)  #是列向量的2范数，但是输入的张量必须是浮点或是复数，不能是整数
      # 输出为tensor,现在转为float ，不然 #RLNE= RLNE.detach().numpy()    #这是添加的，用于CPU
    out = float(out)
    return out


def PSNR(im_ori, im_rec):
    mse = torch.mean(abs(im_ori - im_rec) ** 2)
    peakval = torch.max(abs(im_ori))
    out = 10 * math.log10(abs(peakval) ** 2 / mse)
    return out                                #输出为float


def get_ssim(a,b):
    answer=0
    a=a.cpu()
    b=b.cpu()
    a=a.detach().numpy()
    b=b.detach().numpy()
    a=np.array(a)
    b=np.array(b)
    number=a.shape[0]
    for n in range(number):
        x = a[n, :, :]
        y = b[n, :, :]
        c = ssim(x, y, data_range=np.max([np.max(a), np.max(b)]))
        answer=answer+c
    answer=answer/number
    return answer
#
#这个SSIM用的是加窗7*7,所以图像的边长不能小于7
#该版本支持处理如（1,256,256）的数据，也支持（256,256）的数据


def get_ssim2(a,b):
    a=np.array(a)
    b=np.array(b)
    c = ssim(a, b, data_range=np.max([np.max(a), np.max(b)]))
    return c

#这个函数在loss_show.py