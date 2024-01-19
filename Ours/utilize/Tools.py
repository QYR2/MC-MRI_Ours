import torch
import numpy as np
import h5py as h5
import random
import os


def get_data(filename):
    with h5.File(filename) as f:
        # 'trnImage', 'trnOrg', 'trnMask'
        trnImage, trnOrg, trnMask , trnT1= f['trnImage'][:], f['trnOrg'][:], f['trnMask'][:] ,f['trnT1'][:]
    trnImage = trnImage['real'] + trnImage['imag'] * 1j
    trnOrg = trnOrg['real'] + trnOrg['imag'] * 1j
    trnT1 = trnT1['real'] + trnT1['imag'] * 1j

    trnImage = torch.tensor(trnImage.transpose([2, 1, 0]), dtype=torch.complex64)
    trnOrg = torch.tensor(trnOrg.transpose([2, 1, 0]), dtype=torch.complex64)
    trnMask = torch.tensor(trnMask.transpose([2, 1, 0]))
    trnT1 = torch.tensor(trnT1.transpose([2,1,0]), dtype=torch.complex64)

    # trnImage = torch.tensor(trnImage.transpose([2, 1, 0]), dtype=torch.float32)
    # trnOrg = torch.tensor(trnOrg.transpose([2, 1, 0]), dtype=torch.float32)
    # trnMask = torch.tensor(trnMask.transpose([2, 1, 0]), dtype=torch.float32)
    # trnT1 = torch.tensor(trnT1.transpose([2,1,0]), dtype=torch.float32 )   #万一审稿人，要你用abs图像：
    #

    return trnImage, trnOrg, trnMask , trnT1


#函数说明，这里的get_data表示的是读取datasets文件下的.mat数据







def get_testdata(filename):
    with h5.File(filename) as f:
        # 'trnImage', 'trnOrg', 'trnMask'
        trnImage, trnOrg, trnMask , trnT1= f['trnImage'][:], f['trnOrg'][:], f['trnMask'][:] ,f['trnT1'][:]
    trnImage = trnImage['real'] + trnImage['imag'] * 1j
    trnOrg = trnOrg['real'] + trnOrg['imag'] * 1j
    trnT1 = trnT1['real'] + trnT1['imag'] * 1j

    trnImage = torch.tensor(trnImage.transpose([2, 1, 0]), dtype=torch.complex64)
    trnOrg = torch.tensor(trnOrg.transpose([2, 1, 0]), dtype=torch.complex64)
    trnMask = torch.tensor(trnMask.transpose([2, 1, 0]), dtype=torch.float32)
    trnT1 = torch.tensor(trnT1.transpose([2, 1, 0]), dtype=torch.complex64)

    # trnImage = torch.tensor(trnImage.transpose([2, 1, 0]), dtype=torch.float32)
    # trnOrg = torch.tensor(trnOrg.transpose([2, 1, 0]), dtype=torch.float32)
    # trnMask = torch.tensor(trnMask.transpose([2, 1, 0]), dtype=torch.float32)
    # trnT1 = torch.tensor(trnT1.transpose([2,1,0]) , dtype=torch.float32)   #万一审稿人，要你用abs图像：


    i = -(trnImage.size(0) // 10)
    return trnImage[i:], trnOrg[i:], trnMask[i:] , trnT1[i:]   #明白

#函数说明，这里的get_testdata也是读取的是 dataset下面dat数据集


def undersampling_rate(mask):
    rate_ = mask.sum() / (mask.shape[-1] * mask.shape[-2]*mask.shape[-3])
    return rate_

#该函数是计算采样率的函数


def complex2double(image, dim=1):
    output = torch.stack((image.real, image.imag), dim=dim)
    # output = torch.abs(image).resize(image.shape[0],1,image.shape[1],image.shape[2])  #万一审稿人，要你用abs图像：reality（BS，256,256）——》 reality（BS，1,256,256）
    return output    #目的是把复数张量的第二位表示为2个实数张量，实部+虚部,分别存储到2个通道上


#例如把读取的数据(BS，1,256,256)的复数变为(BS,2,256,256)


def double2complex(image_real=0, image_imag=1, image='none'):
    if image == 'none':
        output = torch.complex(image_real, image_imag)
    else:
        output = torch.complex(image[:, 0, ...], image[:, 1, ...])
        # output = output.resize(image.shape[0],image.shape[-2],image.shape[-1])  万一审稿人，要你用abs图像：reality（BS，256,256）《=== reality（BS，1,256,256）
    return output
    #使用手段是double2complex(image=a),把a进行转为没有通道的复数张量
    #目的是把1个张量,按照第二个通道位置合并表示为复数的形式,比如：实部是实数（24,1,256,256）,虚部是实数（24,1,256,256）
    #变为复数(24,256,256)，复数没有通道
    #这里都是有image=XXX



#下面2个函数只能是用在通道数为1的情况下,或是没有通道数的情况下，确保数据为复数

def tc_fft2c(x):
    kx = int(x.shape[-2])
    ky = int(x.shape[-1])
    axes = (-2, -1)
    k_space = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x), dim=axes)) / np.sqrt(kx * ky)   #这里虽然是np,但是输出类型是tensor
    return k_space


#这个函数是先对图像进行移频，再对图像傅立叶变换，再移频，除以系数256



def tc_ifft2c(x):
    kx = int(x.shape[-2])
    ky = int(x.shape[-1])
    axes = (-2, -1)
    image = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x), dim=axes)) * np.sqrt(kx * ky)
    return image


def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)                          # 保证后续使用random函数时，产生固定的随机数
   # torch.cuda.manual_seed(seed)                  # 为当前GPU设置
    torch.use_deterministic_algorithms(False)      #选取算法不固定
    #torch.backends.cudnn没有了，参考官方手册

def count_xigema(x,y,z):

    print("RLNE满足3西格玛的个数：%d"%(
        np.sum( (x>=np.mean(x)- np.std(x) ) & (x<=np.mean(x)+ np.std(x)))))

    print("PSNR满足3西格玛的个数：%d" % (
        np.sum( (y >= np.mean(y) -   np.std(y) ) &  (y <= np.mean(y) +  np.std(y)))))

    print("SSIM满足3西格玛的个数：%d" % (
        np.sum((z >= np.mean(z) -   np.std(z)) &  (z<= np.mean(z) +   np.std(z)))))

#这个函数是为了确保测试出来的数据满足3西格玛的精度