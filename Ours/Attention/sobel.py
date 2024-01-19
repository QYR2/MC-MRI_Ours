import torch
import numpy as np
from torch import nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F


def nn_conv2d_sobel(im):
    #这里的im
    batch_size , channels , height ,width = im.size()
    im_channel1 = im[:,0,:,:].view(batch_size,1,height,width)
    im_channel2 = im[:,1,:,:].view(batch_size,1,height,width)

    # 用nn.Conv2d定义卷积操作
    conv_op_Gx = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1,bias=False)
    conv_op_Gy = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1,bias=False)

    # 定义sobel算子参数
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1] ], dtype='float32' )
    sobel_kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],dtype='float32' )

    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel_x = sobel_kernel_x.reshape((1,1,3,3))
    sobel_kernel_y = sobel_kernel_y.reshape((1,1,3,3))


    # 给卷积操作的卷积核赋值
    conv_op_Gx.weight.data = torch.from_numpy(sobel_kernel_x)
    conv_op_Gx.weight.data=conv_op_Gx.weight.data.cuda()

    conv_op_Gy.weight.data = torch.from_numpy(sobel_kernel_y)
    conv_op_Gy.weight.data = conv_op_Gy.weight.data.cuda()


    # 对图像进行卷积操作
    edge_detect_Gx_channel1 = conv_op_Gx(im_channel1)
    edge_detect_Gy_channel1 = conv_op_Gy(im_channel1)
    edge_detect_channel1 = torch.abs(edge_detect_Gx_channel1)+torch.abs(edge_detect_Gy_channel1)

    edge_detect_Gx_channel2 = conv_op_Gx(im_channel2)
    edge_detect_Gy_channel2 = conv_op_Gy(im_channel2)
    edge_detect_channel2 = torch.abs(edge_detect_Gx_channel2)+torch.abs(edge_detect_Gy_channel2)

    edge_detect = torch.cat( (edge_detect_channel1,edge_detect_channel2) ,dim=1)


    return edge_detect




"""  这是sobel算子的代码，参考https://blog.csdn.net/qq_43027065/article/details/124104231
 
            """

"""
Sobel, I., Feldman, G., "A 3x3 Isotropic Gradient Operator for Image Processing", presented at the Stanford Artificial Intelligence Project (SAIL) in 1968.
"""