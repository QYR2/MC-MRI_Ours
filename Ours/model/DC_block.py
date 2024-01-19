#DC模块是一个函数，不是1个网络
#DC模块和total_TV模块

import torch
import torch.nn as nn
import utilize.Tools as Tools

#这里的input_data和mask必须是torch.tensor(dtype=torch.float32)
#input_data和output_data 必须是(BS,2,256,256)实数数据

#注意！这里的dc_block 可适用于 引导重建 和 联合重建  和   T1全catT2欠 重建

def dc_block(origin_data,mask,reconstruction_data):
    mymask = torch.ones_like(mask) - mask      #现在的mymask的维度是(BS,256,256) ，读取到的mask维度是(BS,256,256)，现在是float64的tensor
    mymask = torch.tensor(mymask, dtype=torch.float32)

    origin_data = Tools.double2complex(image=origin_data)       # (BS,256,256)复数数据
    FXu = Tools.tc_fft2c(origin_data)  # 这是欠采样的原始T2的K空间数据

    reconstruction_data = Tools.double2complex(image=reconstruction_data)  # (BS,256,256)复数数据
    FXu2 = Tools.tc_fft2c(reconstruction_data)  # 这是欠采样的重建T2的K空间数据

    answer = torch.multiply(mymask,FXu2)+FXu
    answer = Tools.tc_ifft2c(answer)   #要转到图像域
    answer = Tools.complex2double(answer)

    return  answer

#对应的公式是