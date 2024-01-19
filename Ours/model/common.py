import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1= nn.init.normal_(torch.rand(1),0.00001,0).cuda()
        # self.conv2 = nn.Conv2d(in_channels=4,out_channels=2,kernel_size=3,padding=1,stride=1,dilation=1,bias=False)
        # self.conv3 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1, stride=1, dilation=1, bias=False)
        # self.conv2.weight=Parameter(nn.init.normal_(torch.randn(1),0.00001,0))
        # self.conv3.weight=Parameter(nn.init.normal_(torch.randn(1),0.00001,0))
    def forward(self,x):
        x=torch.mul(x,self.conv1)
        return x


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=True, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):

        res = self.body(x).mul(self.res_scale)
        res += x


        return res
class cnn(nn.Module):
    def __init__(self):
        super(cnn,self).__init__()
        self.conv1= nn.init.normal_(torch.rand(1),0.00001,0).cuda()
        # self.conv2 = nn.Conv2d(in_channels=4,out_channels=2,kernel_size=3,padding=1,stride=1,dilation=1,bias=False)
        # self.conv3 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1, stride=1, dilation=1,bias=False )
        # self.conv2.weight = Parameter(nn.init.normal_(torch.randn(1), 0.00001, 0))
        # self.conv3.weight = Parameter(nn.init.normal_(torch.randn(1), 0.00001, 0))
    def forward(self,x):
        x=torch.mul(x,self.conv1)
        return x

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
