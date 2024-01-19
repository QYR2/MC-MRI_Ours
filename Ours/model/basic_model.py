import torch
import torch.nn as nn
import torchvision.transforms as transforms
import Attention.cross_selfattention as CSA
import utilize.patch_cut_stack_recover as PCSR
import model.Resnet as Resnet
import model.common as common
import Attention.sobel as sobel
import Attention.Gaussian_Filter as GF
import Attention.RCSTB as RS
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=int, default=1)
parser.add_argument("--zeta", type=int, default=1)
parser.add_argument("--lamuda", type=int, default=1)
opt = parser.parse_args()


class basic_model(nn.Module):
    def __init__(self):
        super(basic_model, self).__init__()
        self.CSA = CSA.cross_selfattention(in_channels=512)  #这是输入的通道数
        self.conv = nn.Conv2d(in_channels=4,out_channels=2,kernel_size=3,padding=1,stride=1,dilation=1,)
        self.conv1 = common.CNN()
        self.fuse = nn.Conv2d(in_channels=4,out_channels=2,kernel_size=3,padding=1,stride=1,dilation=1,)
        self.conv2 = common.cnn()
        self.res_swin = RS.RES_SWIN()
        self.sobel = sobel.nn_conv2d_sobel
        self.GF =GF.get_gaussian_kernel()


    def forward(self, input_t2, input_t1):    #input_t1表示的是已经通过sobel出来的

        input_t2= self.res_swin(input_t2)
        input_t2_sobel =self.sobel(im=input_t2)
        input_t2_GF = self.GF(input_t2)
        lost_information =  input_t2 - input_t2_GF - input_t2_sobel

        output_t2 = PCSR.cut_image(input_t2_sobel,16).unsqueeze(dim=0)
        output_t1 = PCSR.cut_image(input_t1,16).unsqueeze(dim=0)
        output_t1 ,output_t2 = self.CSA(output_t1,output_t2)
        output_t2 = PCSR.cut_image_recover(output_t2.squeeze(dim=0)  , patch_size=16, channel=2, batch=1)
        output_t1 = PCSR.cut_image_recover(output_t1.squeeze(dim=0)  , patch_size=16, channel=2, batch=1)
        lost_information=self.conv2(lost_information)

        T2_reconstrucion_before_fuse = self.conv(torch.cat((output_t2,output_t1),dim=1)) + output_t2
        T2_reconstrucion = self.conv1(self.fuse(torch.cat((T2_reconstrucion_before_fuse,input_t2_GF),dim=1))) + lost_information + input_t2



        torch.cuda.empty_cache()
        return T2_reconstrucion       #最终的方式









