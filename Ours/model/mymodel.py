import torch.nn as nn
import torch
import model.basic_model as net  #文件夹,文件夹.文件夹.文件    第1个一定是文件夹 最后1个一定是文件
import utilize.Tools as Tools
import model.DC_block as DC
import Attention.sobel as sobel
#这里是你的生成器部分！

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.block1 = net.basic_model()



    def forward(self, input_data,mask,t1):
        x1 = input_data
        t1 = sobel.nn_conv2d_sobel(t1)
        x2 = self.block1(x1,t1)
        x3 = DC.dc_block(x1,mask,x2)    #原始，mask,重建

        x4 = self.block1(x3,t1)
        x5 = DC.dc_block(x1,mask,x4)    #原始，mask,重建


        torch.cuda.empty_cache()
        return x5





