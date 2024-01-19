import torch
import torch.nn as nn
import utilize.patch_cut_stack_recover as PCSR

#下面这个cross_selfattention 的代码 是计算 x与y

class cross_selfattention(nn.Module):
    def __init__(self,in_channels):
        super(cross_selfattention,self).__init__()

        self.query_con = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)
        self.key_con = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)
        self.value_con = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)
        self.bn512 = nn.BatchNorm2d(512, affine=True)
        self.relu = nn.LeakyReLU(inplace=False)
        self.query_conv = nn.Sequential(self.query_con,self.bn512,self.relu)
        self.key_conv = nn.Sequential(self.key_con,self.bn512,self.relu)
        self.value_conv = nn.Sequential(self.value_con, self.bn512, self.relu)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x,y):


        batch_size_x , channels_x ,height_x ,width_x =x.size()
        batch_size_y, channels_y, height_y, width_y = y.size()

        query_x =self.query_conv(x).view(batch_size_x,-1,width_x*height_x)
        query_y = self.query_conv(y).view(batch_size_y, -1, width_y*height_y)  #把堆叠的图像块拉成行

        key_x = self.key_conv(x).view(batch_size_x,-1,width_x*height_x).permute(0,2,1)  #这个代码的含义是：将key的：比如 16×32×32 的维度变为 1×16×（32×32）
        key_y = self.key_conv(y).view(batch_size_y, -1, width_y*height_y).permute(0,2,1)
        #在这里key_x表示的是对key_x的转置

        energy_x = torch.bmm(query_x,key_y)  #这里的torch.bmm 表示的是矩阵乘法，英文含义是： batch matrix-matrix  公式如：b*n*m 与 b×m×p 进行对应的矩阵乘法 ，获得 b×n×p
        attention_x = self.softmax(energy_x/torch.sqrt(torch.tensor(key_y.size(2))))
        value_x = self.value_conv(y).view(batch_size_y,-1,width_y*height_y)
        out_x = torch.bmm(attention_x,value_x)
        out_x = out_x.view(batch_size_x,channels_x,height_x,width_x)
        out_x = self.sigmoid(out_x)



        energy_y = torch.bmm(query_y,key_x)  # 这里的torch.bmm 表示的是矩阵乘法，英文含义是： batch matrix-matrix  公式如：b*n*m 与 b×m×p 进行对应的矩阵乘法 ，获得 b×n×p
        attention_y = self.softmax(energy_y/torch.sqrt(torch.tensor(key_x.size(2))))
        value_y = self.value_conv(x).view(batch_size_x, -1, width_x * height_x)
        out_y = torch.bmm(attention_y,value_y)
        out_y = out_y.view(batch_size_y, channels_y, height_y, width_y)
        out_y = self.sigmoid(out_y)


        return out_x , out_y


