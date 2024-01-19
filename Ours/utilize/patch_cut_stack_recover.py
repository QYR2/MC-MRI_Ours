import torch
import torch.nn as nn
import numpy as np

def cut_image(image, patch_size):
    batch,channel,width, height = image.size()   #添加括号
    item_width = int(width / patch_size)
    item_height = int(height / patch_size)
    item_number = int(batch*channel*item_height*item_width)  #获取图像块的个数
    box_list = []
    number=0 #这里的number是索引的个数
    for k in range(0 ,batch)  :
        for z in range(0 ,channel):
            for i in range(0, item_width):
                for j in range(0, item_height):
                    number=number+1
                    box = image[ k ,z,  j *patch_size : (j + 1) * patch_size,  i *patch_size: (i + 1) * patch_size]
                    box_list.append(box) #通过切片索引建立patch
    box_list_stack=box_list[0].resize(1,patch_size,patch_size)
    for x in range(1,number):
        box_list_stack = torch.cat((box_list_stack, box_list[x].resize(1,patch_size,patch_size)),dim=0)

    return box_list_stack


def cut_image_recover(cut_image, patch_size, channel, batch):
    numbers = cut_image.size(0)
    height=  int(np.sqrt(patch_size * patch_size * numbers / batch / channel))
    width =  int(np.sqrt(patch_size * patch_size * numbers / batch / channel))
    item_width = int(width / patch_size)
    item_height = int(height / patch_size)

    image_recover=torch.zeros(batch,channel,height,width).cuda()
    count=-1
    for k in range(0, batch):
        for z in range(0, channel):
            for i in range(0, item_width):
                for j in range(0, item_height):
                    count=count+1
                    image_recover[k, z, j * patch_size: (j + 1) * patch_size, i * patch_size: (i + 1) * patch_size]=cut_image[count]

                       # 通过切片索引建立patch
    return image_recover

# image = torch.randn(1,2,4,4)
# image_list = cut_image(image, patch_size=2)
# answer=cut_image_recover(image_list,patch_size=2,channel=2,batch=1)
# print("OK")

# image = torch.randn(1,3,256,256)
# image_list = cut_image(image, patch_size=32)
#
# print("OK")



"""

  item_width  item_height 分别表示的是图像块的宽的个数、高的个数

 """