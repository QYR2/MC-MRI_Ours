import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np



class image2patch(nn.Module):
    def __init__(self, image_size=4, psize=2, stride=1):
        super(image2patch, self).__init__()
        window_size = image_size + 1 - psize
        mask = torch.arange(window_size*window_size)
        mask = mask.view(window_size, window_size)
        cur = torch.arange(0, window_size, stride)
        if not cur[-1] == window_size - 1:
            cur = torch.cat((cur, torch.LongTensor([window_size - 1])))
        mask = mask[cur,:]
        mask = mask[:,cur]
        self.mask = mask.view(-1)
        self.sizes = torch.LongTensor([psize, window_size, image_size])

    def patch_size(self):
        return self.mask.size(0), self.sizes[0]**2

    def forward(self, input_data):
        return patch_tf.apply(input_data, self.mask, self.sizes)

class patch2image(nn.Module):
    def __init__(self, image_size=4, psize=2, stride=1):
        super(patch2image, self).__init__()
        window_size = image_size + 1 - psize
        mask = torch.arange(window_size*window_size)
        mask = mask.view(window_size, window_size)
        cur = torch.arange(0, window_size, stride)
        if not cur[-1] == window_size - 1:
            cur = torch.cat((cur, torch.LongTensor([window_size - 1])))
        mask = mask[cur,:]
        mask = mask[:,cur]
        self.mask = mask.view(-1)
        self.sizes = torch.LongTensor([psize, window_size, image_size])
        self.ave_mask = ave_mask_com(self.mask, self.sizes)

    def forward(self, input_data):
        return image_tf.apply(input_data, self.mask, self.sizes, self.ave_mask)

def to_patch(x, mask, sizes, ave_mask=None, mode='sum'):
    batch_size = x.size(0)
    channels = x.size(1)
    global channel_patch          #改该过过过过过过过过过过过过过过过过过过
    channel_patch =channels #全局变量，跳出函数限制，甚至跳出你的类，但是不跳出你的文件限制          #改该过过过过过过过过过过过过过过过过过过
    patch_set = torch.zeros(batch_size*channels, sizes[1]**2, sizes[0]**2, device=x.device)
    if mode == 'ave':
        x = x / ave_mask
    for i in range(sizes[0]):
        for j in range(sizes[0]):
            index = i * sizes[0] + j
            temp = x[:,:,i:i+sizes[1],j:j+sizes[1]]
            temp = temp.contiguous().view(batch_size*channels, -1)
            patch_set[:,:,index] = temp
    output = patch_set[:,mask]
    return output



def to_image(x, mask, sizes, ave_mask=None, mode='sum'):
    batch_size = x.size(0)
    patch_set = torch.zeros(batch_size, sizes[1]**2, sizes[0]**2, device=x.device)
    output= torch.zeros(int(batch_size/channel_patch), channel_patch, sizes[2], sizes[2], device=x.device)     #改该过过过过过过过过过过过过过过过过过过
    patch_set[:,mask] = x
    for i in range(sizes[0]):
        for j in range(sizes[0]):
            index = i * sizes[0]  + j
            temp = patch_set[:,:,index]
            temp = temp.resize(int(batch_size/(channel_patch)), channel_patch, (sizes[1].numpy()), (sizes[1].numpy()))     #改该过过过过过过过过过过过过过过过过过过
            output[:,:,i:i+sizes[1],j:j+sizes[1]] = \
                output[:,:,i:i+sizes[1],j:j+sizes[1]] + temp
    if mode == 'ave':
        output = output / ave_mask
    return output


def ave_mask_com(mask, sizes):
    ave_mask = torch.zeros(sizes[2], sizes[2])
    patch_set = torch.zeros(sizes[1]**2, sizes[0]**2)
    patch_set[mask] = 1.0
    for i in range(sizes[0]):
        for j in range(sizes[0]):
            index = i * sizes[0]  + j
            temp = patch_set[:,index]
            temp = temp.view(sizes[1], sizes[1])
            ave_mask[i:i+sizes[1],j:j+sizes[1]] = \
                ave_mask[i:i+sizes[1],j:j+sizes[1]] + temp
    ave_mask = ave_mask.view(1, 1, sizes[2], sizes[2])
    return ave_mask

class patch_tf(Function):
    @staticmethod
    def forward(self, input_data, mask, sizes):
        self.save_for_backward(mask, sizes)
        output = to_patch(input_data, mask, sizes)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = None
        mask, sizes = self.saved_tensors
        grad_input = to_image(grad_output, mask, sizes)
        return grad_input, None, None


class image_tf(Function):
    @staticmethod
    def forward(self, input_data, mask, sizes, ave_mask):
        ave_mask = ave_mask.to(input_data.device)
        self.save_for_backward(mask, sizes, ave_mask)
        output = to_image(input_data, mask, sizes, ave_mask=ave_mask, mode='ave')
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = None
        mask, sizes, ave_mask = self.saved_tensors
        grad_input = to_patch(grad_output, mask, sizes, ave_mask=ave_mask, mode='ave')
        return grad_input, None, None, None


# cut_image=image2patch()
# cut_image_recover=patch2image()

# image = torch.randn(1,2,4,4)
# image_list = cut_image(image)
# answer=cut_image_recover(image_list)
# print("OK")

#通过debug可以得知无法处理(1,2,256,256),但是经过修改可以完成！
