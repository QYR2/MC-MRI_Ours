import torch
path='../saved_models/model_epoch_00.pth'
pretrained_dict =torch.load(path)
tensor_list=list()
for k ,v in pretrained_dict.items():
    print(k)
    # 上面的代码是打出模型的参数名称，参数值，和参数大小













#demo:
# import torch
# path='../saved_models/model_epoch_0001.pth'
# pretrained_dict =torch.load(path)
# tensor_list=list()
# for k ,v in pretrained_dict.items():
#     #print(k) , print(v) , print(v.size())
#     # 上面的代码是打出模型的参数名称，参数值，和参数大小
#     print(k),tensor_list.append(v)
#
# print(tensor_list[0])  #这个表示的是查看第1个
# print(len(tensor_list))

#现在我们要获取每个参数变量的权重，这样就可以不用debug


