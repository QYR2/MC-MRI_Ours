import matplotlib.pyplot as plt
from utilize import Tools as Tools
import utilize.evaluation as eval
import torch
def show_picture(showit,showit2,showit3,index):

        # if index==3 or index==0:
        if index == 3:

                plt.figure(figsize=(10, 6), dpi=1000)
                plt.subplot(221)
                plt.imshow(showit, cmap="gray")
                plt.title("reconstruction image")
                plt.axis('off')


                plt.subplot(222)
                plt.imshow(showit2, cmap="gray")
                plt.title("ground truth image")
                plt.axis('off')

                plt.subplot(223)
                plt.imshow(showit3, cmap="gray")
                plt.title("undersampled image")
                plt.axis('off')

                plt.subplot(224)
                cm = plt.cm.get_cmap("jet")  # 使用jet算法，实现error map的色力图
                plt.imshow(abs(showit3 - showit2), cm)
                plt.title("error image")
                plt.colorbar()  # 显示色力图
                plt.clim(0, 0.2)  # 绘制色力图的范围
                plt.axis('off')  # 去掉坐标轴
                plt.savefig("pictures/pic-{}.svg".format(index + 1))  # 这个是保存文件格式
                # plt.show()

                print("重建SSIM", eval.get_ssim2(showit, showit2))
                print("原始SSIM", eval.get_ssim2(showit3, showit2))
                showit_tensor = torch.tensor(showit, dtype=torch.float32)
                showit_tensor2 = torch.tensor(showit2, dtype=torch.float32)
                showit_tensor3 = torch.tensor(showit3, dtype=torch.float32)
                print("重建RLNE", eval.RLNE(showit_tensor, showit_tensor2))
                print("原始RLNE", eval.RLNE(showit_tensor3, showit_tensor2))
                print("重建PSNR", eval.PSNR(showit_tensor, showit_tensor2))
                print("原始PSNR", eval.PSNR(showit_tensor3, showit_tensor2))
                print("-" * 20)

#用于保存实验的图像