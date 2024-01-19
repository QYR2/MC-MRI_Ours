import time
import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse
import numpy as np
from utilize import Tools as Tools
import model.mymodel as model
import utilize.evaluation  as eval
from utilize import show_picture as show_picture



parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")  #如果有26份图片，batchsize=8，那么会有4次来显示，因为下面显示是第0张图片显示
parser.add_argument("--seed", type=int, default=42)
opt = parser.parse_args()
cuda = True if torch.cuda.is_available() else False


class net():
    def __init__(self):
        self.model = model.model()
        image_u, label_data, mask ,t1= Tools.get_testdata('datasets/data.mat')   #这最好要改动一下，用test数据,不能用训练验证的数据,现在这是留出法

        self.image_u = torch.tensor(image_u, device='cuda')
        self.label_data = torch.tensor(label_data, device='cuda')
        self.mask = torch.tensor(mask, device='cuda')
        self.t1=torch.tensor(t1,device='cuda')
        self.accurate_time = 0


        self.undersampling_rate = Tools.undersampling_rate(mask)
        print('self.undersampling_rate: ', self.undersampling_rate)
        test_data = TensorDataset(self.image_u, self.label_data, self.mask , self.t1)
        print('test_data numbers: ', image_u.shape[0])
        Tools.get_random_seed(opt.seed) #添加随机种子42
        self.test_data = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)

        if cuda:
            self.model = self.model.cuda()

    def test(self):
        checkpt_file = 'saved_models/model_epoch_.pth'  #测试的时候，是指用这个pth，不会用其他的
        self.model.load_state_dict(torch.load(checkpt_file))
        print('-->start testing')
        all_PSNR1 = []
        all_RLNE1 = []
        all_RLNE = []
        all_PSNR = []
        all_SSIM = []
        all_SSIM1 = []
        accurate_time=0


        for index, data in enumerate(self.test_data):     #这个idnex表示批次的第几个，这里批次大小为8，那么idnex为0，1，2，3
            test_data, lable_data, mask , t1 = data
            regu = torch.max(abs(test_data))
            test_data, lable_data , t1= test_data / regu, (
                (lable_data).view(test_data.shape[0],  test_data.shape[-2], test_data.shape[-1])) / regu , t1/regu


            test_data = Tools.complex2double(test_data)
            t1=Tools.complex2double(t1)
            with torch.no_grad():                                       #with语句的作用是：当XXX时 YYY，如果XXX不满足，就不要干

                tic2 = time.time()
                output = self.model(test_data,mask,t1)
                toc2 = time.time()
                accurate_time=toc2-tic2+accurate_time

                output = Tools.double2complex(image=output)
                test_data = Tools.double2complex(image=test_data)
            torch.cuda.empty_cache()

            showit = abs(output[0, :, :]).cpu()
            showit = showit.detach().numpy()
            showit2 = abs(lable_data[0, :, :]).cpu()
            showit2 = showit2.detach().numpy()  #showit2是真实图像
            showit3 = abs(test_data[0, :, :]).cpu()
            showit3 = showit3.detach().numpy() #showit3是欠采样的图像

            show_picture.show_picture(showit, showit2, showit3,index)

            RLNE = eval.RLNE(lable_data, output)
            RLNE1 = eval.RLNE(lable_data,  test_data )
            PSNR = eval.PSNR(lable_data, output)
            PSNR1 = eval.PSNR(lable_data, test_data)
            SSIM = eval.get_ssim(abs(lable_data), abs(output))
            SSIM1 = eval.get_ssim(abs(lable_data), abs(test_data))

             # 不用添加cpu语句了

            all_RLNE.append(RLNE)
            all_RLNE1.append(RLNE1)
            all_PSNR.append(PSNR)
            all_PSNR1.append(PSNR1)
            all_SSIM.append(SSIM)
            all_SSIM1.append(SSIM1)


        all_RLNE = np.array(all_RLNE)
        all_RLNE1 = np.array(all_RLNE1)
        all_PSNR = np.array(all_PSNR)
        all_PSNR1 = np.array(all_PSNR1)
        all_SSIM = np.array(all_SSIM)
        all_SSIM1 = np.array(all_SSIM1)



        print('test---------mean', " [RLNE: 之后%f/之前%f] : [PSNR: 之后%f/之前%f] : [SSIM: 之后%f/之前%f]" % ( all_RLNE.sum() / (all_RLNE.size),
            all_RLNE1.sum() / (all_RLNE1.size), all_PSNR.sum() / (all_PSNR.size), all_PSNR1.sum() / (all_PSNR1.size),
            all_SSIM.sum()/all_SSIM.size, all_SSIM1.sum()/all_SSIM1.size))

        print('test---------std', " [RLNE: 之后%f/之前%f] : [PSNR: 之后%f/之前%f] : [SSIM: 之后%f/之前%f]" % (
            np.std(all_RLNE), np.std(all_RLNE1), np.std(all_PSNR), np.std(all_PSNR1), np.std(all_SSIM), np.std(all_SSIM1) ) )

        accurate_time=accurate_time/opt.batch_size
        print('测试准确所用的时间：', accurate_time, 's')
        torch.cuda.empty_cache()


if __name__ == "__main__":
    tic = time.time()
    network = net()  # 实例化
    network.test()
    toc = time.time()
    print('测试所用的时间：', toc-tic, 's')

