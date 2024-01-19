import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np

from utilize import Tools as Tools, evaluation as eval
from torch.utils.data import DataLoader, TensorDataset
from model import mymodel as model    #这里model就是 这个model_cnn_dc的文件
from myloss import myssim as myssim



parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")   #轮次epochs
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--model_save_path", type=str, default="./saved_models/1st")         #表示的是在上一级/saved_models文件夹/第1个文件
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")    #表示的是批次的大小
parser.add_argument("--seed", type=int, default=42)


opt = parser.parse_args()
cuda = True if torch.cuda.is_available() else False

class net():
    def __init__(self):

        self.loss1 = nn.MSELoss()
        self.loss2 = nn.BCEWithLogitsLoss()
        self.start = 0
        self.epoch = opt.epochs
        self.model = model.model( )
        self.path = opt.model_save_path
        self.mySSIM = myssim.SSIM().to("cuda")

        self.optimizer_generate = optim.AdamW(self.model.parameters(), lr=opt.lr)

        image_u, label_data, mask,t1 = Tools.get_data('datasets/data.mat')
        self.image_u = torch.tensor(image_u, device='cuda')
        self.label_data = torch.tensor(label_data, device='cuda')
        self.mask = torch.tensor(mask, device='cuda')
        self.t1=torch.tensor(t1,device='cuda')


        toc = time.time()
        print(toc - tic, 's')
        self.undersampling_rate = Tools.undersampling_rate(mask)
        print('self.undersampling_rate: ', self.undersampling_rate)

        i = -(self.label_data.size(0) // 10)
        # 划分数据集：训练、测试

        train_data = TensorDataset(self.image_u[:2 * i, ...],
                                   self.label_data[:2 * i, ...],
                                   self.mask[:2 * i, ...],
                                    self.t1[:2 * i,...] )
        val_data = TensorDataset(self.image_u[2 * i:i, ...],
                                 self.label_data[2 * i:i, ...],
                                 self.mask[2 * i:i, ...],
                                self.t1[2*i:i, ...] )

        print('train_data numbers: ', image_u.shape[0] + 2 * i)


        Tools.get_random_seed(opt.seed)
        self.train_data = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
        self.val_data = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True)

        self.scheduler_generate = optim.lr_scheduler.StepLR(self.optimizer_generate, step_size=10, gamma=0.9)

        if cuda:
            self.model = self.model.cuda()





    def train(self):
        loss_min = 1
        loss_val_show=[]
        loss_train_show=[]
        Dxrate_show = []
        

        for epoch in range(self.start, self.epoch):
            all_loss = []
            all_Dxrate = []
            all_RLNE = []
            all_RLNE1 = []
            all_PSNR = []
            all_PSNR1 = []
            all_SSIM =[]
            all_SSIM1=[]


            for index, data in enumerate(self.train_data):
                train_data, lable_data, mask , t1 = data

                regu = torch.max(abs(train_data))
                train_data, lable_data , t1 = train_data / regu, (
                    (lable_data).view(train_data.shape[0],  train_data.shape[-2], train_data.shape[-1])) / regu ,t1/regu

                train_data = Tools.complex2double(train_data)
                t1= Tools.complex2double(t1)

                output = self.model(train_data,mask,t1)
                output = Tools.double2complex(image=output)
                train_data = Tools.double2complex(image=train_data)


                self.optimizer_generate.zero_grad()
                gloss =  self.loss1(abs(output),abs(lable_data))
                loss=gloss
                loss.backward()

                self.optimizer_generate.step()
                torch.cuda.empty_cache()


                RLNE = eval.RLNE(lable_data, output)
                RLNE1 = eval.RLNE(lable_data, train_data)
                PSNR = eval.PSNR(lable_data, output)
                PSNR1 = eval.PSNR(lable_data, train_data)
                SSIM = eval.get_ssim(abs(lable_data), abs(output))
                SSIM1 = eval.get_ssim(abs(lable_data), abs(train_data))



                all_loss.append(loss.item())
                all_RLNE.append(RLNE)
                all_RLNE1.append(RLNE1)
                all_PSNR.append(PSNR)
                all_PSNR1.append(PSNR1)
                all_SSIM.append(SSIM)
                all_SSIM1.append(SSIM1)


            all_loss = np.array(all_loss)
            all_RLNE = np.array(all_RLNE)
            all_RLNE1 = np.array(all_RLNE1)
            all_PSNR = np.array(all_PSNR)
            all_PSNR1 = np.array(all_PSNR1)
            all_SSIM =  np.array(all_SSIM)
            all_SSIM1 = np.array(all_SSIM1)
            all_Dxrate = np.array(all_Dxrate)


            print('-->start training')
            print('---------', "[Epoch %d/%d] : [实际loss: %f] : [RLNE: 之后%f/之前%f] : [PSNR: 之后%f/之前%f] : [SSIM: 之后%f/之前%f]" % (
                epoch + 1, self.epoch,all_loss.sum() / (all_loss.size), all_RLNE.sum() / (all_RLNE.size),all_RLNE1.sum() / (all_RLNE1.size),
                all_PSNR.sum() / (all_PSNR.size), all_PSNR1.sum() / (all_PSNR1.size), all_SSIM.sum()/all_SSIM.size, all_SSIM1.sum()/all_SSIM1.size ))

            loss_train_it=all_loss.sum() / (all_loss.size)
            loss_train_show.append(loss_train_it)
            loss_Dxrate_it=all_Dxrate.sum()/(all_Dxrate.size)
            Dxrate_show.append(loss_Dxrate_it)

            self.scheduler_generate.step()
            torch.cuda.empty_cache()


            if epoch % 1 == 0:
                loss_val = self.val(epoch)
                loss_val_show.append(loss_val)

                if loss_val <loss_min:
                    loss_min = loss_val
                    torch.save(self.model.state_dict(),
                               '%s/model_epoch_%04d.pth' % ('saved_models', epoch + 1))

            np.save("myloss/loss_val.npy", loss_val_show)
            np.save("myloss/loss_train.npy", loss_train_show)



    def val(self, epoch):
        print('-->start validation')
        all_RLNE1 = []
        all_PSNR1 = []
        all_loss = []
        all_RLNE = []
        all_PSNR = []
        all_SSIM = []
        all_SSIM1 = []

        for index, data in enumerate(self.val_data):
            val_data, lable_data, mask , t1= data

            regu = torch.max(abs(val_data))
            val_data, lable_data , t1= val_data / regu, (
                (lable_data).view(val_data.shape[0],  val_data.shape[-2], val_data.shape[-1])) / regu  , t1/regu


            val_data = Tools.complex2double(val_data)
            t1=Tools.complex2double(t1)
            with torch.no_grad():
                output = self.model(val_data,mask,t1)
                output = Tools.double2complex(image=output)
                val_data = Tools.double2complex(image=val_data)

            loss=self.loss1(abs(output),abs(lable_data))
            torch.cuda.empty_cache()

            RLNE = eval.RLNE(lable_data, output)
            RLNE1 = eval.RLNE(lable_data, val_data)
            PSNR = eval.PSNR(lable_data, output)
            PSNR1 = eval.PSNR(lable_data, val_data)
            SSIM = eval.get_ssim(abs(lable_data), abs(output))
            SSIM1 = eval.get_ssim(abs(lable_data), abs(val_data))


            #不用添加改CPU语句了
            all_loss.append(loss.item())
            all_RLNE.append(RLNE)
            all_RLNE1.append(RLNE1)
            all_PSNR.append(PSNR)
            all_PSNR1.append(PSNR1)
            all_SSIM.append(SSIM)
            all_SSIM1.append(SSIM1)

        all_loss = np.array(all_loss)
        all_RLNE = np.array(all_RLNE)
        all_RLNE1 = np.array(all_RLNE1)
        all_PSNR = np.array(all_PSNR)
        all_PSNR1 = np.array(all_PSNR1)
        all_SSIM = np.array(all_SSIM)
        all_SSIM1 = np.array(all_SSIM1)



        print('val---------', "[实际MSE_loss: %f] : [RLNE: 之后%f/之前%f] : [PSNR: 之后%f/之前%f]: [SSIM: 之后%f/之前%f] " % (
           all_loss.sum()/all_loss.size, all_RLNE.sum() / (all_RLNE.size),all_RLNE1.sum() / (all_RLNE1.size),
            all_PSNR.sum() / (all_PSNR.size), all_PSNR1.sum() / (all_PSNR1.size), all_SSIM.sum()/all_SSIM.size, all_SSIM1.sum()/all_SSIM1.size))
        torch.cuda.empty_cache()

        return  all_loss.sum()/all_loss.size



if __name__ == "__main__":
    tic = time.time()
    network = net()  # 实例化
    network.train()
    toc = time.time()
    print('训练所用的时间：', toc - tic, 's')
