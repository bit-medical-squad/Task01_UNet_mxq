"""

训练脚本
"""

# from tqdm import tqdm
import os
import argparse

from time import time

import torch
from torch.utils.data import DataLoader
from dataloaders.Dataset import Data2d
import utils.losses as loss
from networks.unet import UNet2D
import torch.nn as nn


import SimpleITK as sitk
from skimage import measure
import scipy.ndimage as ndimage
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")

parser = argparse.ArgumentParser()

parser.add_argument("--val_GT", type=str, default='/share/xianqim/UNet/data/label_val')
parser.add_argument("--val_CT", type=str,default='/share/xianqim/UNet/data/img_val',)
parser.add_argument("--chk_dir", type=str, default='/share/xianqim/UNet/model')

parser.add_argument('--max_iterations', type=int,  default=500, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.00001, help='maximum epoch number to train')
parser.add_argument('--gpu', type=str,  default='5', help='GPU to use')
parser.add_argument('--go_on', type=bool,  default=False, help='weather continue train')
parser.add_argument('--go_on_chk', type=str)

args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

val_GT=args.val_GT
val_CT=args.val_CT
chk_dir=args.chk_dir

go_on=args.go_on
go_on_chk=args.go_on_chk
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr



def plot_progress(epoch,all_tr_losses,all_val_losses,all_val_eval_metrics,chk_dir):

    font = {'weight': 'normal','size': 18}

    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(30, 24))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    x_values = list(range(epoch + 1))

    ax.plot(x_values, all_tr_losses, color='b', ls='-', label="loss_tr")

    ax.plot(x_values, all_val_losses, color='r', ls='-', label="loss_val, train=False")

    if len(all_val_eval_metrics) == len(x_values):
        ax2.plot(x_values, all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2.set_ylabel("evaluation metric")
    ax.legend()
    ax2.legend(loc=9)

    fig.savefig(os.path.join(chk_dir, "progress.png"))
    plt.close()

if __name__ == "__main__":

    net=UNet2D.cuda()
    if go_on:
        net.load_state_dict(torch.load(go_on_chk))
    net.train()

    #定义损失函数
    loss2d=nn.BCELoss()

    #加载数据
    trainloader = DataLoader(Data2d, batch_size=batch_size, shuffle=True, pin_memory=True)

    #定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    #学习率衰减
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    start = time()
    mean_loss_list=[]
    mean_valloss_list=[]
    mean_dice_list=[]
    for epoch in range(max_iterations):
        scheduler.step()
        mean_loss = []
        
        for step, (volume,label)in enumerate(trainloader):
            volume_batch, label_batch = volume.cuda(), label.cuda()
            outputs = net(volume_batch)
    
            label_batch = label_batch.unsqueeze(1)
            loss=loss2d(outputs, label_batch)


            with torch.no_grad():
                mean_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                      .format(epoch, step, loss.item(),(time() - start) / 60))


        with torch.no_grad():
            mean_loss = sum(mean_loss) / len(mean_loss)
            mean_loss_list.append(mean_loss)


        dice_list = []
        mean_loss=[]
        for file_index, file in enumerate(os.listdir(val_CT)):
        
            # 将CT读入内存
            volume = sitk.ReadImage(os.path.join(val_CT, file), sitk.sitkInt16)
            volume_array = sitk.GetArrayFromImage(volume)
            seg = sitk.ReadImage(os.path.join(val_GT,file), sitk.sitkUInt8)
            seg_array = sitk.GetArrayFromImage(seg)
            with torch.no_grad():

                volume_tensor = torch.FloatTensor(volume_array).cuda()
                volume_tensor = volume_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
                label_tensor = torch.FloatTensor(seg_array).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
                outputs = net(volume_tensor)

                loss=loss2d(outputs, label_tensor)
                mean_loss.append(loss.item())

                outputs = outputs.squeeze()

        
            pred_seg = outputs.cpu().detach().numpy()
        

            
            pred_seg[pred_seg > 0.5] = 1
            pred_seg = pred_seg.astype(np.uint8)

            dice = (2 * pred_seg * seg_array).sum() / (pred_seg.sum() + seg_array.sum())
            print('file: {}, dice: {:.3f}'.format(file, dice))
            dice_list.append(dice)

        mean_loss = sum(mean_loss) / len(mean_loss)
        mean_valloss_list.append(mean_loss)

        mean_dice=sum(dice_list) / len(dice_list)
        mean_dice_list.append(mean_dice)
        print('mean dice: {}'.format(sum(dice_list) / len(dice_list)))
        

        plot_progress(epoch,mean_loss_list,mean_valloss_list,mean_dice_list,chk_dir)

        if epoch % 10 == 0:
            # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
            save_mode_path = os.path.join(chk_dir, 'net_{}-{:.3f}-{:.3f}.pth'.format(epoch,loss.item(),mean_loss))
            torch.save(net.state_dict(),save_mode_path)

        


