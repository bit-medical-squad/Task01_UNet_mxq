import os
import argparse
import torch
from time import time
import numpy as np
import SimpleITK as sitk

from networks.unet import UNet2D



parser = argparse.ArgumentParser()

parser.add_argument("--test_GT", type=str, default='/share/xianqim/UNet/data/label_val')
parser.add_argument("--test_CT", type=str,default='/share/xianqim/UNet/data/img_val')
parser.add_argument("--pred_dir", type=str, default='/share/xianqim/UNet/data/pred')
parser.add_argument("--chk_dir", type=str, default='/share/xianqim/UNet/model/net_300-0.059-0.066.pth')
parser.add_argument('--gpu', type=str,  default='5', help='GPU to use')


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


test_GT=args.test_GT
test_CT=args.test_CT
pred_dir=args.pred_dir
chk_dir=args.chk_dir



if __name__ == "__main__":
    net = UNet2D.cuda()
    net.load_state_dict(torch.load(chk_dir))
    net.eval()

    dice_list=[]
    seg_31=[]
    pred_31=[]
    seg_32=[]
    pred_32=[]
    seg_34=[]
    pred_34=[]

    for file_index, file in enumerate(os.listdir(test_CT)):
        volume = sitk.ReadImage(os.path.join(test_CT, file), sitk.sitkInt16)
        volume_array = sitk.GetArrayFromImage(volume)
        seg = sitk.ReadImage(os.path.join(test_GT,file), sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        with torch.no_grad():

            volume_tensor = torch.FloatTensor(volume_array).cuda()
            volume_tensor = volume_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            label_tensor = torch.FloatTensor(seg_array).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
            outputs = net(volume_tensor).squeeze(0).squeeze(0)   
    
            pred_seg = outputs.cpu().detach().numpy()
            pred_seg[pred_seg > 0.5] = 1
            pred_seg = pred_seg.astype(np.uint8)


            dice = (2 * pred_seg * seg_array).sum() / (pred_seg.sum() + seg_array.sum())
            print('file: {}, dice: {:.3f}'.format(file, dice))
            dice_list.append(dice)
            
            pred=sitk.GetImageFromArray(pred_seg)
            sitk.WriteImage(pred,os.path.join(pred_dir,file))

            if file.split('_')[0].split('-')[-1]=='31':
                seg_31.append(seg_array)
                pred_31.append(pred_seg)
            elif file.split('_')[0].split('-')[-1]=='32':
                seg_32.append(seg_array)
                pred_32.append(pred_seg)
            elif file.split('_')[0].split('-')[-1]=='34':
                seg_34.append(seg_array)
                pred_34.append(pred_seg)

    print('mean dice: {}'.format(sum(dice_list) / len(dice_list)))

    dice_31 = (2 * np.array(pred_31) *  np.array(seg_31)).sum().sum() / ( np.array(pred_31).sum().sum() +  np.array(seg_31).sum().sum())
    print('31dice',dice_31)

    dice_32 = (2 * np.array(pred_32) *  np.array(seg_32)).sum().sum() / ( np.array(pred_32).sum().sum() +  np.array(seg_32).sum().sum())
    print('32dice',dice_32)

    dice_34 = (2 * np.array(pred_34) *  np.array(seg_34)).sum().sum() / ( np.array(pred_34).sum().sum() +  np.array(seg_34).sum().sum())
    print('34dice',dice_34)




