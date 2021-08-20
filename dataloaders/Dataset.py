import os

import torch
from torch.utils.data import Dataset as dataset
import SimpleITK as sitk
import numpy as np


class Dataset(dataset):
    def __init__(self, CT_dir, GT_dir):

        self.CT_list = list(map(lambda x: os.path.join(CT_dir, x), os.listdir(CT_dir)))
        self.GT_list = list(map(lambda x: os.path.join(GT_dir, x),  os.listdir(GT_dir)))


    def __getitem__(self, index):

        CT_path = self.CT_list[index]
        GT_path = self.GT_list[index]

        # 将CT和金标准读入到内存中
        CT = sitk.ReadImage(CT_path)
        GT = sitk.ReadImage(GT_path)

        CT_nd = sitk.GetArrayFromImage(CT)
        GT_nd = sitk.GetArrayFromImage(GT)

        if len(CT_nd.shape) == 2:
            CT_nd = np.expand_dims(CT_nd, axis=2)
            GT_nd=np.expand_dims(GT_nd,axis=2)
        # HWC to CHW
            CT = CT_nd.transpose((2, 0, 1))
            GT=GT_nd.transpose((2,0,1))

        # 处理完毕，将array转换为tensor
        CT_array = torch.from_numpy(CT).float()
        GT_array = torch.from_numpy(GT).float().squeeze(0)

        return CT_array, GT_array

    def __len__(self):

        return len(self.CT_list)

CT_dir = "/share/xianqim/UNet/data/img_process_600"
GT_dir = "/share/xianqim/UNet/data/label_600"

Data2d = Dataset(CT_dir, GT_dir)


