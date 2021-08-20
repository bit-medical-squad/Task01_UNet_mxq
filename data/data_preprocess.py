import os
import argparse

import SimpleITK as sitk
import numpy as np



'''
###############
数据预处理，仅进行灰度归一化
归一化方式参考nnUNet
计算全部数据的前景强度信息，然后标准化
################
'''

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--CT_folder', type=str,default='/share/xianqim/UNet/data/img_600')
	parser.add_argument('--outputCT_folder', type=str, default='/share/xianqim/UNet/data/img_process_600')
	parser.add_argument('--GT_folder', type=str,default='/share/xianqim/UNet/data/label_600')

	args = parser.parse_args()

	#获取原始CT,GT存放地址，设置保存预处理后数据地址
	CT_folder=args.CT_folder
	outputCT_folder=args.outputCT_folder
	GT_folder=args.GT_folder

	#统计所有数据的前景像素
	voxels=[]

	for file in os.listdir(CT_folder):
		image=sitk.ReadImage(os.path.join(CT_folder,file))
		image_array=sitk.GetArrayFromImage(image)

		label=sitk.ReadImage(os.path.join(GT_folder,file))
		label_array=sitk.GetArrayFromImage(label)

		voxels=np.append(voxels,image_array[label_array>0])

	#计算强度相关信息
	mean = np.mean(voxels)
	sd = np.std(voxels)
	percentile_99_5 = np.percentile(voxels, 99.5)
	percentile_00_5 = np.percentile(voxels, 00.5)


	#对每个数据进行强度归一化
	for file in os.listdir(CT_folder):
		image=sitk.ReadImage(os.path.join(CT_folder,file))
		image_array=sitk.GetArrayFromImage(image)

		image_array = np.clip(image_array, percentile_00_5, percentile_99_5)#根据所有数据前景强度的5%，裁减掉无用信息
		image_array = (image_array - mean) / sd#根据所有数据的前景强度均值方差进行标准化

		output_image=sitk.GetImageFromArray(image_array)
		sitk.WriteImage(output_image,os.path.join(outputCT_folder,file))


