import pywt
import numpy as np
import SimpleITK as sitk
import sys
sys.path.append('/home/vision/diska4/shy/NerfDiff/Wavelet/lib')
import os

from util import *


def RawDataProcess(Data_path, save_path):

    Datalist = os.listdir(Data_path)
    for name in Datalist:
        print(name)
        vol_tensor = read_mha_asnp(os.path.join(Data_path, name))
        # print(vol_tensor.shape)
        vol_tensor = vol_tensor[:, :, 30:80]
        slice_num = 50
        for i in range(slice_num):
            vol_slice = vol_tensor[:, :, i]
            # print(vol_slice)
            save_img_tensor(f'./test{i}.png', vol_slice)
        exit(0)

import SimpleITK as sitk
import torch
from torch import Tensor
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2


import sys
sys.path.append('..')



def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return False
    else:
        return True

def normalize(img, convert2uint8=False):
    if convert2uint8:
        return ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
    else:
        return (img - np.min(img)) / (np.max(img) - np.min(img))


def Convert_Channel(image): 
    return cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)


def Sample_Slice(volume_dir, save_dir, convert_type=True, convert_channel=True):
    try:
        volume_list = os.listdir(volume_dir)
    except:
        raise f'Wrong volume dir. Got {volume_dir}'

    exists_or_mkdir(save_dir)

    print(f'Volume Number : {len(volume_list)}')
    print(f'Save Path : {save_dir}')
    for _, volume_name in tqdm(enumerate(volume_list), total=len(volume_list)):

        volume = sitk.ReadImage(os.path.join(volume_dir, volume_name))
        volume_array = sitk.GetArrayFromImage(volume)[30:80, :, :]

        slice_num = 50

        for i in range(slice_num):
            img = volume_array[i]
            if convert_type:
                img = normalize(img, True)
            if convert_channel:
                img = Convert_Channel(img)
            cv2.imwrite(os.path.join(save_dir, volume_name[:-7] + f'_{i}' + '.png'), img)
            

def Sample_volume(volume_array, index):

    img = volume_array[index]
    img = normalize(img, True)
    img = Convert_Channel(img)
    # img_list.append(img)
    return img



def imgshow(img, name='test'):
    plt.figure(figsize=(20,20))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.savefig(f'./{name}.png')

def Wavelet_transform(img_dir):
    imglist = sorted(os.listdir(img_dir))
    for name in imglist:
        print(name)
        img = read_img_asnp(os.path.join(img_dir, name))
        print(img)
        # print(img.shape)
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        img = cv2.resize(img, None, fx=2, fy=2)
        cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
        print(cA[32])
        # print(img.shape)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        # img = normalize(img, True)
        # img = normalize(img)
        # print(img)
        # img = Convert_Channel(img)
        # cv2.imwrite('./ttt.png', img)
        break

if __name__ == '__main__':
    img_dir = '/home/vision/diska4/shy/NerfDiff/data/LIDC/CTslice'
    Wavelet_transform(img_dir)
    # vol_dir = '/home/vision/diska4/shy/NerfDiff/data/LIDC/CT'
    # save_dir = '/home/vision/diska4/shy/NerfDiff/data/LIDC/CTslice'
    # Sample_Slice(vol_dir, save_dir, True, False)


