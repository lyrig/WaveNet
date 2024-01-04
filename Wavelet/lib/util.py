import os
import cv2
import torch
import torch.nn.functional as F
from torch.nn import init
from pytorch_msssim import ssim  as ssim_pt
from skimage.metrics import structural_similarity as sk_SSIM
import torchvision
import random
import numpy as np
import SimpleITK as sitk
import torchvision.transforms as transforms
import PIL.Image as Image
import imageio
import argparse
import json


def random_set(seed=None):
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def merge_images(images, fixed_imgs, frame_raw):
    images = torch.split(images, frame_raw, dim=0)
    fixed_imgs = torch.split(fixed_imgs, frame_raw, dim=0)
    merged_images = torch.cat([torch.cat([image, fixed_img], dim=0) for image, fixed_img in zip(images, fixed_imgs)], dim=0)
    return merged_images


def read_img_astensor(file_path, cuda_device=0):
    trans = transforms.Compose([transforms.ToTensor()])
    img = Image.open(file_path).convert('L')
    img = trans(img)
    img = img.cuda(cuda_device)
    return img.squeeze()


def read_mha_astensor(file_path, cuda_device='cpu'):
    sitkimage = sitk.ReadImage(file_path)
    # print(sitkimage.spacing)
    volume = sitk.GetArrayFromImage(sitkimage)
    volume = torch.Tensor(volume).to(cuda_device)
    return volume.squeeze()


def read_img_asnp(file_path):
    trans = transforms.Compose([transforms.ToTensor()])
    img = Image.open(file_path).convert('L')
    img = trans(img)
    img = img.squeeze().numpy()
    return img


def read_mha_asnp(file_path):
    sitkimage = sitk.ReadImage(file_path)
    volume = sitk.GetArrayFromImage(sitkimage)
    return volume.squeeze()


def save_img_tensor(path, img):
    img = img.squeeze()
    img = img.data.cpu().numpy()
    imageio.imwrite(path, img)


def save_mha_tensor(path, mha, spacing=None, isVector=False):
    outputs = mha.squeeze()
    outputs = outputs.data.cpu().numpy()
    volume = sitk.GetImageFromArray(outputs, isVector=isVector)
    if spacing is not None:
        volume.SetSpacing(spacing)
    sitk.WriteImage(volume, path)

def read_img_astensor(file_path, cuda_device='cpu'):
    trans = transforms.Compose([transforms.ToTensor()])
    img = Image.open(file_path).convert('L')
    img = trans(img)
    return img

def normalize(img, convert2uint8=False):
    if convert2uint8:
        return ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
    else:
        return (img - np.min(img)) / (np.max(img) - np.min(img))


def save_img_tensor(path, img):
    img = normalize(img, True)
    cv2.imwrite(path, img)
    return True


def load_config(load_path):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(load_path, 'r') as f:
        args.__dict__ = json.load(f)
    return args


if __name__ == '__main__':


    vol_tensor = read_mha_astensor('/home/vision/diska4/shy/NerfDiff/result/FTNerf/CodeNerf/exp1/101_FTCodeNerf1_iter_1000.nii.gz')
    # print(vol_tensor.shape)
