import torch.utils.data as data
import os
import torch
import SimpleITK as sitk
import numpy as np
from torchvision.transforms import transforms

def nii2numpy(nii_path):
    itk_img = sitk.ReadImage(nii_path)
    img = sitk.GetArrayFromImage(itk_img)
    out = np.asarray(img) / 1.0
    out = np.expand_dims(out, 0)
    return out  # 1*285*512*512


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if (os.path.splitext(file)[1] == '.nii'):
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


class MyDataset(data.Dataset):
    def __init__(self, imgs_path, masks_path):
        imgs = []
        img = get_listdir(imgs_path)
        mask = get_listdir(masks_path)
        n = len(img)
        for i in range(n):
            imgs.append([img[i], mask[i]])
        self.imgs = imgs
        self.images_transform = transforms.ToTensor()
        self.masks_transform = transforms.ToTensor()

    def __getitem__(self, index):
        images_path, masks_path = self.imgs[index]
        image = torch.tensor(nii2numpy(images_path))
        mask = torch.tensor(nii2numpy(masks_path))
        return image, mask

    def __len__(self):
        return len(self.imgs)

# 1*285*512*512
