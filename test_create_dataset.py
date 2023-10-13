import os

import torch
from PIL import Image

# from fit_sunet import pre2Lab2RGB
from torch.utils import data
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import cv2
import torchvision.transforms.functional as TF
import random
from skimage.color import rgb2lab, lab2rgb



class Mydataset_su_info(data.Dataset):
    def __init__(self, image_names, transform):
        self.img_name = image_names
        self.transforms = transform

    def __getitem__(self, index):
        image = self.img_name[index]
        name = image.split('/')[-1]
        # gt_name = "/imagenet_256_val/" + name
        # gt_name = '/mnt/ai2022/th/data_imagenet/imagenet_256_val/' + name
        inp_img = Image.open(image).convert('RGB')
        # tar_img = Image.open(gt_name).convert('RGB')
        inp_img = TF.to_tensor(inp_img)
        # tar_img = TF.to_tensor(tar_img)


        return inp_img,  name

    def __len__(self):
        return len(self.img_name)
def train_transform_seg():
    train_transform = A.Compose([
        # A.Resize(224, 224),
        # A.RandomRotate90(),
        # A.Flip(p=0.5),

        ToTensorV2()], p=1.)
    return train_transform


test_transform_seg = A.Compose([
    # A.Resize(224, 224),
    ToTensorV2()], p=1.)


class Mydataset_inference_val(data.Dataset):
    def __init__(self, image_names, transform):
        self.img_name = image_names
        self.transforms = transform

    def __getitem__(self, index):

        image = self.img_name[index]
        name = image.split('/')[-1]
        img = cv2.imread(image)[:, :, ::-1]
        img = cv2.resize(cv2.imread(image),(256,256))[:, :, ::-1]
        lab_img = rgb2lab(img)
        L = lab_img[:, :, 0]
        h, w = L.shape
        L = L.reshape(h, w, 1)

        data = self.transforms(image=L)
        return data['image'], name

    def __len__(self):
        return len(self.img_name)

def pre2Lab2RGB(predicted):
    codebook = np.load('/mnt/ai2022/th/MY/Two-stage_Image_Colorization_via_Color_Codebook/color_codebook/codebook_n697.npy', allow_pickle=True).item()
    batch, h, w = predicted.shape
    predicted_rgb = np.ones((batch, h, w, 3))
    predicted = predicted.cpu().numpy()

    for iter, i in enumerate(predicted):
        for row in range(len(i)):
            for col in range(len(i[row])):
                element = i[row][col]
                x = codebook[element]
                predicted_rgb[iter, row, col] = x

    for _, i in enumerate(predicted_rgb):
        predicted_rgb[_] = lab2rgb(i)
    return predicted_rgb

def set_seed(seed):  # seed的数值可以随意设置
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)