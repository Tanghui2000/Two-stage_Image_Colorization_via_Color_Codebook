import glob
import os

import time
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
from test_create_dataset import test_transform_seg, Mydataset_inference_val, pre2Lab2RGB, set_seed

import argparse
import torch
import numpy as np
import cv2

from metrics.psnr_lpips_ssim_colorfulness_fid import avg_ssim_psnr, avg_lpips, calculate_colorfulness, calculate_fid

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('--val_data', default='/imagenet_256_val', type=str,
                    help='path to val no1_data directory')
parser.add_argument('--Weights_path',
                    default='/mnt/ai2022/th/MY/Two-stage_Image_Colorization_via_Color_Codebook/checkpoints/first.pth', type=str,
                    help='load Weights_path')

parser.add_argument('--batch_size', default=12, type=int, help='batchsize')
parser.add_argument('--workers', default=16, type=int, help='batchsize')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--device', default='cuda', type=str, help='use cuda')
parser.add_argument('--save_path', type=str, default='/mnt/ai2022/th/MY/Two-stage_Image_Colorization_via_Color_Codebook/save_data/first', help='checkpoints path')
parser.add_argument('--devicenum', default='0', type=str, help='use devicenum')
parser.add_argument('--seed', default='2023', type=int, help='seed_num')
parser.add_argument('--num_classes', type=int,
                    default=697, help='output channel of network')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.devicenum

set_seed(seed=args.seed)

begin_time = time.time()

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = args.device


print('Start loading no1_data.')
# 加载图片路径
val_path = args.val_data
imgs_val = glob.glob(val_path + '/*')

after_read_date = time.time()
print('data_time', after_read_date - begin_time)
print('imgs done')

test_transform = test_transform_seg
best_acc_final = []

path = args.Weights_path


def visualize(path):
    masks_savepath = val_path
    predicted_savepath = args.save_path
    if not os.path.exists(predicted_savepath):
        os.makedirs(predicted_savepath)
    model = smp.Unet(in_channels=1, encoder_name='efficientnet-b3', encoder_weights='imagenet', classes=697).to(device)
    # 加载数据集
    val_dataset = Mydataset_inference_val(imgs_val, test_transform)
    # 创建数据加载器
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    # 加载保存的权重

    model.load_state_dict(torch.load(path))
    model.eval()

    pbar = tqdm(total=len(val_loader), unit='batch', desc='Extract')
    for batch_idx, (imgs, name) in enumerate(val_loader):
        imgs = imgs.to(device)
        imgs = imgs.float()
        with torch.no_grad():
            output = model(imgs)
        output = output.argmax(1)
        output_RGB = pre2Lab2RGB(output)

        for i in range(imgs.shape[0]):
            if os.path.exists(name[i]):
                pass
            else:
                predicted_save = output_RGB[i]
                predicted_save = (predicted_save - np.min(predicted_save)) * 255 / (
                        np.max(predicted_save) - np.min(predicted_save))
                predicted_save = predicted_save.astype(np.uint8)
                predicted_save = cv2.cvtColor(predicted_save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(predicted_savepath + '/' + name[i], predicted_save)
        pbar.update(1)

    pbar.close()

    # 计算指标
    fid_score, fid_score_convert = calculate_fid(predicted_savepath, masks_savepath)
    colorfulness_ave = calculate_colorfulness(predicted_savepath)
    lpips, lpips_avg_convert = avg_lpips(predicted_savepath, masks_savepath)
    _, psnr, _, psnr_convert = avg_ssim_psnr(predicted_savepath, masks_savepath)
    pbar.close()

    model_savedir=args.save_path
    f = open(model_savedir + '_log' + '.txt', "a")
    f.write(' psnr_ave' + str(psnr) + ' lpips_ave' + str(lpips) +
            ' colorfulness_ave' + str(colorfulness_ave) + ' fid_ave' + str(fid_score)
            + 'psnr_convert' + str(psnr_convert) + 'lpips_avg_convert' + str(lpips_avg_convert) + '\n')
    f.close()

    print('successfully.')


if __name__ == '__main__':
    visualize(path)

after_net_time = time.time()
print('net_time', after_net_time - after_read_date)
print('finish')
