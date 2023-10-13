from models.model.SUNet_ori_pre import SUNet_model
import yaml
from models.utils import network_parameters
import glob
import os
import time
from torch.utils.data import DataLoader
from test_create_dataset import test_transform_seg, Mydataset_su_info, set_seed
import argparse
import torch
import numpy as np
import cv2
from tqdm import tqdm
from metrics.psnr_lpips_ssim_colorfulness_fid import avg_ssim_psnr, avg_lpips, calculate_colorfulness, calculate_fid

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--task', type=str, default='lightweight_sr', help='classical_sr, lightweight_sr, real_sr, '
                                                                         'gray_dn, color_dn, jpeg_car, color_jpeg_car')
parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')
parser.add_argument('--lq_data', default='/mnt/ai2022/th/MY/Two-stage_Image_Colorization_via_Color_Codebook/save_data/fist', type=str,
                    help='path to val no1_data directory')
parser.add_argument('--gt_data', default='', type=str,
                    help='path to gt directory')
parser.add_argument('--Weights_path', default='/mnt/ai2022/th/MY/Two-stage_Image_Colorization_via_Color_Codebook/checkpoints/second.pth', type=str, help='load Weights_path')
parser.add_argument('--batch_size', default=12, type=int, help='batchsize')
parser.add_argument('--workers', default=16, type=int, help='batchsize')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--seed', default='2023', type=int, help='seed_num')
parser.add_argument('--device', default='cuda', type=str, help='use cuda')
parser.add_argument('--save_path', type=str, default='/mnt/ai2022/th/MY/Two-stage_Image_Colorization_via_Color_Codebook/save_data/second', help='checkpoints path')
parser.add_argument('--devicenum', default='0', type=str, help='use devicenum')

args = parser.parse_args()
opt = {}
opt['n_thread'] = 13
os.environ['CUDA_VISIBLE_DEVICES'] = args.devicenum
set_seed(seed=args.seed)
begin_time = time.time()

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = args.device

print('Start loading no1_data.')
# 加载图片路径
val_path = args.val_data
imgs_val = glob.glob(val_path+ '/*')


after_read_date = time.time()
print('data_time', after_read_date - begin_time)
print('imgs done')


test_transform =test_transform_seg
best_acc_final = []
## Load yaml configuration file
with open('./models/training_pre_ori.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']


val_dataset = Mydataset_su_info(imgs_val, test_transform)
valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

def inference(model):
    masks_savepath = args.gt_data
    predicted_savepath = args.save_path
    if not os.path.exists(predicted_savepath):
        os.makedirs(predicted_savepath)
    model.eval()
    # with torch.no_grad():
    pbar = tqdm(total=len(valloader), unit='batch', desc='Extract')
    for batch_idx, (imgs, name) in enumerate(valloader):
        imgs = imgs.to(device)
        imgs = imgs.float()
        with torch.no_grad():
            masks_pred = model(imgs)
            masks_pred = torch.sigmoid(masks_pred)


            for i in range(imgs.shape[0]):

                # 第二阶段用
                output1 = masks_pred[i].data.float().cpu().clamp_(0, 1).numpy()
                if output1.ndim == 3:
                    output1 = np.transpose(output1[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                output1 = (output1 * 255.0).round().astype(np.uint8)  # float32 to uint8
                cv2.imwrite(predicted_savepath + '/' + name[i], output1)

                pbar.update(1)

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

    ## Build Model
    print('==> Build the model')
    model = SUNet_model(opt)
    model.load_from(opt)
    p_number = network_parameters(model)
    model.cuda()
    model = torch.nn.DataParallel(model)
    Weights_path = args.Weights_path

    model.load_state_dict(torch.load(Weights_path))
    model = model.to(device)
    inference(model)

after_net_time = time.time()
print('net_time', after_net_time - after_read_date)
print('finish')
