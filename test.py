import argparse
import os

# from metrics import calculate_ave_psnr_ssim_lpips, calculate_colorfulness, calculate_fid
from psnr_lpips_ssim_colorfulness_fid import avg_ssim_psnr, avg_lpips, calculate_colorfulness, calculate_fid
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--gt', default='/mnt/ai2022/th/data_imagenet/imagenet_256_val', type=str,help='path to gt')
parser.add_argument('--pre_data', default='/mnt/ai2022/th/data_imagenet/inference/sunet/pre2_coco_val2017_256_5k_ckpt2', type=str,help='path to pre_data directory')
parser.add_argument('--save_name', default='sunet_coco_val2017_256_5k_ckpt2', type=str, help='save_name')
parser.add_argument('--savedir', default='/mnt/ai2022/th/color_imagenet/test_codebookxr', type=str, help='savedir')
parser.add_argument('--devicenum', default='1', type=str, help='use devicenum')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.devicenum
masks_savepath =args.gt
predicted_savepath= args.pre_data
gt_path = args.gt
savedir = args.savedir + '/'
name = args.save_name


lpips ,lpips_avg_convert = avg_lpips(predicted_savepath, masks_savepath)
fid_score, fid_score_convert = calculate_fid(predicted_savepath, masks_savepath)
colorfulness_ave = calculate_colorfulness(gt_path)
ssim, psnr, ssim_convert, psnr_convert =avg_ssim_psnr(predicted_savepath, masks_savepath)


# print(colorfulness_ave)
f = open(savedir + name + '.txt', "a")
f.write(' psnr_ave' + str(psnr) + ' ssim_ave' + str(ssim) + ' lpips_ave' + str(lpips) +
        ' colorfulness_ave' + str(colorfulness_ave) + ' fid_ave' + str(fid_score)  + '\n'
        ' psnr_convert' + str(psnr_convert) + ' ssim_convert' + str(ssim_convert) + 'lpips_avg_convert' + str(lpips_avg_convert) + '\n')
f.close()

