# Two-stage Image Colorization via Color Codebook

## Prerequisites
* Python 3.8
* PyTorch 1.12
* NVIDIA GPU + CUDA cuDNN

## Getting Started
1. Clone this repo:
```
git clone https://github.com/Tanghui2000/Two-stage_Image_Colorization_via_Color_Codebook.git
```
```
cd Two-stage_Image_Colorization_via_Color_Codebook
```
2. Install the python requirements
```
pip install -r requirement.txt
```
## Pretrained Model

1. Pretrained models would place in ./checkpoints.

    Download the pretrained model and place it in the `checkpoints/` folder.
    You can download it from <a href="https://pan.baidu.com/s/1aHbL3ZjiEq2RWtxl6ZfB5Q"> here </a>, password：tyv4

2. Color codebook would place in ./color_codebook.

## The first stage: color classification
Please follow the command below for the first stage prediction
```
python pre1_info.py --val_data='/imagenet_256_val' --Weights_path='checkpoints/first.pth' --save_path='save_data/first' --devicenum='0'
```
All the prediction results would save in save_data/first folder.

## The second stage: color refinement
Please follow the command below for the second stage prediction
```
python pre2_info.py --lq_data='save_data/first' --gt_data='/imagenet_256_val' --Weights_path='checkpoints/second.pth' --save_path='save_data/second' --devicenum='0'
```
All the prediction results would save in save_data/second folder.
