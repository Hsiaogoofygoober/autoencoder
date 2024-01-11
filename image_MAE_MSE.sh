#!/bin/bash
# use_spectral_norm_D whether to add spectral norm to D, it helps improve results

input="/home/mura/mura_data/cf_mura/mura"
modelpath="/home/mura/AutoEncoder/NYCU_pytorch_AE/model/autoencoder_conv_1024_256_relu_sigmoid_3_layers_32_16_8_CyclicLR/200.pth"
batchs=64
num_workers=6
encodesize=1024
devices="0"

python image_MAE_MSE.py \
 --input=$input  --modelpath=$modelpath \
 --batchs=$batchs --num_workers=$num_workers \
 --encodesize=$encodesize \
 --devices=$devices