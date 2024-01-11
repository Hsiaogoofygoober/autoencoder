#!/bin/bash
# use_spectral_norm_D whether to add spectral norm to D, it helps improve results


input="/home/zach/test_c101/1"
# input='/home2/mura/daniel/cf_mura_autoencoder/test_img/spot'

# modelpath="/home/mura/Min/localization/model/conv_1024_3_layers_32_16_8_k3s2p1_CyclicLR/200.pth"
modelpath="/home/zach/autoencoder/MODEL_C101/V1/200.pth"
imagepath="../test_C101_resultRGB"

batchs=1
num_workers=6
encodesize=1024
devices="0"

for th_percent in $(seq 0.01 0.01 0.01)
do
    for min_area in $(seq 10 10 10)
    do
        python test_save_image.py \
        --input=$input  --modelpath=$modelpath --imagepath=$imagepath \
        --batchs=$batchs --num_workers=$num_workers \
        --encodesize=$encodesize \
        --devices=$devices \
        --th_percent=$th_percent --min_area=$min_area
    done
done

