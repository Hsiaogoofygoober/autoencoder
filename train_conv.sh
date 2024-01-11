#!/bin/bash
# use_spectral_norm_D whether to add spectral norm to D, it helps improve results

input="/home/zach/c101_1"
savedir="/home/zach/autoencoderRGB/MODEL_C101"
testname="V1"
epochs=200
batchs=16
lr=0.001
num_workers=6
encodesize=1024
decodesize=256
devices="0"

python train_conv.py \
 --input=$input  --savedir=$savedir --testname=$testname \
 --epochs=$epochs --batchs=$batchs --num_workers=$num_workers \
 --encodesize=$encodesize --decodesize=$decodesize \
 --devices=$devices