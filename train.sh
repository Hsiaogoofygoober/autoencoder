#!/bin/bash
# use_spectral_norm_D whether to add spectral norm to D, it helps improve results

input="/home/mura/mura_data/cf_mura/normal"
savedir="/home/mura/daniel/cf_mura_autoencoder/MODEL"
testname="autoencoder_1024_256_tanh_sigmoid_2_layers"
epochs=200
batchs=64
lr=0.001
num_workers=6
encodesize=1024
decodesize=256
devices="1"

python train.py \
 --input=$input  --savedir=$savedir --testname=$testname \
 --epochs=$epochs --batchs=$batchs --num_workers=$num_workers \
 --encodesize=$encodesize --decodesize=$decodesize \
 --devices=$devices