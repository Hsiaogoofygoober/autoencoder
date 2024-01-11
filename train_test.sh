#!/bin/bash
# use_spectral_norm_D whether to add spectral norm to D, it helps improve results

traininput="/home/mura/mura_data/cf_mura/normal"
savedir="/home/mura/AutoEncoder/NYCU_pytorch_AE/model"
testname="autoencoder_1024_256_tanh_sigmoid_2_layers"
epochs=200
batchs=64
lr=0.001
num_workers=6
encodesize=1024
decodesize=256
devices="0"

python train.py \
 --input=$traininput  --savedir=$savedir --testname=$testname \
 --epochs=$epochs --batchs=$batchs --num_workers=$num_workers \
 --encodesize=$encodesize --decodesize=$decodesize \
 --devices=$devices

#!/bin/bash
# use_spectral_norm_D whether to add spectral norm to D, it helps improve results

testinput="/home/mura/mura_data/cf_mura/mura"
modelpath="/home/mura/AutoEncoder/NYCU_pytorch_AE/model/autoencoder_1024_256_tanh_sigmoid_2_layers/200.pth"

python image_MAE_MSE.py \
 --input=$testinput  --modelpath=$modelpath \
 --batchs=$batchs --num_workers=$num_workers \
 --encodesize=$encodesize \
 --devices=$devices