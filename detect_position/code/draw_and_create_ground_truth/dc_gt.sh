#!/bin/bash
# dataset_version='typec+b1'
# dataset_version='typed'
dataset_version='cf_mura_b2'
data_dir='/home2/mura/daniel/data/smura_classification/Horizon/result/0.01_10/fin_res'
save_dir='/home2/mura/daniel/0921_cf_mura/smura'
resized=1

python dc_gt.py \
-dv=$dataset_version \
-dd=$data_dir \
-sd=$save_dir \
-rs=$resized