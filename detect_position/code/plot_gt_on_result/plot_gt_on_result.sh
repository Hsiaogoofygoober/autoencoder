#!/bin/bash


data_dir='/home/mura/daniel/skip-ganomaly-master/output/skipganomaly/smura/test/ori_smura'
csv_dir='/home/mura/daniel/skip-ganomaly-master/data/bounding_box/cf_mura_b2/cf_mura_b2.csv'
save_dir='/home/mura/daniel/skip-ganomaly-master/output/skipganomaly/smura/test/ori_smura_gt'
isResize=0

mkdir -p $save_dir

python plot_gt_on_result.py \
-dd=$data_dir \
-cp=$csv_dir \
-sd=$save_dir \
-rs=$isResize