#!/bin/bash
th_percent=0.01

for min_area in $(seq 1 1 1)
do
    data_dir="/home/mura/Min/localization/cf_mura_rmeg/${th_percent}_${min_area}/fin_res_sp"
    gt_dir="/home/mura/Min/localization/cf_mura_autoencoder/detect_position/cf_mura_b2/resized/actual_pos/ground_truth"
    save_dir="/home/mura/Min/localization/cf_mura_rmeg/${th_percent}_${min_area}"

    python3 calculate_metrics_per_class.py \
    -dd=$data_dir \
    -gd=$gt_dir \
    -sd=$save_dir
done
