#!/bin/bash
th_percent=0.01

for min_area in $(seq 1 1 1)
do
    data_dir="/home/mura/Min/localization/cf_mura_rmeg/${th_percent}_${min_area}/fin_res"
    xml_dir="/home/mura/mura_data/cf_mura_b2/xml"
    save_dir="/home/mura/Min/localization/cf_mura_rmeg/${th_percent}_${min_area}/fin_res_sp"
    mkdir -p $save_dir
    
    python3 separate_res_class.py \
    -dd=$data_dir \
    -xd=$xml_dir \
    -sd=$save_dir
done

# BlackSpot
# WhiteSpot
# Butterfly
# Dirt
# Horizon
# PinBed
# Vertical
