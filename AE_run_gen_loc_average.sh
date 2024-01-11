#!/bin/bash
# ===== basic =====
base_dir="./detect_position/"
results_dir="./detect_position/"
loadSize=1024
# measure_mode="MAE"
gpu_ids="0"

resolution="resized"
isPadding=1 # padding = true
isResize=1 # Resize = true

# overlap_strategy="average"


# sup_th_strategy='dynamic'
# sup_th_strategy='fixed'
# declare min_area_list=(25)
# declare grad_th_list=(0.5)

dataset_version="cf_mura_b2"
unsup_test_normal_path="/home/mura/mura_data/d23_merge/test/test_normal_8k/" # for unsupervised model
unsup_test_smura_path="/home/mura/mura_data/cf_mura_b2/img/" # for unsupervised model
normal_num=0
smura_num=5370

# ===== generate ground truth =====
data_dir='/home/mura/mura_data/'
save_dir='/home/mura/Min/localization/AutoEncoder_mura_copy/detect_position/'
python3 /home/mura/Min/localization/AutoEncoder_mura_copy/detect_position/code/draw_and_create_ground_truth/dc_gt.py \
-dv=$dataset_version \
-dd=$data_dir \
-sd=$save_dir \
-rs=$isResize


# # plot gt
# data_dir="/home/mura/Min/localization/cf_mura_b2"
# gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
# csv_path="${base_dir}/${dataset_version}/${dataset_version}.csv"
# save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/${overlap_strategy}/${th}_diff_pos_area_${min_area}/imgs_gt"
# python3 ./detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
# -cp=$csv_path \
# -dd=$data_dir \
# -gd=$gt_dir \
# -sd=$save_dir \
# -rs=$isResize # for resizing

# # cal dice and recall & precision
# gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
# save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/${overlap_strategy}/${th}_diff_pos_area_${min_area}"
# python3 ./detect_position/code/calculate_metrics/calculate_metrics.py \
# -dd=$data_dir \
# -gd=$gt_dir \
# -sd=$save_dir


# data_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/${overlap_strategy}/"
# save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}"
# python3 ./detect_position/code/summary_exp_result/summary_exp_result.py \
# -dd=$data_dir \
# -sd=$save_dir \
# -os=$overlap_strategy

# ===== combine sup ===== 要跑結合必須 unsup 要先跑!
# if [ $sup_th_strategy == 'dynamic' ];
# then
#     for topk in $(seq 0.01 0.01 0.10) # 0.01 = 1% ~ 0.10 = 10%
#     do
#         # generate gradcam
#         python3 sup_gradcam.py \
#         --data_version=$dataset_version --loadSize=$loadSize --testing_smura_dataroot=$unsup_test_smura_path \
#         --sup_model_version=$sup_model_version \
#         --checkpoints_dir=$checkpoints_dir \
#         --resolution=$resolution \
#         --sup_th_strategy=$sup_th_strategy \
#         --top_k=$topk \
#         --gpu_ids=$gpu_ids

#         # plot gt
#         data_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam//${sup_model_version}/${sup_th_strategy}/${topk}/imgs"
#         gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
#         csv_path="${base_dir}/${dataset_version}/${dataset_version}.csv"
#         save_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam/${sup_model_version}/${sup_th_strategy}/${topk}/imgs_gt"
#         python3 ./detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
#         -cp=$csv_path \
#         -dd=$data_dir \
#         -gd=$gt_dir \
#         -sd=$save_dir \
#         -rs=$isResize # for resizing

#         # cal dice and recall & precision
#         gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
#         save_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam/${sup_model_version}/${sup_th_strategy}/${topk}"
#         python3 ./detect_position/code/calculate_metrics/calculate_metrics.py \
#         -dd=$data_dir \
#         -gd=$gt_dir \
#         -sd=$save_dir

#         for un_th in $(seq 0.010 0.01 0.100) # 1% ~ 5%
#         do
#             for min_area in ${min_area_list[@]}
#             do
#                 # combine gradcam
#                 sup_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam/${sup_model_version}/${sup_th_strategy}/${topk}/imgs"
#                 unsup_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/${overlap_strategy}/${un_th}_diff_pos_area_${min_area}/imgs"
#                 save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/sup_${topk}_unsup_${un_th}_${min_area}/imgs"
#                 python3 ./detect_position/code/combine_gradcam/combine_gradcam.py \
#                 -upd=$unsup_dir \
#                 -spd=$sup_dir \
#                 -sd=$save_dir

#                 # plot gt
#                 data_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/sup_${topk}_unsup_${un_th}_${min_area}/imgs"
#                 gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
#                 csv_path="${base_dir}/${dataset_version}/${dataset_version}.csv"
#                 save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/sup_${topk}_unsup_${un_th}_${min_area}/imgs_gt"
#                 python3 ./detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
#                 -cp=$csv_path \
#                 -dd=$data_dir \
#                 -gd=$gt_dir \
#                 -sd=$save_dir \
#                 -rs=$isResize # for resizing

#                 # cal dice and recall & precision
#                 gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
#                 save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/sup_${topk}_unsup_${un_th}_${min_area}"
#                 python3 ./detect_position/code/calculate_metrics/calculate_metrics.py \
#                 -dd=$data_dir \
#                 -gd=$gt_dir \
#                 -sd=$save_dir
#             done
#         done
#     done

#     data_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/"
#     save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}"
#     python3 ./detect_position/code/summary_exp_result/summary_exp_result.py \
#     -dd=$data_dir \
#     -sd=$save_dir \
#     -os=$overlap_strategy \
#     -ic
    
# elif [ $sup_th_strategy == 'fixed' ];
# then
#     for grad_th in ${grad_th_list[@]}
#     do
#         # generate gradcam
#         python3 sup_gradcam.py \
#         --data_version=$dataset_version --loadSize=$loadSize --testing_smura_dataroot=$unsup_test_smura_path \
#         --sup_model_version=$sup_model_version \
#         --checkpoints_dir=$checkpoints_dir \
#         --resolution=$resolution \
#         --sup_th_strategy=$sup_th_strategy \
#         --sup_gradcam_th=$grad_th \
#         --gpu_ids=$gpu_ids

#         # plot gt
#         data_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam//${sup_model_version}/${sup_th_strategy}/${grad_th}/imgs"
#         gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
#         csv_path="${base_dir}/${dataset_version}/${dataset_version}.csv"
#         save_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam/${sup_model_version}/${sup_th_strategy}/${grad_th}/imgs_gt"
#         python3 ./detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
#         -cp=$csv_path \
#         -dd=$data_dir \
#         -gd=$gt_dir \
#         -sd=$save_dir \
#         -rs=$isResize # for resizing

#         # cal dice and recall & precision
#         gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
#         save_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam/${sup_model_version}/${sup_th_strategy}/${grad_th}"
#         python3 ./detect_position/code/calculate_metrics/calculate_metrics.py \
#         -dd=$data_dir \
#         -gd=$gt_dir \
#         -sd=$save_dir

#         for un_th in $(seq 0.100)
#         do
#             for min_area in ${min_area_list[@]}
#             do
#                 # combine gradcam
#                 sup_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam//${sup_model_version}/${sup_th_strategy}/${grad_th}/imgs"
#                 unsup_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/${overlap_strategy}/${un_th}_diff_pos_area_${min_area}/imgs"
#                 save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/sup_${grad_th}_unsup_${un_th}_${min_area}/imgs"
#                 python3 ./detect_position/code/combine_gradcam/combine_gradcam.py \
#                 -upd=$unsup_dir \
#                 -spd=$sup_dir \
#                 -sd=$save_dir

#                 # plot gt
#                 data_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/sup_${grad_th}_unsup_${un_th}_${min_area}/imgs"
#                 gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
#                 csv_path="${base_dir}/${dataset_version}/${dataset_version}.csv"
#                 save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/sup_${grad_th}_unsup_${un_th}_${min_area}/imgs_gt"
#                 python3 ./detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
#                 -cp=$csv_path \
#                 -dd=$data_dir \
#                 -gd=$gt_dir \
#                 -sd=$save_dir \
#                 -rs=$isResize # for resizing

#                 # cal dice and recall & precision
#                 gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
#                 save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/sup_${grad_th}_unsup_${un_th}_${min_area}"
#                 python3 ./detect_position/code/calculate_metrics/calculate_metrics.py \
#                 -dd=$data_dir \
#                 -gd=$gt_dir \
#                 -sd=$save_dir
#             done
#         done        
#     done

#     data_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/"
#     save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}"
#     python3 ./detect_position/code/summary_exp_result/summary_exp_result.py \
#     -dd=$data_dir \
#     -sd=$save_dir \
#     -os=$overlap_strategy \
#     -ic
# fi



