# 資料相關 files
1. dataloader.py
2. dataset.py

# 模型相關 file
1. model.py

# 參數設定相關 file
1. option.py

# 訓練 & 測試 files
1. train_conv.py
```bash
bash train_conv.sh
```
2. test_save_image.py
```bash
bash test_save_image.sh
```

# 其他常用 files
1. draw_and_create_ground_truth (生成 bounding box 在原圖上 & 生成 ground truth)
2. plot_gt_on_result (生成 bounding box 在二值化圖上)
3. calculate_metrics (計算 dice mean recall precision 命中張數)
