import cv2
import numpy as np

def find_white_connected_areas(image, threshold=200):
    # 將圖像轉換為灰度
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 二值化圖像
    _, thresholded = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    # 連通元件標記
    _, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded)
    
    # 找到白色區域的面積
    white_area_areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, len(stats))]
    
    return white_area_areas




img = cv2.imread("/home2/mura/daniel/cf_mura_rmeg_minarea_100/0.01_10/fin_res/TD85Q18244N_004_2.jpg", cv2.IMREAD_GRAYSCALE)

if img is not None:
    white_areas = find_white_connected_areas(img)
    print("白色連通區域面積:", white_areas)
else:
    print("無法讀取照片")

# white = np.count_nonzero(img > 200)

# print(white)