import os
import cv2
import numpy as np
import random
import pandas as pd
from PIL import Image, ImageDraw
from collections import defaultdict
from glob import glob

target = 'VCD'

def join_path(p1,p2):
    return os.path.join(p1,p2)

def cal_area(region, per):
    satisfying_pixels = np.sum(region == 255)
    total_pixels = region.size
    percentage = (satisfying_pixels / total_pixels) * 100
    # print(satisfying_pixels)
    # print(total_pixels)
    return percentage >= per

def cal_IOU(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    iou_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    are_rect2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    all_area = area_box1 + are_rect2 - iou_area
    return max(iou_area / all_area, 0)

def vertical(x, y, w, h):
    # if (w / h > 10) or (h / w > 10): return True
    # else: return False
    if (h / w > 4) and (x < 1004) and (x > 20) and (w * h > 2000): return True
    else: return False
    
def horizon(w, h):
    if (w / h > 2) and (y < 1004) and (w * h > 2000): return True
    else: return False
    
def spot(x, y, w, h):
    if (w / h < 1.5) and (h / w < 1.5):
        if (w * h) > 500: return True
        elif (x < 1004) and (y < 1004) and (x > 20) and (y > 20) and (w*h) > 100: return True
        else: return False

df = pd.read_csv('/home2/mura/daniel/unknown/cf_mura_test_data/testdata_crop.csv')
# print(df)

# save_dir = f'/home2/mura/daniel/data/smura_classification/{target}/result_pred-input_1029/bounding_box'
# img_folder = f'/home2/mura/daniel/data/smura_classification/{target}/result_pred-input_1029/fin_res'
save_dir = f'/home2/mura/daniel/data/smura_classification/unknown/result_pred-input_1029/bounding_box'
img_folder = f'/home2/mura/daniel/data/smura_classification/unknown/result_pred-input_1029/fin_res'

img_list = glob(f"{join_path(img_folder, '*jpg')}")
# print(img_list)
tp = 0
fp = 0
fn = 0

k = 0
# random_file = random.sample(os.listdir(img_folder), 10)


for img_path in img_list:
    # tp = 0
    # fp = 0
    # fn = 0
    file_name = img_path.split('/')[-1]
    print(file_name)
    # if k == 10:
    #     break
    # k += 1

    img = Image.open(img_path)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    fn_series_list = df[df['fn']==file_name]
    
    # 刪掉target外的瑕疵種類
    mark = 0
    L = fn_series_list.shape[0]
    for i in range(L):
        tmp = fn_series_list.iloc[i-mark]
        if target == 'Spot':
            if tmp['smura_name'] != 'WhiteSpot' and tmp['smura_name'] != "BlackSpot":
                fn_series_list = fn_series_list.drop(fn_series_list.index[i-mark])
                mark += 1
        elif tmp['smura_name'] != target:
            fn_series_list = fn_series_list.drop(fn_series_list.index[i-mark])
            mark += 1
        
    # print(fn_series_list)
    
    if fn_series_list.shape[0] != 0:
        # 計算欲框起來之面積位置
        actual_pos_list = []
        for i in range(0, fn_series_list.shape[0]):
            fn_series = fn_series_list.iloc[i]
            actual_pos_list.append((int(fn_series['x0']/2), int(fn_series['y0']/1.586914), int(fn_series['x1']/ 2), int(fn_series['y1']/1.586914)))
        
        #print(actual_pos_list)

        # 將圖片從RGB轉成BGR 以利下面region計算
        open_cv_image = np.array(img) 
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        open_cv_image = cv2.dilate(open_cv_image, (5, 5), 1)
        
        for actual_pos in actual_pos_list:
            # 畫上ground truth
            draw = ImageDraw.Draw(img)  
            draw.rectangle(actual_pos, outline ="yellow", width=2)

            # 計算ground truth面積大小
            region = open_cv_image[actual_pos[1]:actual_pos[3], actual_pos[0]:actual_pos[2]]

            # 找到預測框的contours且用矩形框起來
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            contours, hierachy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

            tMP=[]

            for bbox in bounding_boxes:
                [x, y, w, h] = bbox
                # 判斷處理哪種瑕疵及處理其相對應的條件
                if target == 'Horizon':
                    if horizon(w, h):
                        draw.rectangle([x, y, x+w, y+h], outline='red', width=2)
                elif target == 'Vertical':
                    if vertical(x, y, w, h):
                        draw.rectangle([x, y, x+w, y+h], outline='red', width=2)
                elif target == 'Spot':
                    if spot(x, y, w, h):
                        draw.rectangle([x, y, x+w, y+h], outline='red', width=2)
                        tMP.append([x, y, w, h])
                        
        # if file_name == 'TD86Q1803TD_001_0.jpg':
        # img.save(join_path(save_dir,file_name))
        # print(sorted(actual_pos_list))
        bounding_boxes = sorted(bounding_boxes)
        actual_pos_list = sorted(actual_pos_list)
        tmp_box = (0, 0, 0, 0)
        while len(actual_pos_list):
            # print(actual_pos_list[0])
            near = 0
            j = 0
            flag = 0
            for box in bounding_boxes:
                Iou = cal_IOU((box[0], box[1], box[0]+box[2], box[1]+box[3]), actual_pos_list[0])
                if Iou > near:
                    near = Iou
                    tmp_box = box
                    flag = j
                j += 1
                    
            #print(flag)
            # with open('fn>5.txt', 'a') as f:
            #     f.write(f'{bounding_boxes}\n, {actual_pos}\n')

            # if file_name == 'TD85Q171QCQ_002_3.jpg':
            #     with open('fn>5.txt', 'a') as f:
            #         f.write(f'{bounding_boxes }, {actual_pos_list}\n')
            
            
            # print((tmp_box[0], tmp_box[1], tmp_box[0]+tmp_box[2], tmp_box[1]+tmp_box[3]), actual_pos_list[0])
                
            iou = cal_IOU((tmp_box[0], tmp_box[1], tmp_box[0]+tmp_box[2], tmp_box[1]+tmp_box[3]), actual_pos_list[0])
            [x1, y1, w1, h1] = tmp_box
            tmp_box = (0, 0, 0, 0)
            print('iou: ', iou)
            # if iou != 0:
            #     del bounding_boxes[i]
            if iou > 0.05:
                if target == 'Spot':
                    if spot(x1, y1, w1, h1):
                        print(x1, y1, w1, h1)
                        tp += 1
                        if bounding_boxes: del bounding_boxes[flag]
                    else: fn += 1
                        
                elif target == 'Vertical':
                    if vertical(x1, y1, w1, h1):
                        print(x1, y1, w1, h1, 'tp+1\n')
                        tp += 1
                        if bounding_boxes: del bounding_boxes[flag]     
                    else: fn += 1
                        
                elif target == 'Horizon':
                    if horizon(w1, h1):
                        print(x1, y1, w1, h1)
                        tp += 1
                        if bounding_boxes: del bounding_boxes[flag]      
                    else: fn += 1
                        
            else:
                if iou == 0:
                    fn += 1
                else:
                    if bounding_boxes:
                        fn += 1
                        del bounding_boxes[flag]

            del actual_pos_list[0]
            #print(f'TP: ', tp)
            #print(f'FP: ', fp)
            #print(f'FN: ', fn)

        while len(bounding_boxes):
            [x, y, w, h] = bounding_boxes[0]
            if target == 'Spot':
                if spot(x, y, w, h):
                    fp += 1
                # if (w / h < 2 and h / w < 2):
                #     if (w * h) > 500:
                #         fp += 1
                #     elif x < 900 and y < 1000 and x > 24 and y > 24:
                #         fp += 1
            elif target == 'Vertical':
                if vertical(x, y, w, h): 
                    fp += 1
                    print('fp:', x, y, w, h)
            elif target == 'Horizon':
                if horizon(w, h): 
                    fp += 1
                    print('fp:', x, y, w, h)
            # else:
            #     if w / h > 20 or h / w > 20:
            #         fp += 1
            del bounding_boxes[0]

        print(f'TP: ', tp)
        print(f'FP: ', fp)
        print(f'FN: ', fn)

        # if fn >= 4:
        #     with open('fn>5.txt', 'a') as f:
        #         f.write(f'{file_name },{tp },{fp },{fn }\n')
        
        if tp + fp == 0:
            print('precision: 0')
        else:
            print(f'precision: ', tp /(tp+fp))
        if tp + fn == 0:
            print('recall: 0')
        else:
            print(f'recall: ', tp /(tp+fn))
            

        # 判斷ground truth及預測的面積比例
        # is_white = cal_area(region, 50)
        # if is_white:
        #     print('Yes')
        # else:
        #     print('No')
        
        img.save(join_path(save_dir,file_name))
        
    print('---------------------')   

    
       