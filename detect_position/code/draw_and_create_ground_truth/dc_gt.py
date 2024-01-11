import os
import argparse
from collections import defaultdict 
from glob import glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw
import xmltodict
import sys
import csv
parser = argparse.ArgumentParser()
parser.add_argument('-dv', '--dataset_version', type=str, default=None, required=True)
parser.add_argument('-dd', '--data_dir', type=str, default=None, required=True)
parser.add_argument('-sd', '--save_dir', type=str, default=None, required=True)
parser.add_argument('-rs', '--resized', type=int, default=None, required=True)
# parser.add_argument('-ls', '--large', type=int, default=None, required=True)


def add_row(info_fn, fn, obj):
    info_fn['fn'] = fn
    info_fn['smura_name'] = obj['name']
    x0 = int(obj['bndbox']['xmin'])
    info_fn['x0'] = x0
    y0 = int(obj['bndbox']['ymin'])
    info_fn['y0'] = y0
    x1 = int(obj['bndbox']['xmax'])
    info_fn['x1'] = x1
    y1 = int(obj['bndbox']['ymax'])
    info_fn['y1'] = y1
    info_fn['x_center'] = (x0+x1)//2
    info_fn['y_center'] = (y0+y1)//2
    # info_fn['w'] = x1-x0+1
    # info_fn['h'] = y1-y0+1
    info_fn['w'] = x1-x0
    info_fn['h'] = y1-y0
    return info_fn

def join_path(p1,p2):
    return os.path.join(p1,p2)
    
args = parser.parse_args()
dataset_version = args.dataset_version
data_dir = args.data_dir
save_dir = args.save_dir
isResize = args.resized
# isLarge = args.larged

os.makedirs(join_path(save_dir, dataset_version), exist_ok=True)
# 將所有標註 xml 統整成標註 df
# xml_dir = join_path(data_dir, f'{dataset_version}/xml')
xml_dir = '/home2/mura/daniel/0921_cf_mura/smura/xml'
# print(xml_dir)
xml_list = glob(f"{join_path(xml_dir, '*xml')}")
# print(xml_list)
# count=0
# for i in xml_list:
#     print(i)
#     count+=1
# print(count)


info_fn_list = []
for xml_path in xml_list:
    with open(xml_path) as fd:   
        # print(xml_path)     
        json_fd = xmltodict.parse(fd.read())
        # print(json_fd)
        if isinstance(json_fd['annotation']['object'], list):
            for obj in json_fd['annotation']['object']:
                info_fn = defaultdict()
                # print(info_f)
                # print(json_fd['annotation']['filename'])
                # print(obj)
                info_fn_list.append(add_row(info_fn, json_fd['annotation']['filename'], obj))
        else:
            info_fn = defaultdict()
            info_fn_list.append(add_row(info_fn, json_fd['annotation']['filename'], json_fd['annotation']['object']))

df = pd.DataFrame.from_dict(info_fn_list)
df.to_csv(join_path(save_dir, f'{dataset_version}/{dataset_version}.csv'), index=False, header=True)
raise
# 基於標註 df 將實際 mura 位置標註在圖上
if isResize:
    resolution = "resized"
elif isResize:
    resolution = "large"
else:
    resolution = "origin"

save_dir = join_path(save_dir, f'{dataset_version}/{resolution}/reconstructed_actual_pos')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(join_path(save_dir, f'bounding_box'), exist_ok=True)
os.makedirs(join_path(save_dir, f'ground_truth'), exist_ok=True)

img_dir = data_dir
img_list = glob(f"{join_path(img_dir, '*jpg')}")
# print(img_list)
for img_path in img_list:
    fn = img_path.split('/')[-1]
    print(fn)
    if join_path(xml_dir, f"{fn[:-4]}.xml") not in xml_list:
        continue
    img = Image.open(img_path)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if isResize:
        img = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(img, (1920,1080), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    fn_series_list = df[df['fn']==fn]
    
    actual_pos_list = []
    for i in range(0, fn_series_list.shape[0]):
        fn_series = fn_series_list.iloc[i]
        if isResize:
            actual_pos_list.append((int(fn_series['x0']/2), int(fn_series['y0']/1.586914), int(fn_series['x1']/ 2), int(fn_series['y1']/1.586914)))
        # elif isLarge:
        #     actual_pos_list.append((int(fn_series['x0']/0.252), int(fn_series['y0']/0.25), int(fn_series['x1']/0.14), int(fn_series['y1']/0.25)))
        else:
            actual_pos_list.append((int(fn_series['x0']), int(fn_series['y0']), int(fn_series['x1']), int(fn_series['y1'])))
    
    if isResize:
        binary_img = np.zeros((1024,1024)) # for binary mask
    # if isLarge:
    #     binary_img = np.zeros((4320,7680)) # for binary mask
    else:
        binary_img = np.zeros((1080,1920)) # for binary mask

    for actual_pos in actual_pos_list:
        draw = ImageDraw.Draw(img)  
        draw.rectangle(actual_pos, outline ="yellow")
        # print(actual_pos)
        binary_img[actual_pos[1]:actual_pos[3], actual_pos[0]:actual_pos[2]] = 255

    # print(join_path(save_dir, fn))
    img.save(join_path(save_dir, f'bounding_box/{fn}'))
    binary_img = Image.fromarray(binary_img).convert('L')
    binary_img.save((join_path(save_dir, f'ground_truth/{fn}')))
