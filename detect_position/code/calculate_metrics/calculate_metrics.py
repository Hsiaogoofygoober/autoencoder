import os
from collections import defaultdict
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', type=str, default=None, required=True)
parser.add_argument('-gd', '--gt_dir', type=str, default=None, required=True)
parser.add_argument('-sd', '--save_dir', type=str, default=None, required=True)

def compute_recall_precision(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    return recall, precision, f1

def dice_coefficient(img1, img2):    
    # Ensure the images have the same shape
    assert img1.shape == img2.shape, "Error: Images have different shapes"
    # Calculate the Dice coefficient
    # Calculate the intersection
    intersection = np.sum(img1 * img2)
    total_white_pixel = np.sum(img1) + np.sum(img2)

    dice = (2 * intersection) / total_white_pixel
    return dice

def join_path(p1,p2):
    return os.path.join(p1,p2)

ignore_imgs = [ 'TD86Q1804T7_004_1.jpg',
                'TD86Q1803W4_004_1.jpg',
                'TD86Q1804HP_001_1.jpg',
                'TD86Q1803W4_001_1.jpg',
                'TD86Q1804HP_003_1.jpg',
                'TD86Q1803W4_003_1.jpg',
                'TD86Q1804T7_003_1.jpg',
                'TD86Q1804HP_004_1.jpg',
                'TD86Q1804T7_001_1.jpg'
              ]
if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.data_dir
    gt_dir = args.gt_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    row_data = defaultdict(float)
    pixels_imgs = []
    pixels_gt = []
    count = 1
    for fn in os.listdir(gt_dir):
        if fn in ignore_imgs:
            continue
        count += 1
        # Load the images
        gt_img = np.array(Image.open(join_path(gt_dir,fn)))/255
        diff_img = np.array(Image.open(join_path(data_dir,fn)))/255
        dice = dice_coefficient(gt_img, diff_img)
        row_data[fn] = dice

        # pixels_gt.append(gt_img)
        # pixels_imgs.append(diff_img)

    # pixels_gt = np.array(pixels_gt).flatten()
    # pixels_imgs = np.array(pixels_imgs).flatten()
    # recall, precision, f1 = compute_recall_precision(pixels_gt, pixels_imgs)

    df = pd.DataFrame(data=list(row_data.items()),columns=['fn','dice'])
    print(f"finished, dice mean:{df['dice'].mean()}")
    df.to_csv(join_path(save_dir, f'dice.csv'),index=False)
    with open (join_path(save_dir, f"result.txt"), 'w') as f:
        msg = f"All img: {count}\n" 
        msg += f"hit num: {df[df['dice']>0].shape[0]}\n"
        # msg  = f"hit num: {df[df['dice'] != 0].shape[0]}\n"
        msg += f"dice mean: {df['dice'].mean()}\n"
        # .
        f.write(msg)