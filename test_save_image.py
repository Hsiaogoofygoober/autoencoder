import torch
import torch.nn as nn
from model import AutoEncoderConv
# from CAE_model_V2 import CAE
from CAE_model import CAE
from dataloader import mura_dataloader
from option import parse_option
from statistics import mean, stdev
import os
import numpy as np
import cv2
import time
from sklearn import preprocessing
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
opts = parse_option()

# ===== function =====


def save_images(img_index, img_name, images, save_path):
    for index, image in enumerate(images):
        array = image.reshape(1024, 1024) * 255
        cv2.imwrite(os.path.join(save_path, img_name), array)

# dirt 可能會有狀況
def get_frame_region(imga, erode=5, dilate=45, debug=False): #dilate原21
    imga = np.transpose(imga, (1, 2, 0))[..., 0]
    mask = imga*255>=180
    
    mask = mask | (imga==0)
    mask = cv2.erode(np.uint8(mask), np.ones((erode, erode)))
    mask = (mask>0)|(imga*255>=250)
    mask = cv2.dilate(np.uint8(mask), np.ones((dilate, dilate)))
    n_components, components, stats, _ = cv2.connectedComponentsWithStats(mask)

    for n in range(1, n_components):
        # if stats[n][-1] < 10000:
        # print(stats[n][-1])
        # 3151
        if stats[n][-1] < 77692:
            mask[components==n] = 0
            
    return mask==0 

def remove_small_areas_opencv(image):
    image = image.astype(np.uint8)
    
    # 使用 connectedComponents 函數
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    
    # 指定面積閾值
    min_area_threshold = opts.min_area
    
    # 遍歷所有區域
    for i in range(1, num_labels):
        # 如果區域面積小於閾值，就將對應的像素值設置為黑色
        if stats[i, cv2.CC_STAT_AREA] < min_area_threshold:
            labels[labels == i] = 0
    
    # 將標籤為 0 的像素設置為白色，其它像素設置為黑色
    result = labels.astype('uint8')
    # print(np.unique(labels))
    result[result == 0] = 0
    result[result != 0] = 1
    return result

# ===================

# create save path
save_dir = os.path.join(opts.imagepath, f'{str(opts.th_percent)}_{opts.min_area}_blue')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Result save path: {save_dir}")

# set device
device = torch.device(f"cuda:{opts.devices}" if (torch.cuda.is_available()) else "cpu")
print(f'Using {device} for inference')

# Load the saved model
model = CAE().to(device)
model.load_state_dict(torch.load(opts.modelpath, map_location=device))
model.eval()

# Create a DataLoader for the test dataset
test_dataloader = mura_dataloader(opts)
print("Number of testing data: ", len(test_dataloader))

# loss function
mse_loss = nn.MSELoss(reduction='none').to(device)


all_MSE = []
excpet_list = []
# Iterate over the test dataset using the DataLoader
with torch.no_grad():
    number_of_image = 1
    for i, data in enumerate(test_dataloader):
        img = data[0]
        img_name = data[1][0]
        print(f'{time.ctime()} -> Image {number_of_image}: {img_name}')
        
        # 重建重建圖
        input = img.to(device)
        prediction = model(input)
        
        prediction = prediction.detach().cpu().numpy()
        input = input.detach().cpu().numpy()

        

        # 存原圖與重建圖
        
        input_numpy = input[0]
        prediction_numpy = prediction[0]
        input_numpy = (input_numpy.transpose((1, 2, 0)) * 255).astype(np.uint8)
        prediction_numpy = (prediction_numpy.transpose((1, 2, 0)) * 255).astype(np.uint8)
        print("input numpy shape: ",input_numpy.shape)
        #cv2.imwrite(f'/home/zach/test_C101_resultRGB/defect_{img_name}', input_numpy)
        #cv2.imwrite(f'/home/zach/test_C101_resultRGB/normal_{img_name}', prediction_numpy)
        
        


        # rm edge
        try:
            #找到要去邊界的位置
            #mask = get_frame_region(input[0])
            
            input = np.transpose(input[0], (1, 2, 0))[..., 2]
            prediction = np.transpose(prediction[0], (1, 2, 0))[..., 2]
            print("input shape: ", input.shape)
            #原圖和重建圖相減
            
            diff = (prediction - input)


            diff_min, diff_max = np.min(diff), np.max(diff)
            
            # print(diff_min, diff_max)

            #將相減後的差異平移到正數並乘上200倍
            diff = diff - diff_min
            diff = (diff * 200).astype(np.uint8)
            print(diff)
            print(diff.shape)
            cv2.imwrite(save_dir+'/'+img_name, diff) 
            
            #diff = cv2.GaussianBlur(diff, (3,3), 0)
            #diff = cv2.dilate(diff, (3,3), iterations=5)
            #diff = cv2.GaussianBlur(diff, (3,3), 0)
            #diff = cv2.erode(diff, (3,3), iterations=1)
            #diff = cv2.Canny(blur, 50, 150)
            # 視覺化 
            # avg = 127 - np.mean(diff3) 
            # diff3 = diff2 + avg
            # k = diff3.reshape(1024, 1024)
            # # cv2.imwrite(f'/home/mura/daniel/cf_mura_autoencoder/tmp_img_result/{img_name}', k)
            # cv2.imwrite(f'/home/mura/daniel/data/smura_classification/Spot/visualize/{img_name}', k)

            #image = cv2.GaussianBlur(diff2, (3,3), 0) #(3,3)
            #marked_image = image
            
            #average_color = image.mean(axis=0).mean(axis=0)
            #threshold = 1.5  # 設定一個閾值，決定高於平均值和低於平均值的範圍
            #arked_image = np.where(np.abs(image - average_color) > threshold, 255, 0).astype(np.uint8)
            
            #kernel1 = np.ones((5,5), np.uint8)
            #kernel2 = np.ones((3,3), np.uint8)
            #marked_image = cv2.dilate(marked_image, kernel1)
            #marked_image = cv2.erode(marked_image, kernel1)
            
            #marked_image = cv2.erode(marked_image, np.ones((7,1), np.uint8))
            
            # marked_image = cv2.erode(marked_image, np.ones((3,3), np.uint8))
            # 鄉剪枝後的圖 path
            
            #cv2.imwrite(save_dir+'/'+img_name, marked_image)

            # diff = mse_loss(prediction, input)[0]
            #diff = np.square(prediction - input)
            #num_pixels = diff.flatten().shape[0] # only panel
            #num_top_pixels = int(num_pixels * opts.th_percent)
            #filter = np.partition(diff.flatten(), -num_top_pixels)[-num_top_pixels]
            #print(f"Panel pixel: {num_pixels}")
            #print(f"Threshold: {filter}")
            #diff = (diff >= filter)
            
            #res = remove_small_areas_opencv(diff)
            
            #mse = diff.mean().item()
            #print("mse:", mse)
            #all_MSE.append(mse)
        except:
            excpet_list.append(img_name)
            print('raise except')
            continue
        '''
        # 取百分比 (去邊的前處理，需算mask黑色以外的面積的前幾％)
        # 1. 展平为一维数组
        flat_tensor = diff.flatten()
        flat_tensor=torch.from_numpy(flat_tensor)
        diff=torch.from_numpy(diff)
        # 2. 找到第10%值
        percentile_10_index = int(0.05 * len(flat_tensor))
        percentile_90_index = int(0.95 * len(flat_tensor))
        percentile_10_value = torch.kthvalue(flat_tensor, percentile_10_index).values
        percentile_90_value = torch.kthvalue(flat_tensor, percentile_90_index).values
        percentile_10_value = percentile_10_value.item()
        percentile_90_value = percentile_90_value.item()
        print(percentile_10_index)
        print(percentile_90_index)
        print(percentile_10_value)
        print(percentile_90_value)
        #num_top_pixels = int(num_pixels * top_k)
        #filter, _ = diff.view(-1).kthvalue(num_pixels - num_top_pixels)
        #print(f"Theshold: {filter}")
 
        # 二值化
        binary_tensor = torch.where((diff <= percentile_10_value) | (diff >= percentile_90_value), torch.tensor(255), torch.tensor(0.0))
        #diff[diff>=percentile_10_value] = 1
        #diff[diff<percentile_10_value] = 0
        cv2.imwrite(save_dir+'/'+img_name, binary_tensor.numpy())
        #save_images(i, img_name[0], binary_tensor, save_dir)
        #save_images(i, img_name[0], prediction, save_dir)
        '''
        

        del prediction, input
        # torch.cuda.empty_cache()
        number_of_image+=1


    #print(f'Mean MSE Loss: {mean(all_MSE)}')
    #print(f'Standard deviation MSE Loss: {stdev(all_MSE)}')

    #MSE_loss_path = os.path.join(save_dir, "test_result.txt")
    #MSE_loss_file = open(MSE_loss_path, 'w')

    #MSE_loss_file.write(f'Mean MSE Loss: {mean(all_MSE)}\n')
    #MSE_loss_file.write(f'Standard deviation MSE Loss: {stdev(all_MSE)}\n')
    #MSE_loss_file.write(f'Except imgs: {len(excpet_list)}\n')
    #for exc_img in excpet_list:
    #    MSE_loss_file.write(f'{exc_img}\n')
    #MSE_loss_file.close()