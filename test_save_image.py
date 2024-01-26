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
save_dir = os.path.join(opts.imagepath, f'{str(opts.th_percent)}_{opts.min_area}_color')
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
        img_name = img_name.replace("JPG", "PNG")
        print(f'{time.ctime()} -> Image {number_of_image}: {img_name}')
        
        # 重建重建圖
        input = img.to(device)
        prediction = model(input)
        
        input_numpy = input[0]
        prediction_numpy = prediction[0]

        prediction_numpy = prediction_numpy.detach().cpu().numpy()
        input_numpy = input_numpy.detach().cpu().numpy()

        

        # 存原圖與重建圖
        

        input_numpy = np.transpose(input_numpy, (1, 2, 0))
        input_numpy = (input_numpy*255).astype(np.uint8)
        prediction_numpy = np.transpose(prediction_numpy, (1, 2, 0))
        prediction_numpy = (prediction_numpy*255).astype(np.uint8)
        cv2.imwrite(f'/home/zach/test_C101_resultRGB256to16V3/defect_{img_name}', input_numpy)
        cv2.imwrite(f'/home/zach/test_C101_resultRGB256to16V3/normal_{img_name}', prediction_numpy)
        
        # rm edge
        try:
            reconstruct_image = prediction_numpy
            origin_image = input_numpy
            x, y, width, height = 200, 200, 624, 624  # Example values
            black_image = np.zeros_like(reconstruct_image)
            origin_image = origin_image[y:y+height, x:x+width]
            reconstruct_image = np.round(reconstruct_image / 16) * 16

            #store 16 pixel reconstruct image
            cv2.imwrite(f'/home/zach/test_C101_resultRGB256to16V3/normal_16_{img_name}', reconstruct_image)
            reconstruct_image = reconstruct_image[y:y+height, x:x+width]
            reconstruct_image = np.int16(reconstruct_image)
            origin_image = np.int16(origin_image)
            output = reconstruct_image - origin_image

            output_minR, output_maxR = np.min(output[...,0]), np.max(output[...,0])
            output_minG, output_maxG = np.min(output[...,0]), np.max(output[...,0])
            output_minB, output_maxB = np.min(output[...,0]), np.max(output[...,0])

            flat_tensorR = output[...,0].flatten()
            flat_tensorG = output[...,1].flatten()
            flat_tensorB = output[...,2].flatten()
            flat_tensorR=torch.from_numpy(flat_tensorR)
            flat_tensorR = torch.unique(flat_tensorR)
            flat_tensorG=torch.from_numpy(flat_tensorG)
            flat_tensorG = torch.unique(flat_tensorG)
            flat_tensorB=torch.from_numpy(flat_tensorB)
            flat_tensorB = torch.unique(flat_tensorB)

            output=torch.from_numpy(output)
            bottom = 0.2
            top = 0.8
            # 2. 保留top 與 buttom的值
            percentileR_10_index = int(bottom * len(flat_tensorR))
            percentileG_10_index = int(bottom * len(flat_tensorG))
            percentileB_10_index = int(bottom * len(flat_tensorB))
            percentileR_upper_index = int(top * len(flat_tensorR))
            percentileG_upper_index = int(top * len(flat_tensorG))
            percentileB_upper_index = int(top * len(flat_tensorB))

            percentileR_10_value = torch.kthvalue(flat_tensorR, percentileR_10_index).values
            percentileG_10_value = torch.kthvalue(flat_tensorG, percentileG_10_index).values
            percentileB_10_value = torch.kthvalue(flat_tensorB, percentileB_10_index).values
            percentileR_90_value = torch.kthvalue(flat_tensorR, percentileR_upper_index).values
            percentileG_90_value = torch.kthvalue(flat_tensorG, percentileG_upper_index).values
            percentileB_90_value = torch.kthvalue(flat_tensorB, percentileB_upper_index).values


            percentileR_10_value = percentileR_10_value.item()
            percentileG_10_value = percentileG_10_value.item()
            percentileB_10_value = percentileB_10_value.item()
            percentileR_90_value = percentileR_90_value.item()
            percentileG_90_value = percentileG_90_value.item()
            percentileB_90_value = percentileB_90_value.item()

            # 二值化
            binary_tensorR = torch.where((output[...,0] >= percentileR_90_value)|(output[...,0] <= percentileR_10_value), torch.tensor(255), torch.tensor(0))
            binary_tensorG = torch.where((output[...,1] >= percentileG_90_value)|(output[...,1] <= percentileG_10_value), torch.tensor(255), torch.tensor(0))
            binary_tensorB = torch.where((output[...,2] >= percentileB_90_value)|(output[...,1] <= percentileB_10_value), torch.tensor(255), torch.tensor(0))

            rgb_image = torch.stack([binary_tensorR, binary_tensorG, binary_tensorB], dim=0)

            rgb_image = np.clip(rgb_image, 0, 255)

            # Transpose the tensor to have the channels as the last dimension (H x W x C)
            rgb_image = rgb_image.permute(1, 2, 0)

            rgb_image = rgb_image.numpy().astype(np.uint8)
            black_image[y:y+height, x:x+width] = rgb_image

            cv2.imwrite(f'{save_dir}/{img_name}', black_image)
        except:
            excpet_list.append(img_name)
            print('raise except')
            continue
        
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