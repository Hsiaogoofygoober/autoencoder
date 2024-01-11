#Import the necessary libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image, ImageEnhance
image = cv2.imread('/home2/mura/daniel/cf_mura_autoencoder/tmp_img_result/TD86Q1803TD_003_0.jpg')

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
# marked_image = cv2.Canny(gray, 80, 180) #180->200->220

# method 2
image = cv2.GaussianBlur(image, (3,3), 0)
ignored_value = 127
masked_image = np.ma.masked_equal(image, ignored_value)
average_color = np.mean(masked_image, axis=(0, 1))
threshold = 15
high_values = np.logical_and(image > (average_color + threshold), image != ignored_value)
low_values = np.logical_and(image < (average_color - threshold), image != ignored_value)
marked_image = np.zeros_like(image)
marked_image[high_values] = 255
marked_image[low_values] = 255

# method 1
# image = cv2.GaussianBlur(image, (3,3), 0) #threshold=12
# average_color = image.mean(axis=0).mean(axis=0)
# threshold = 12  # 設定一個閾值，決定高於平均值和低於平均值的範圍
# marked_image = np.where(np.abs(image - average_color) > threshold, 255, 0).astype(np.uint8)
# marked_image = cv2.erode(marked_image, (3,3))
# marked_image = cv2.dilate(marked_image, (3,3))

cv2.imwrite('Masked Image2.jpg', marked_image)









