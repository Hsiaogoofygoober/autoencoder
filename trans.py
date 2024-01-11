import cv2
import os
import numpy as np
from glob import glob

def join_path(p1,p2):
    return os.path.join(p1,p2)

target = 'Vertical'
img_folder = f'/home2/mura/daniel/data/smura_classification/{target}/result_pred-input'
save_path = f'/home2/mura/daniel/data/smura_classification/{target}/result_pred-input_binary/fin_res'
img_list = glob(f"{join_path(img_folder, '*jpg')}")

for img_path in img_list:
    name = img_path.split('/')[-1]
    print(name)

    # name = 'TD86Q1805X6_004_1.jpg'

    img = cv2.imread(f'/home2/mura/daniel/data/smura_classification/{target}/result_pred-input/{name}', 0)
    result = cv2.imread(f'/home2/mura/daniel/data/smura_classification/{target}/img/{name}')
    result = cv2.resize(result, (1024,1024), interpolation=cv2.INTER_AREA)

    blur = cv2.GaussianBlur(img, (3,3), 0) #5->3
    # blur = cv2. erode(img, np.ones((3,3)))
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2BGRA)

    cv2.imwrite('opening_blur.jpg', blur)
    
    # avg_color_per_row = np.average(gray, axis=0)
    # avg_color = np.average(avg_color_per_row, axis=0)
    # print(avg_color)

    # color_diff = np.abs(img - avg_color[0])

    # threshold_value = 100
    # binary_image = (color_diff > threshold_value).astype(np.uint8)
    # cv2.imwrite('opening_bi.jpg', binary_image)
    # # rec, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    # dilate = cv2.dilate(binary_image, np.ones((3,3)))
    # erode = cv2.dilate(dilate, np.ones((3,3)))

    canny = cv2.Canny(gray, 100, 220) #180->200->220
    canny = cv2.dilate(canny, np.ones((2,2)))
    canny = cv2.dilate(canny, np.ones((2,2)))
    cv2.imwrite('opening_ca.jpg', canny)
    # raise
    cv2.imwrite(join_path(save_path, name), canny)

    # contours, _ = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # for contour in contours:
        
    #     x, y, w, h = cv2.boundingRect(contour)
    #     print(x, y, w, h)
    #     cv2.rectangle(result, (x, y), (x + 10, y + 10), (0, 255, 0), 2)

    # cv2.imwrite('opening.jpg', result)
# erode = cv2. dilate(img, np.ones((3,3)))
# cv2.imwrite('erode.jpg', erode)

# gray = cv2.cvtColor(erode,cv2.COLOR_BGR2GRAY)
# print(gray.shape)
# rec, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
# cv2.imwrite('binary.jpg', binary)

# contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  

# bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
# i = 0
# for bbox in bounding_boxes:
#     [x, y, w, h] = bbox
    
#     if ((w/h)<2 or (h/w)<2) and ((w*h) > 5) and x > 0:
#         print(bbox)
#         cv2.rectangle(img, (x, y), (x+20, y+20), (0, 0, 255), 2, cv2.LINE_AA)
#         i = 1

# cv2.imwrite('resul.jpg', img)

# cv2.imwrite('gray.jpg', gray)
