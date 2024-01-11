import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from numpy import asarray
from sklearn.metrics import mean_squared_error

path1 = '/home2/mura/daniel/cf_mura_autoencoder/tmp_img_result/con_TD86Q1805G6_003_2.jpg'
path2 = '/home2/mura/daniel/cf_mura_autoencoder/tmp_img_result/ori_TD86Q1805G6_003_2.jpg'
fake = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
real = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

# fake_numpy = asarray(fake)
# real_numpy = asarray(real)

   
# diff = np.square(fake - real)
diff = mean_squared_error(fake, real)

print(diff.mean().item()) 