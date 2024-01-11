# import cv2
# import numpy as np

# # Load the ground truth and prediction images
# truth_image = cv2.imread('/home/mura/AutoEncoder/NYCU_pytorch_AE/model/autoencoder_conv_1024_256_relu_sigmoid_3_layers_32_16_8_CyclicLR/image_mura/996_TD85Q182FQV_001_1_input.jpg', cv2.IMREAD_GRAYSCALE)
# prediction_image = cv2.imread('/home/mura/AutoEncoder/NYCU_pytorch_AE/model/autoencoder_conv_1024_256_relu_sigmoid_3_layers_32_16_8_CyclicLR/image_mura/996_TD85Q182FQV_001_1_prediction.jpg', cv2.IMREAD_GRAYSCALE)

# # Calculate the Mean Squared Error (MSE) for the mean
# mean_mse = np.mean((np.mean(truth_image) - np.mean(prediction_image)) ** 2)

# # Calculate the Mean Squared Error (MSE) for the standard deviation
# std_mse = np.mean((np.std(truth_image) - np.std(prediction_image)) ** 2)

# # Print the results
# print("Mean MSE:", mean_mse)
# print("Standard Deviation MSE:", std_mse)

import cv2
import torch
import torch.nn as nn

# Assuming image1_data and image2_data are the pixel values of the images
image1_data = cv2.imread('/home/mura/AutoEncoder/NYCU_pytorch_AE/model/autoencoder_conv_1024_256_relu_sigmoid_3_layers_32_16_8_CyclicLR/image_mura/516_T112D5859C_002_crop3_input.jpg', cv2.IMREAD_GRAYSCALE)
image2_data = cv2.imread('/home/mura/AutoEncoder/NYCU_pytorch_AE/model/autoencoder_conv_1024_256_relu_sigmoid_3_layers_32_16_8_CyclicLR/image_mura/516_T112D5859C_002_crop3_prediction.jpg', cv2.IMREAD_GRAYSCALE)

# # Convert the image data to PyTorch tensors
# image1 = torch.tensor(image1_data, dtype=torch.float32)
# image2 = torch.tensor(image2_data, dtype=torch.float32)

# # Reshape the images to have a batch dimension of 1
# image1 = image1.unsqueeze(0)
# image2 = image2.unsqueeze(0)

# # Create an instance of the nn.MSELoss class
# mse_loss = nn.MSELoss()

# # Calculate the MSE loss between the two images
# mse = mse_loss(image1, image2)

# # Calculate the standard deviation for every pixel
# mse_std = torch.sqrt(mse)

# # Print the results
# print("MSE:", mse.item())
# print("MSE Standard Deviation:", mse_std.item())

# Convert the image data to PyTorch tensors
image1 = torch.tensor(image1_data, dtype=torch.float32)
image2 = torch.tensor(image2_data, dtype=torch.float32)

image1 = torch.div(image1 - torch.min(image1), torch.max(image1) - torch.min(image1))
image2 = torch.div(image2 - torch.min(image2), torch.max(image2) - torch.min(image2))

# Calculate the squared differences
squared_diff = torch.pow(image1 - image2, 2)

# Calculate the MSE for every pixel
mse_per_pixel = torch.mean(squared_diff, dim=0)

# Calculate the mean and standard deviation of the MSE
mse_mean = torch.mean(mse_per_pixel)
mse_std = torch.std(mse_per_pixel)

# Print the results
print("Mean MSE per pixel:", mse_mean.item())
print("MSE Standard Deviation per pixel:", mse_std.item())