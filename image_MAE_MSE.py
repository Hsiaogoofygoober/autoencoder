import torch
import torch.nn as nn
from model import AutoEncoderConv
from dataloader import mura_dataloader
from option import parse_option
from statistics import mean
import os
import numpy as np
import matplotlib.pyplot as plt
opts = parse_option()


device = torch.device(f"cuda:{opts.devices}" if (torch.cuda.is_available()) else "cpu")
print(f'Using {device} for inference')

# Load the saved model
model = AutoEncoderConv().to(device)
model.load_state_dict(torch.load(opts.modelpath))
model.eval()

# Create a DataLoader for the test dataset
test_dataloader = mura_dataloader(opts)
print("Number of testing data: ", len(test_dataloader))

# Initialize variables to track predictions and labels
# all_predictions = []
# all_inputs = []
all_mae = []

# Calculate MAE loss
mae_loss = nn.L1Loss().to(device)
# mae = mae_loss(all_predictions, all_inputs)

# Calculate MSE loss
# mse = mse_loss(all_predictions, all_inputs)


# Iterate over the test dataset using the DataLoader
with torch.no_grad():
    number_of_image = 1
    for i, data in enumerate(test_dataloader):
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        elif isinstance(data, (list, tuple)):
            data = [d.to(device) for d in data]

        print(number_of_image)
        # Move the inputs and labels to the device
        input = data.to(device) 
        # Forward pass to obtain predictions
        prediction = model(input)

        prediction = prediction.detach()
        input = input.detach()

        mae = mae_loss(prediction, input).item()

        # Append the predictions and labels to the lists
        # all_predictions.append(prediction)
        # all_inputs.append(input)
        all_mae.append(mae)

        del prediction, input, mae
        torch.cuda.empty_cache()
        number_of_image+=1

print('MAE Loss: ', mean(all_mae))

MAE_MSE_loss_path = os.path.join(os.path.dirname(opts.modelpath), "MAS_MSE_loss.txt")
MAE_MSE_loss_file = open(MAE_MSE_loss_path, 'w')
MAE_MSE_loss_file.write(f'MAE Loss: {mean(all_mae)}\n')
MAE_MSE_loss_file.close()