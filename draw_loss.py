import matplotlib.pyplot as plt
import os

loss_path = "/home/mura/AutoEncoder/NYCU_pytorch_AE/model/autoencoder_conv_1024_256_relu_sigmoid_3_layers/loss_log.txt"  # Replace with the actual file path
with open(loss_path, 'r') as loss_file:
    lines = loss_file.readlines()
    loss_list = [float(line.strip()) for line in lines]  # Remove leading/trailing whitespaces and newline characters

epochs = range(1, len(loss_list) + 1)  # Generating epochs based on the length of the loss list

# plt.figure(figsize=(30, 10))
plt.plot(loss_list)  # Plotting the loss values with epochs
plt.xlabel('Epoch')
plt.ylabel('L1 Loss')
plt.title('L1 Loss with Epochs')
# plt.grid(True)
plt.savefig(os.path.join(os.path.dirname(loss_path), 'Loss_plot.png'))
