import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from option import parse_option
from dataloader import mura_dataloader
from model import AutoEncoderTest
import os
import time
import matplotlib.pyplot as plt
opts = parse_option()
model_save_path = os.path.join(opts.savedir, opts.testname)

# Use the parsed arguments in your program logic
print(f'Input file: {opts.input}')
print(f'Output file: {opts.output}')
print(f'Save model directory: {opts.savedir}')
# print(f'Mode: {mode}')
print(f'Number of epochs: {opts.epochs}')
print(f'Number of batchs: {opts.batchs}')
print(f'Learning rate: {opts.lr}')
print(f'Encode input size: {opts.encodesize}')
print(f'Decode input size: {opts.decodesize}')
print(f'Devices: {opts.devices}')

device = torch.device(f"cuda:{opts.devices}" if (torch.cuda.is_available()) else "cpu")
print(f'Using {device} for inference')

dataloaders = mura_dataloader(opts)
print("Number of training data: ", len(dataloaders))

model = AutoEncoderTest(opts).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
# optimizer = torch.optim.SGD(model_ae.parameters(), lr=0.1, momentum=0.8)
loss_function = nn.MSELoss()

# loss_function = nn.L1Loss().to(device)

# loss_function = nn.MSELoss().to(device)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 80, 130, 160, 190], gamma=0.2)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs / 5), eta_min=2e-10)

if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

print(f"Model save to path: {model_save_path}")

# Train
log_loss=[]
loss_file = open(os.path.join(model_save_path, 'loss_log.txt'), 'w')
for epoch in range(opts.epochs):
    start_time = time.time()
    total_loss = 0
    number_of_image = 1
    for data in dataloaders:
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        elif isinstance(data, (list, tuple)):
            data = [d.to(device) for d in data]
        print(number_of_image)
        # print(data.shape)
        inputs = data.view(-1, opts.encodesize * opts.encodesize).to(device) 
        model.zero_grad()

        # Forward
        # codes, decoded = model(inputs)
        outputs = model(inputs)
        
        # loss = loss_function(decoded, inputs)
        loss = loss_function(outputs, inputs)
        
        
        loss.backward()

        optimizer.step()
        total_loss+=loss
        log_loss.append(loss.item())
        number_of_image+=1

    total_loss /= len(dataloaders.dataset)
    scheduler.step()
    end_time = time.time()
    print('[{}/{}] Loss:{} Time:{}'.format(epoch+1, opts.epochs, total_loss.item(), end_time - start_time))
    loss_file.write(str(total_loss.item()) + '\n')
    torch.save(model, os.path.join(model_save_path, str(epoch + 1) + ".pth"))
# print('[{}/{}] Loss:'.format(epoch+1, epochs), total_loss.item())
loss_file.close()

loss_image = plt.figure()
plt.plot(log_loss)
loss_image.savefig(os.path.join(model_save_path, 'Loss_plot.png'))