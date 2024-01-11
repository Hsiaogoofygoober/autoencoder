import torch
import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape=args

    def forward(self, x):
        return x.view(self.shape)
    

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        # Encoder
        # 1024*1024
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                      #512*512
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                      #256*256
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                      #128*128
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),                      #64*64
            # nn.Flatten(),                                            
            # nn.Linear(131072, 1024)
        )


        # Decoder
        self.decoder = nn.Sequential(
            # torch.nn.Linear(1024, 131072),
            # Reshape(-1, 128, 32, 32),
            # nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(4, 64, kernel_size=3),
            # nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x