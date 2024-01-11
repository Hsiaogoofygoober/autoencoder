import torch
import torch.nn as nn

class AutoEncoderConv(nn.Module):
    def __init__(self):
        super(AutoEncoderConv, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            # nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=2),
            # nn.ReLU(True),
            # nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(True),
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=3 , stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x