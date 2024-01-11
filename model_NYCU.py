import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, opts):
        super(Encoder, self).__init__()
        encodeSize = opts.encodesize * opts.encodesize
        decodeSize = opts.decodesize

        # Encoder
        self.encoder = nn.Sequential(
            # nn.Linear(3*128*128, 100*100),
            # nn.Tanh(),
            nn.Linear(encodeSize, decodeSize)
            # nn.Tanh(),
            # nn.Linear(64*64, 32*32),
            # nn.Tanh(),
            # nn.Linear(32*32, 16*16),
            # nn.Tanh()          
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        codes = self.encoder(inputs)
        return codes
    
class Decoder(nn.Module):
    def __init__(self, opts):
        super(Decoder, self).__init__()
        encodeSize = opts.encodesize * opts.encodesize
        decodeSize = opts.decodesize

        # Decoder
        self.decoder = nn.Sequential(
            # nn.Linear(16*16, 32*32),
            # nn.Tanh(),
            # nn.Linear(32*32, 64*64),
            # nn.Tanh(),
            nn.Linear(decodeSize, encodeSize)
            # nn.Tanh(),
            # nn.Linear(100*100, 3*128*128),
            # nn.Sigmoid()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        outputs = self.decoder(inputs)
        return outputs
    
class AutoEncoder(nn.Module):
    def __init__(self, opts):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = Encoder(opts)
        # Decoder
        self.decoder = Decoder(opts)
        
    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)
        return codes, decoded