import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace as pdb
from resnet import ResNet18


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        
        # w_out = 1 + (w_in + 2pad - k) / s
        self.encoder = nn.Sequential(                       # (b, 3, 32, 32)
            nn.Conv2d(3, 64, 3, stride=1, padding=1),       # (b, 64, 32, 32)
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),     # (b, 128, 16, 16)
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),    # (b, 256, 8, 8)
            nn.ReLU(True),
            nn.MaxPool2d(2)                                 # (b, 256, 4, 4)
        )

        # w_out = k + (w_in - 1) * s - 2pad
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=1),      # (b, 128, 8, 8)    ks=8(target)-(4-1)=5
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 9, stride=1),       # (b, 64, 16, 16)   ks = 16-(8-1) = 9
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 17, stride=1),        # (b, 3, 32)        ks = 32-(16-1) = 17
            nn.Tanh()
        )

    def forward(self, x):                   # (b, 3, 32, 32)
        x1 = self.encoder(x)                # (b, 256, 4, 4)
        x  = self.decoder(x1)               # (b, 3, 32, 32)
        return x1, x



class ResAE(nn.Module):
    def __init__(self):
        super(ResAE, self).__init__()
        self.encoder = ResNet18()                           # (b, 512, 1, 1)

        # w_out = k + (w_in - 1) * s - 2pad
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=1),      # (b, 256, 4, 4)    ks=4(target)-(2-1)=3
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 5, stride=1),      # (b, 128, 8, 8)    ks=8(target)-(4-1)=5
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 9, stride=1),       # (b, 64, 16, 16)   ks = 16-(8-1) = 9
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 17, stride=1),        # (b, 3, 32)        ks = 32-(16-1) = 17
            nn.Tanh()
        )

    def forward(self, x):                   # (b, 3, 32, 32)
        x1 = self.encoder(x)                # (b, 512, 1, 1)
        x  = self.decoder(x1)               # (b, 3, 32, 32)
        return x1, x