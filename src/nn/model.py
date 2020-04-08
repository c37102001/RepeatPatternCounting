import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace as pdb


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=11):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):                           # (b, 3, 32, 32)
        out = F.relu(self.bn1(self.conv1(x)))       # (b, 64, 32, 32)
        out = self.layer1(out)                      # (b, 64, 16, 16)
        out = self.layer2(out)                      # (b, 128, 8, 8)
        out = self.layer3(out)                      # (b, 256, 4, 4)
        out = self.layer4(out)                      # (b, 512, 2, 2)
        out = F.avg_pool2d(out, 2)                  # (b, 512, 1, 1)
        
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


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