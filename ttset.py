import torchvision
import torch.nn as nn
import torch


layers = []
in_channels = 3
cfgs = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 1024, 1024, 1024, 'M']
batch_norm=False
for v in cfgs:
    if v == 'M':
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    else:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
print(layers[0:22])
