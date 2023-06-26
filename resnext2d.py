import torchvision
import torch.nn as nn
import torch

class resnext2d(nn.Module):
    def __init__(self, n_classes=1):
        super(resnext2d, self).__init__()
        a=torchvision.models.resnext101_32x8d()._modules
        self.layer1=nn.Sequential(a["conv1"],a["bn1"],a['relu'],a['maxpool'],a["layer1"],a["layer2"])
        self.layer2=nn.Linear(512,n_classes)

    def forward(self, x):
        out = self.layer1(x).mean(-1).mean(-1)
        out=self.layer2(out)
        return out