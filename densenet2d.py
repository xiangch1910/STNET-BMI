import torchvision
import torch.nn as nn
import torch

class densenet2d(nn.Module):
    def __init__(self, n_classes=1):
        super(densenet2d, self).__init__()
        self.layer1 = torchvision.models.densenet121().features
        self.layer2=nn.Linear(1024,n_classes)

    def forward(self, x):
        out = self.layer1(x).mean(-1).mean(-1)
        out=self.layer2(out)
        return out