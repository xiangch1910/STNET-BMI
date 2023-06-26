import torchvision
import torch.nn as nn
import torch
import pandas as pd
import os
class VGG16_fc6_2d(nn.Module):
    def __init__(self, n_classes=1):
        super(VGG16_fc6_2d, self).__init__()
        self.vgglayer = torchvision.models.vgg19_bn(pretrained=True).features
        a = torchvision.models.resnet18(pretrained=True)._modules
        self.resnetlayer=nn.Sequential(a["conv1"],a["bn1"],a['relu'],a['maxpool'],a["layer1"],a["layer2"],a["layer3"],a["layer4"])
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        #self.fc=nn.Sequential(nn.Linear(50176,4096),nn.BatchNorm2d(4096),nn.ReLU(),nn.Linear(4096,n_classes))
        self.fc = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, n_classes))
    def forward(self, x):
        out_dep=self.vgglayer(x[0].cuda())
        out_hog=0.1*self.resnetlayer(x[1].cuda())
        out=torch.cat((out_dep,out_hog),dim=1)
        out=self.avgpool(out)
        out = torch.flatten(out,1)
        # if x[2]!=-1:
        #     new=out
        #     new=pd.DataFrame(new.detach().cpu().numpy())
        #     new.to_csv("F:/BMI/PictureDataset/mesh/sne/%d.csv"%x[2], index=False)
        out=self.fc(out)
        return out
