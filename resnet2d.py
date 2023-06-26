# import torchvision
# import torch.nn as nn
# import torch
#
# class resenet2d(nn.Module):
#     def __init__(self, n_classes=1):
#         super(resenet2d, self).__init__()
#         a=torchvision.models.resnet152()._modules
#         self.layer1=nn.Sequential(a["conv1"],a["bn1"],a['relu'],a['maxpool'],a["layer1"],a["layer2"])
#         self.layer2=nn.Linear(512,n_classes)
#
#     def forward(self, x):
#         out = self.layer1(x).mean(-1).mean(-1)
#         out=self.layer2(out)
#         return out

import torchvision
import torch.nn as nn
import torch
#from gcblock import ContextBlock

class resnet2d(nn.Module):
    def __init__(self, n_classes=1):
        cfgs = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 1024, 1024, 1024, 'M']
        super(resnet2d, self).__init__()
        batch_norm=True
        layers = []
        in_channels = 3
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
        self.vgg_st1=nn.Sequential(*layers[0:22])
        self.vgg_st2 = nn.Sequential(*layers[23:32])
        self.vgg_st3 = nn.Sequential(*layers[33:42])
        # self.attention1 =ContextBlock(64)
        # self.attention2 =ContextBlock(256)
        # self.attention3 =ContextBlock(512)
        # self.attention4 =ContextBlock(1024)
        a=torchvision.models.resnet50(pretrained=True)._modules
#         self.layer1=nn.Sequential(a["conv1"],a["bn1"],a['relu'],a['maxpool'],a["layer1"],a["layer2"])
#         self.layer2=nn.Linear(512,n_classes)
        self.layer0=nn.Sequential(a["conv1"],a["bn1"],a['relu'],a['maxpool'])
        self.layer1=nn.Sequential(a["layer1"])
        self.layer2=nn.Sequential(a["layer2"])
        self.layer3=nn.Sequential(a["layer3"])
        self.layer4=nn.Sequential(a["layer4"])

        self.avgpool = nn.Sequential(a["avgpool"])
        self.fc=nn.Linear(1024,n_classes)
        #self.fc = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, n_classes))
    def forward(self, x):
        out=self.layer0(x[1])
        out_mesh_1=self.vgg_st1(x[0])
        out_mesh_2 = self.vgg_st2(out_mesh_1)
        out_mesh_3 = self.vgg_st3(out_mesh_2)
        out=self.layer1(out)+out_mesh_1
        out=self.layer2(out)+out_mesh_2
        out=self.layer3(out)+out_mesh_3
      #  out = self.layer4(out)
        out=self.avgpool(out)
        out=torch.flatten(out,1)
        out=self.fc(out)
        return out





