import torchvision
import torch.nn as nn
import pandas as pd

class VGG16_fc6_2d(nn.Module):
    def __init__(self, n_classes=1):
        super(VGG16_fc6_2d, self).__init__()
        self.layer1 = torchvision.models.vgg16_bn().features
        self.layer2=nn.Linear(512,10)
        self.layer3 = nn.Linear(10, n_classes)
    def forward(self, x):
        out = self.layer1(x[0].cuda()).mean(-1).mean(-1)
        out = self.layer2(out)
        if x[2]!=-1:
            new=out
            new=pd.DataFrame(new.detach().cpu().numpy())
            new.to_csv("F:/BMI/PictureDataset/mesh/sne/%d.csv"%x[2], index=False)
        out=self.layer3(out)
        return out
