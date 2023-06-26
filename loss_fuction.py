import torch.nn as nn
import torch
import numpy as np

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        #self.weight = []  #设置4个类别的权重
        
        self.weight = torch.from_numpy(np.array([1,1,1])).float()
        self.loss_f = nn.SmoothL1Loss()
    def forward(self,preds,targets):
        classPreds = preds[:,:3]
        valuePreds = preds[:,-1]
        classTargets=  targets[:,0]
        valueTargets = targets[:,1]
        L1Loss = self.loss_f(valuePreds.double(),valueTargets.double())
        valueLoss = (valuePreds - valueTargets).pow(2).mean()/2
        MAE=(valuePreds - valueTargets).abs().mean()
        # loss = classLoss.mean() + valueLoss.mean()
        return [L1Loss,valueLoss,MAE]


