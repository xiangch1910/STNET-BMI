from __future__ import print_function, division
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import math
from PIL import Image as IM
from matplotlib import pyplot as plt
import random
# import moxing as mox
# mox.file.shift("os", "mox")

class OverloadDataset(Dataset):
    def __init__(self, WorkShop, DataSet,max,min,inforStore, type, CrossValInd, encoding="RGB", transform=None):
        self.encoding = encoding
        inforF = []
        nameList = ["data0.csv","data1.csv","data2.csv","data3.csv"]
        for i in range(4):
            infor = pd.read_csv(os.path.join(inforStore,nameList[i])).values.tolist()
        # for i in range(3):
        #     with mox.file.File(os.path.join(inforStore,nameList[i]), "r") as f:
        #         infor = pd.read_csv(f).values.tolist()
#             with mox.file.File(os.path.join(os.path.join("obs://jdrh-xch/data/PictureDatasets/liuchaoran", nameList[i+4])), "r") as f: 
#                 infor2 = pd.read_csv(f).values.tolist()
                
            anchor = int(len(infor) / 5)
            if type == "tra":
                inforS = np.array(infor[: CrossValInd * anchor] + infor[(CrossValInd + 1) * anchor:])
                inforF.append(inforS)
            elif type == "val":
                inforS = np.array(infor[CrossValInd * anchor:(CrossValInd + 1) * anchor]).tolist()
                inforF += inforS
        if type == "val":
            inforF = np.array(inforF)

        self.infor = inforF
        self.type = type
        self.transform = transform
        self.WorkShop = WorkShop
        self.DataSet=DataSet
        self.max=max
        self.min=min


    def __len__(self):
        return len(self.infor)

    def __getitem__(self, idxS):
        if self.type == "tra":
            cla = int(idxS % 4)
            infor = self.infor[cla]
            #idx = np.random.randint(0, len(infor), 1)
            idx=[int(idxS/4)%(len(infor))]
            videoPath = os.path.join(self.WorkShop,infor[idx, 0][0]).replace("\\", "/")
            name = infor[idx, 0][0]
            pppath = os.path.join(self.WorkShop, infor[idx, 1][0]).replace("\\", "/")
        else:
            cla = False
            idx = idxS
            infor = self.infor
            videoPath = os.path.join(self.WorkShop, infor[idx,0]).replace("\\", "/")
            name = infor[idx,0]
            pppath = os.path.join(self.WorkShop, infor[idx, 1]).replace("\\", "/")
        a=self.max
        b=self.min
        # oldpic = int(infor[idx, 1])
        # videoEndPoint = int(infor[idx, 2])
        # Sex = infor[idx, 4]
        # Height=infor[idx,5]
        # Weight=infor[idx,6]
        BMI=infor[idx,7]
        classes = infor[idx, 8]
        ClaIntensity = np.array([int(classes[0]),(float(BMI)-b)/(a-b)])

        # if not cla:
        #     x_data =self.transform[2](np.load(videoPath)[videoStartPoint:videoEndPoint]).permute(1,0,2,3)
        # else:
        #     if cla==0 or cla==2:
        #         x_data =self.transform[0](np.load(videoPath)[videoStartPoint:videoEndPoint]).permute(1,0,2,3)
        #     else:
        #         x_data =self.transform[1](np.load(videoPath)[videoStartPoint:videoEndPoint]).permute(1,0,2,3)
        # y_data = torch.from_numpy(ClaIntensity)
        #x_data = cv2.imdecode(np.fromstring(mox.file.read(videoPath, binary=True), np.uint8), cv2.IMREAD_COLOR)
        #         x_data=cv2.resize(x_data,(224,224))
        #         x_data=torch.from_numpy(x_data)/255.0
        #         x_data=x_data.type(torch.FloatTensor)
        x_data=cv2.imread(videoPath)
        x_data = IM.fromarray(x_data)
        x_data1=cv2.imread(pppath)
        x_data1=IM.fromarray(x_data1)
        if cla == 0:
            x_data = self.transform[0](x_data)
            x_data1 = self.transform[0](x_data1)
        else:
            # x_data = cv2.resize(x_data, (224, 224))
            # x_data = torch.from_numpy(x_data) / 255.0
            # x_data = x_data.type(torch.FloatTensor)
            x_data=self.transform[1](x_data)
            x_data1 = self.transform[1](x_data1)
       # x_data = x_data.permute(2, 0, 1).cuda()
        x_data=x_data.cuda()
        x_data1 = x_data1.cuda()
        y_data = torch.from_numpy(ClaIntensity)

        return [[x_data,x_data1], y_data, name]



class ToolDataset(Dataset):
    def __init__(self,steps):
        self.steps = steps
    def __len__(self):
        return len(self.steps)