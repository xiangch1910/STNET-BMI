import numpy as np
import torch
import glob
import os
import time
import cv2
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from dataloader import OverloadDataset,ToolDataset
import torchvision.transforms as transforms
from Transformer import RandomAffineF,ToTensorF,ResizeF,NormalizeF,RandomRotationF
from torch.utils.data import Dataset, DataLoader
from samplerF import OverLoadSampler
from PIL import Image as IM
from vnet import VNet
from loss_fuction import CrossEntropy
import torchvision
#import moxing as mox
from ResNet3D import generate_model
from scipy.stats import pearsonr
from sklearn import metrics
from VGG16 import VGG16_fc6_2d
from densenet2d import densenet2d
from resnet2d import resnet2d
from  resnext2d import resnext2d

#mox.file.shift("os", "mox")

#from torch.utils.tensorboard import SummaryWriter



def value2class(value,max,min):
    value = value.unsqueeze(1).detach().cpu().numpy()
    valueF = []
    for i in range(value.shape[0]):
        T=value[i][0]*(max-min)+min
        if T <= 18.5:
            pred = 0
        if T > 18.5 and T < 24.9:
            pred = 1
        if T >= 24.9 and T <30:
            pred = 2
        if T>=30:
            pred = 3
        valueF.append(pred)

    pred=torch.from_numpy(np.array(valueF)).cuda()
    return pred

def denormalize(real,max,min):
    c=real
    valuef=[]
    for i in range(len(c)):
        c[i]=c[i]*(max-min)+min
        valuef.append(c[i])
    GR=np.array(valuef)
    #GR = torch.from_numpy(np.array(valuef)).cuda()
    return GR

def classify(pred,target,classes,resList,final = False):
    if final:
        predOneHot = np.array(resList[0])
        targetOneHot = np.array(resList[1])
        accuracyFList = []
        real=0
        for i in range(classes):
            preds = predOneHot[:,i]
            targets = targetOneHot[:,i]
            real=real+np.sum(np.array(preds)*np.array(targets))
            standA = preds - targets
            standB = preds + targets

            standA2 = np.where(standA > 0, 1, 0)
            FP = standA2.sum()
            standA3 = np.where(standA < 0, 1, 0)
            FN = np.abs(standA3).sum()

            standB2 = np.where(standB > 1, 1, 0)
            TP = standB2.sum()
            standB3 = np.where(standB < 1, 1, 0)
            TN = standB3.sum()

            #accuracy = (TP)/(TP + FN)
            recall=(TP)/(TP + FN)
            precison=(TP)/(TP+FP)
            if np.isnan(precison):
                precison=0                
            accuracy = 2 * recall * precison / (recall + precison)
            if np.isnan(accuracy):
                accuracy=0
            accuracyFList.append(accuracy)
        absaccuracy=real/len(predOneHot)
        accuracyF = np.array(accuracyFList)
        return [resList,accuracyF,absaccuracy]
    else:
        #转化为ONE-HOT编码
        predLA = torch.from_numpy(pred)
        targetLA = torch.from_numpy(target)
        predOneHot = torch.zeros(predLA.shape[0],classes).scatter(1, predLA.unsqueeze(-1).long(),1)     
        targetOneHot = torch.zeros(targetLA.shape[0],classes).scatter(1, targetLA.unsqueeze(-1).long(),1)
        resList[0] += predOneHot.numpy().tolist()
        resList[1] += targetOneHot.numpy().tolist()
        return resList

def TRA(WorkShop,NET, dataLoader, optimizer, epochsInd,classes,lossFunction,max,min):
    # print("开始训练")
    NET.train()
    start_time = time.time()

    resList2=[[],[]]
    lossList = []
    namelist=[]
    tramax=max
    tramin=min
    for i, dataList in enumerate(dataLoader):
        #x_data = dataList[0].cuda()
        x_data = dataList[0]
        y_data = dataList[1].cuda()
        name_Data=dataList[2]
        namelist.append(name_Data)
        x_data.append(-1)
        pred = NET(x_data)
        L1Loss,valueLoss,MAE = lossFunction(pred,y_data)
        loss = L1Loss
        lossList.append(loss.detach().cpu().numpy())
        valuepred=pred[:,-1]
        classcal =value2class(valuepred,tramax,tramin)
        resList2=classify(classcal.detach().cpu().numpy(),y_data[:,0].detach().cpu().numpy(),classes,resList2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("\repochs:%d  steps:%d   time:%f s   loss:%f " %
              (epochsInd + 1, i, (time.time() - start_time), loss.mean()), end="")
    resList2,accuracyF2,absaccuracyF2=classify(0,0,classes,resList2,True)
    lossF = np.array(lossList).mean()
    print("    train_cal_accuracy:%f  train_cal_F1:%f train_loss:%f" % (np.mean(absaccuracyF2),np.mean(accuracyF2),lossF))
    return namelist

def VAL(WorkShop,NET, dataLoader,saveList,classes,CrossValInd,lossFunction, bestLossF, bestAccuracyF, isStopping,max,min,namelist):
    with torch.no_grad():
        NET.eval()
        start_time = time.time()

        resList2=[[],[]]
        lossValueF = []
        n = 0
        lossList = []
        saveModel = saveList[0]
        saveRESULT = saveList[1]
        lossData = []
        rmse_list = []
        valmax=max
        valmin=min
        real_GR=[]
        real_pred=[]
        for i, dataList in enumerate(dataLoader):
            #x_data = dataList[0].cuda()
            x_data = dataList[0]
            y_data = dataList[1].cuda()
            x_data.append(i)
            pred = NET(x_data)
            
            L1Loss, valueLoss,MAE = lossFunction(pred, y_data)

            classcal=value2class(pred[:,-1],valmax,valmin)

            lossData.append([L1Loss.detach().cpu().numpy(), valueLoss.detach().cpu().numpy(),MAE.detach().cpu().numpy()])

          ##反归一化
            nor_real = y_data[:, 1].detach().cpu().numpy()
            nor_pred = pred[:, -1].detach().cpu().numpy()
            real_GR+=denormalize(nor_real,valmax,valmin).tolist()
            real_pred+=denormalize(nor_pred,valmax,valmin).tolist()
            resList2=classify(classcal.detach().cpu().numpy(),y_data[:,0].detach().cpu().numpy(),classes,resList2)
            print("\r          val_steps:%d   time:%f s  " %
                  (i, (time.time() - start_time)), end="")



    mae=round(metrics.mean_absolute_error(np.array(real_pred),np.array(real_GR)),3)
    resList2,accuracyF2,absaccuracyF2=classify(0,0,classes,resList2,True)
    meanLossList = np.array(lossData).mean(0).tolist()
    print("  ")
    print("        accuracycal ",end="    ")
    for i in range(classes):
        accuracy2 = accuracyF2[i]
        print("%d:%f" % (i, accuracy2), end="    ")
    meanAccuuracyList2 = [np.array(accuracyF2).mean()]
    for qq in range(len(meanAccuuracyList2)):
        print("mean accuracycal:%f" % meanAccuuracyList2[qq])

    print("absaccuracycal:%f" % absaccuracyF2,end="   ")
    #print("pearsonr:",pearsonr)

   
    print("        loss",end="    ")
    for pp in range(len(meanLossList)):
        if pp==2:
            print("%d:%f"%(pp,meanLossList[pp]*(valmax-valmin)),end="     ")                      
        else:
            print("%d:%f"%(pp,meanLossList[pp]),end="     ") 

    print("mae:",mae)
    print(" ")




    

    print("---------------------------------------------------------------------------------------------------------")


    lossValueM = meanLossList[-1]
    if len(bestLossF) != 0:
        maxnum=0

        for i in range(len(bestAccuracyF)):
            if maxnum<=bestAccuracyF[i][1][0]:
                maxnum=bestAccuracyF[i][1][0]
        if absaccuracyF2 > maxnum:
#             torch.save(NET,
#                        os.path.join(saveModel, "a_resnet152_%d" % CrossValInd + '.pkl'))

            answer=[real_pred,real_GR]
            answerdata=pd.DataFrame(answer)
            answerdata.to_csv(os.path.join(saveRESULT,"a_%dvalue.csv"%CrossValInd))

            orgindata = pd.DataFrame(namelist)
            orgindata.to_csv(os.path.join(saveRESULT, "a_%ddata.csv" % CrossValInd))

            resDataList = meanAccuuracyList2+[absaccuracyF2]+meanLossList+[mae]
            resData = pd.DataFrame(resDataList)
            resData.to_csv(os.path.join(saveRESULT,"a_%d.csv"%CrossValInd))       
        
    if len(bestLossF) != 0 and lossValueM < bestLossF[0][-1]:
        bestLossF = [meanLossList]
        bestAccuracyF = [[meanAccuuracyList2,[absaccuracyF2]]]
#         torch.save(NET,
#                    os.path.join(saveModel, "resnet152_%d" % CrossValInd + '.pkl'))
        
        torch.save(NET.state_dict(),os.path.join(saveModel,"LCR_pram_%d" %CrossValInd+".pkl"))
        
        answer=[real_pred,real_GR]
        answerdata=pd.DataFrame(answer)
        answerdata.to_csv(os.path.join(saveRESULT,"%dvalue.csv"%CrossValInd))
        
        orgindata = pd.DataFrame(namelist)
        orgindata.to_csv(os.path.join(saveRESULT, "%ddata.csv" % CrossValInd))
        
        resDataList = meanAccuuracyList2+[absaccuracyF2]+meanLossList+[mae]
        resData = pd.DataFrame(resDataList)
        resData.to_csv(os.path.join(saveRESULT,"%d.csv"%CrossValInd))

        
    else:
        bestLossF.append(meanLossList)
        bestAccuracyF.append([meanAccuuracyList2,[absaccuracyF2]])
        


    if len(bestLossF) >= 10 or isStopping:
        isStopping = True
    return bestLossF, bestAccuracyF, isStopping

def CrossVal(WorkShop,DataSet,classes,imgSize,base_lr,epochs,batchSize,steps_per_epoch,encoding,name,max,min):
    print(name)
    inforStore = os.path.join(WorkShop,DataSet)
    saveDir = os.path.join(WorkShop, DataSet,"valResult", name)
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)


    saveModel = os.path.join(saveDir, "MODEL")
    if not os.path.exists(saveModel):
        os.mkdir(saveModel)

    saveResult = os.path.join(saveDir, "RESULT")
    if not os.path.exists(saveResult):
        os.mkdir(saveResult)
    saveList = [saveModel,saveResult]

    startTime = time.time()
    bestAccuracyList = []
    bestLossList = []
    crossmax=max
    crossmin=min
    for CrossValInd in range(5):
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("CrossValidationInd : %d"%CrossValInd)
        print("========================")

        ##区分数据集
        traDataset = OverloadDataset(WorkShop = WorkShop,DataSet=DataSet,max=crossmax,min=crossmin,inforStore=inforStore, type = "tra",CrossValInd=CrossValInd,encoding=encoding,
                                 transform=[transforms.Compose([transforms.Resize((imgSize,imgSize),interpolation=IM.BILINEAR),
                                                             transforms.RandomAffine(degrees=15, translate=None, scale=None, shear=None, fillcolor=0),
                                                               transforms.ToTensor()]),
                                            transforms.Compose([transforms.Resize((imgSize,imgSize),interpolation=IM.BILINEAR),
                                                               transforms.ToTensor()])
                                           ])
        
        
        ##加载数据集，拼接batch作为模型输入
        traDataloader = DataLoader(traDataset, batch_size=batchSize,sampler=OverLoadSampler(steps_per_epoch*batchSize), num_workers=0)

        valDataset = OverloadDataset(WorkShop = WorkShop,DataSet=DataSet,max=crossmax,min=crossmin,inforStore=inforStore, type="val", CrossValInd=CrossValInd,encoding=encoding,
                                     transform=[transforms.Compose(
                                         [transforms.Resize((imgSize, imgSize), interpolation=IM.BILINEAR),
                                          transforms.RandomAffine(degrees=15, translate=None, scale=None, shear=None,
                                                                  fillcolor=0),
                                          transforms.ToTensor()]),
                                                transforms.Compose(
                                                    [transforms.Resize((imgSize, imgSize), interpolation=IM.BILINEAR),
                                                     transforms.ToTensor()]),
                                                transforms.Compose(
                                                    [transforms.Resize((imgSize, imgSize), interpolation=IM.BILINEAR),
                                                     transforms.ToTensor()]),
                                           ])
        
        valDataloader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=0)

        
        
        #NET = VNet().cuda()
        NET=resnet2d().cuda()
        #net=torchvision.models.resnet18()
       


        #NET = generate_model(50).cuda()
#       net_state_dict = NET.state_dict()
#         ##加入预训练模型参数
#         pretrained_dict = torch.load(os.path.join('F:/BMI/PictureDataset/mesh/Liuchaoran/valResult/result/MODEL',"LCR_pram_0.pkl"))
#         pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}  #去掉多余的键
#         net_state_dict.update(pretrained_dict_1)#更新新模型的参数字典
#         NET.load_state_dict(net_state_dict)#将更新后的参数放回到网络中
        optimizer = torch.optim.Adam(NET.parameters(), lr=base_lr)
        #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, NET.parameters()), lr=base_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8, last_epoch=-1)

        bestLossF = []
        bestAccuracyF = []
        isStopping = False
        lossFunction = CrossEntropy()
        for i in range(epochs):
            if i == epochs - 1:
                isStopping = True
            scheduler.step(epoch=i)
                
            namelist=TRA(WorkShop=WorkShop,NET=NET, dataLoader=traDataloader, optimizer=optimizer, epochsInd=i,classes=classes,lossFunction=lossFunction,max=crossmax,min=crossmin)
            bestLossF,bestAccuracyF,isStopping = VAL(WorkShop=WorkShop,NET=NET, dataLoader=valDataloader, saveList=saveList, classes=classes,
                                                     CrossValInd= CrossValInd,lossFunction=lossFunction, bestLossF=bestLossF, bestAccuracyF=bestAccuracyF,isStopping=isStopping,max=crossmax,min=crossmin,namelist=namelist)
            if isStopping :
                bestAccuracyList.append(bestAccuracyF[0])
                bestLossList.append(bestLossF[0])
                break


        NET.cpu()
        torch.cuda.empty_cache()

    meanAccuracyArray = np.array(bestAccuracyList)
    meanLossArray = np.array(bestLossList)
    meanAccuracy = meanAccuracyArray.mean(0).tolist()
    
    meanAccuracy=[meanAccuracy[0][0],meanAccuracy[1][0]]
    meanLoss = meanLossArray.mean(0).tolist()
    
    resDataList = meanAccuracy + meanLoss
    resData = pd.DataFrame(resDataList)
    resData.to_csv(os.path.join(saveResult, "F.csv"))

    print(
        "***************************************************************************************************************************")
    print(
        "***************************************************************************************************************************")
    print("Final result:")
    print("mean accuracy",end="   ")
    for kk in range(len(meanAccuracy)):
        print("%d:%f"%(kk,meanAccuracy[kk]),end="   ")

    print("mean loss",end=" ")
    for pp in range(len(meanLoss)):
        if pp==0:
            print("%d:%f" % (pp, meanLoss[pp]),end="   ")
        else:
            
            print("%d:%f" % (pp, meanLoss[pp]*(crossmax-crossmin)),end="   ")

    endTime = time.time()
    elaspedTime = (endTime - startTime)/3600

    print("time:%f hours"%elaspedTime)
    print(name)
    return

WorkShop="F:\\BMI\\PictureDataset\\mesh"
DataSet="VIP_m"
# with mox.file.File(os.path.join(WorkShop,DataSet,"data.csv"), "r") as f:
#     normal = pd.read_csv(f)
# max=normal.loc[:, "BMI"].max()
# min=normal.loc[:,"BMI"].min()
max=57
min=15
CrossVal(WorkShop = WorkShop,DataSet = DataSet,
    classes = 4, imgSize = 224,
    base_lr = 0.0001, epochs = 20,batchSize = 12,steps_per_epoch = 100,encoding = "RGB", name = "result",max=max,min=min)

