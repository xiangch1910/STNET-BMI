from time import time
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn import datasets
from sklearn.manifold import TSNE
data = []
label = []
path = "F:\\BMI\\PictureDataset\\mesh\\best"
for files in os.listdir(path):
    excel = np.array(pd.read_csv(os.path.join(path,files)))
    for i in range(len(excel)):
        data.append(excel[i,:])
    for h in range(len(excel)):
        label.append(h)
data = np.array(data)
label=np.array(label)
sam=data.shape[0]
fea=data.shape[1]
print(sam)
print(fea)

