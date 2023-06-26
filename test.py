from time import time
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn import datasets
from sklearn.manifold import TSNE
import random

def get_data():
    data = []
    label = []
    path = "F:\\BMI\\PictureDataset\\mesh\\sne"
    for files in os.listdir(path):
        excel = np.array(pd.read_csv(os.path.join(path, files)))
        for i in range(len(excel)):
            data.append(excel[i, :])
        for h in range(len(excel)):
            #c=random.randint(0, 1)
            if h!=1:
                c=0
            else:
                c=1
            label.append(c)
    data = np.array(data)
    label = np.array(label)
    n_samples=data.shape[0]
    n_features=data.shape[1]
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1((label[i]+1) / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main():
    data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, perplexity=29,learning_rate=1000,init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show()


if __name__ == '__main__':
    main()