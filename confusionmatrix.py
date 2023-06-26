# coding: utf-8
import numpy as np
import os
import matplotlib.pyplot as plt

confusion_mat=np.array([[2.23,2.2,2.07,2.23],[1.85,2.88,2.26,2.22],[2.06,2.19,3.73,2.25],[2.08,2.16,2.24,5.68]])
classes_name=["VggNet","ResNet","DenseNet","ResNeXt"]
set_name="valid"
out_dir="C:\\Users\\Admin\\Desktop\\中国图象图形学报\\中文"
def show_confMat(confusion_mat, classes_name, set_name, out_dir):
    """
    可视化混淆矩阵，保存png格式
    :param confusion_mat: nd-array
    :param classes_name: list,各类别名称
    :param set_name: str, eg: 'valid', 'train'
    :param out_dir: str, png输出的文件夹
    :return:
    """
    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes_name)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html

    plt.imshow(confusion_mat_N, cmap=cmap)
    #plt.imshow(confusion_mat_N, color='white')
    #plt.colorbar()

    # 设置文字
    plt.rcParams["font.sans-serif"] = "Times New Roman"
    xlocations = np.array(range(len(classes_name)))
    plt.xticks(xlocations, classes_name, rotation=0,weight="black",fontsize=15)
    plt.yticks(xlocations, classes_name, rotation=60,weight="black",fontsize=15)
    plt.xlabel('Mesh',fontdict={"weight": "black"},fontsize=15)
    plt.ylabel('HOG',fontdict={"weight": "black"},fontsize=15)
    plt.title('MAE',fontdict={"weight": "black"},fontsize=15)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            if(i==j and i==3):
               plt.text(x=j, y=i, s=float(confusion_mat[i, j]), va='center', ha='center', color='white',fontdict={"weight": "black"}, fontsize=20)
            else:
                plt.text(x=j, y=i, s=float(confusion_mat[i, j]), va='center', ha='center', color='black',
                         fontdict={"weight": "black"}, fontsize=20)
    # 保存
    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix_' + set_name + '.tif'))
    plt.show()
    plt.close()

show_confMat(confusion_mat, classes_name, set_name, out_dir)


