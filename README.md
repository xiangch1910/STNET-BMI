# STNET-BMI
## Introduction
Body mass index (BMI) is an essential indicator of human health and is calculated based on height and weight. With the increasing applications in multimedia, previous studies have carried out BMI estimation from two-dimensional (2D) frontal face images using computer vision technologies. However, their research ignores the influence of different face postures on BMI estimation. Therefore, this study proposes a 3D convolutional neural network (named STNet) for BMI estimation from an image sequence with face posture information. For a single image, perspective transformation is applied to the 2D face images to assemble them into sequences. For a video, frames with certain face postures were selected for assembly into sequences. Then, the processed sequences are inputted into the STNet for BMI estimation. We validated the proposed method using a public image dataset and a private video dataset. The experimental results indicate that the proposed method can obtain an accurate and robust BMI estimation.
## Update
**[2023/05/17]** Release [STNET-BMI](https://github.com/xiangch1910/STNET-BMI), a 3D deep-learning framework for BMI estimation on image and video datasets.
## The description of the code
### the main program
return.py
### Code for network models
1).densenet2d.py   2).resnet2d.py   3).resnet3d.py   4).resnext2d.py   5).vgg16.py   6).vgg16_fc6_2d.py   7).gcblock.py
### Data Loading and Processing
1).dataloader.py   2).samplerF.py   3).Transformer.py
### Pay attention to the path when running the code
## Requirements
CUDA Version = 11.7

Python = 3.9.7 

Pytorch = 1.12.0 

Sklearn = 0.24.2

Numpy = 1.20.0

Matplotlib = 3.4.3

openface 
## Figures
### The flowchart of data processing
<div align=left><img src="/picture/data_process.jpg" width="65%" height="65%"></div>
Fig. 1: The proposed framework for BMI estimation.

### The network structure
<div align=left><img src="/picture/STNET.jpg" width="65%" height="65%"></div>
Fig. 2: Architecture of STNet.
<div align=left><img src="/picture/attention_block.jpg" width="65%" height="65%"></div>
Fig. 3: Details of the mixed-attention module.

## License

STNET-BMI are completely open for academic research. For inquiries about the commercial licensing of the OpenFace toolkit please visit https://cmu.flintbox.com/#technologies/5c5e7fee-6a24-467b-bb5f-eb2f72119e59
