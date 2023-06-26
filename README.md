# STNET-BMI
Body mass index (BMI) is an essential indicator of human health and is calculated based on height and weight. With the increasing applications in multimedia, previous studies have carried out BMI estimation from two-dimensional (2D) frontal face images using computer vision technologies. However, their research ignores the influence of different face postures on BMI estimation. Therefore, this study proposes a 3D convolutional neural network (named STNet) for BMI estimation from an image sequence with face posture information. For a single image, perspective transformation is applied to the 2D face images to assemble them into sequences. For a video, frames with certain face postures were selected for assembly into sequences. Then, the processed sequences are inputted into the STNet for BMI estimation. We validated the proposed method using a public image dataset and a private video dataset. The experimental results indicate that the proposed method can obtain an accurate and robust BMI estimation.
### the main program
return.py
### Code for network models
1.densenet2d.py   2.resnet2d.py   3.resnet3d.py   4.resnext2d.py   5.vgg16.py   6.vgg16_fc6_2d.py   7.gcblock.py
### Data Loading and Processing
1.dataloader.py   2.samplerF.py   3.Transformer.py
## Pay attention to the path when running the code

