# An Evaluation of Learned Features

## Introduction
An evalaution of local features learned from neural networks. This repository is an improved version of [1], which supports multi-camera datasets and includes some recent networks, e.g., matchnet, hardnet, geodesc, contextdesc, d2net, and superpoint.

The comparison of these networks for UAV images has been presented in our work:

	"Jiang S, Jiang W, Guo B, et al. Learned local features for structure from motion of UAV images: a comparative evaluation[J]. 
	IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2021, 14: 10583 - 10597."
  
[Paper](https://ieeexplore.ieee.org/document/9573512)

For the configuration of this package, please refer to the instruction in [1].

## Resources
### descriptor

tfeat (pytorch)

https://github.com/vbalnt/tfeat

L2-NET (MatConvNet)

https://github.com/yuruntian/L2-Net

Dependency: MatConvNet, cuda, cudnn

compile method: https://www.vlfeat.org/matconvnet/install/#nvcc

compile command-line:

(1) mex -setup

(2) mex -setup C++

(3) cd <MatConvNet>

(4) addpath matlab

(5) vl_compilenn('enableGpu', true, 'cudaRoot', 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1', 'cudaMethod', 'nvcc', 'enableCudnn', true, 'cudnnRoot', 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1', 'Debug', true)

(6) run vl_setupnn

(7) vl_testnn('gpu', true)

WARNING: (1) change the path of cl.exe in vl_compilenn.m according to the version of VS.
         e.g., for VS 2017, D:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64
	 (2) Matlab 2017b is the optimal version. When using other versions, link error maybe solved according to https://blog.csdn.net/u014292102/article/details/80331481

hardnet (pytorch)

https://github.com/DagnyT/hardnet

geodesc (tensorflow)

https://github.com/lzx551402/geodesc

contextdesc (tensorflow)

https://github.com/lzx551402/contextdesc

### descriptor and metric

deepcompare (torch)

https://github.com/szagoruyko/cvpr15deepcompare

deepdesc (torch)

https://github.com/etrulls/deepdesc-release

matchnet (caffe)

https://github.com/hanxf/matchnet

### detector and descriptor

LIFT (tensorflow)

https://github.com/cvlab-epfl/LIFT

super-point (tensorflow)

https://github.com/rpautrat/SuperPoint

SuperPoint only work on images with dimensions divisible by 8 and 
the user is responsible for resizing them to a valid dimension.

D2-NET (pytorch)

https://github.com/mihaidusmanu/d2-net

R2D2 (pytorch)

https://github.com/naver/r2d2

## References

[1] Schonberger J L , Hardmeier H , Sattler T , et al. Comparative Evaluation of Hand-Crafted and Learned Local Features[C]// IEEE Conference on Computer Vision & Pattern Recognition. IEEE Computer Society, 2017.

https://github.com/ahojnnes/local-feature-evaluation
