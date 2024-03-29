# Local feature evaluation

## Introduction

An evaluation of local features learned from neural networks. This repository is an improved version of [the work](https://github.com/ahojnnes/local-feature-evaluation), which supports multi-camera datasets and includes some recent networks, e.g., matchnet, hardnet, geodesc, contextdesc, d2net, and superpoint. The comparison of these networks for UAV images has been presented in the [paper](https://ieeexplore.ieee.org/document/9573512):

![framework](https://github.com/json87/local-feature-evaluation/blob/main/doc/framework.jpg)

## Build

For the configuration of this package, please refer to the [repo](https://github.com/ahojnnes/local-feature-evaluation).

## Resources
### === descriptor

#### tfeat (pytorch)

https://github.com/vbalnt/tfeat

#### L2-NET (MatConvNet)

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

#### hardnet (pytorch)

https://github.com/DagnyT/hardnet

#### geodesc (tensorflow)

https://github.com/lzx551402/geodesc

#### contextdesc (tensorflow)

https://github.com/lzx551402/contextdesc

### === descriptor and metric

#### deepcompare (torch)

https://github.com/szagoruyko/cvpr15deepcompare

#### deepdesc (torch)

https://github.com/etrulls/deepdesc-release

#### matchnet (caffe)

https://github.com/hanxf/matchnet

### === detector and descriptor

#### LIFT (tensorflow)

https://github.com/cvlab-epfl/LIFT

#### super-point (tensorflow)

https://github.com/rpautrat/SuperPoint

SuperPoint only work on images with dimensions divisible by 8 and 
the user is responsible for resizing them to a valid dimension.

#### D2-NET (pytorch)

https://github.com/mihaidusmanu/d2-net

#### R2D2 (pytorch)

https://github.com/naver/r2d2

## References

```
@article{
  author={Jiang, San and Jiang, Wanshou and Guo, Bingxuan and Li, Lelin and Wang, Lizhe},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Learned Local Features for Structure From Motion of UAV Images: A Comparative Evaluation}, 
  year={2021},
  volume={14},
  pages={10583-10597},
}

@INPROCEEDINGS{
  author={Schönberger, Johannes L. and Hardmeier, Hans and Sattler, Torsten and Pollefeys, Marc},
  booktitle={2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Comparative Evaluation of Hand-Crafted and Learned Local Features}, 
  year={2017},
  pages={6959-6968},
}

```
