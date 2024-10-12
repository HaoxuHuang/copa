# CoPa: General Robotic Manipulation through Spatial Constraints of Parts with Foundation Models

[[Project page]](https://copa-2024.github.io/)
[[Paper]](https://arxiv.org/abs/2403.08248)

Haoxu Huang, Fanqi Lin, Yingdong Hu, Shengjie Wang, Yang Gao

This repository is the official implementation of the paper: CoPa: General Robotic Manipulation through Spatial Constraints of Parts with Foundation Models

![](asset/banner.gif)
![](asset/method-overview.gif)

## Get Started
Install SoM following the [instruction](https://github.com/microsoft/SoM#rocket-quick-start).  
Install [graspnetAPI](https://github.com/graspnet/graspnetAPI).  
Download examples and place it under the data directory.  
Run the demo
```console
$ python demo.py
```

## Real-World Deployment
Please follow the instruction in `real_world/README.md`.

## Acknowledgement
- Our grounding module is adapted from [SoM](https://github.com/microsoft/SoM).
- We use [GraspNet](https://graspnet.net/) for grasp candidates generation.
