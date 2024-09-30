# Decoupled Kullback-Leibler (DKL) Divergence Loss
This repository contains the implementation code for our NeurIPS 2024 paper **Decoupled Kullback-Leibler (DKL) Divergence Loss**, [arXiv](https://arxiv.org/pdf/2305.13948v1.pdf).

## Overview
In this paper, we delve deeper into the KullbackLeibler (KL) Divergence loss and observe that it is equivalent to the Doupled Kullback-Leibler (DKL) Divergence loss that consists of 1) a weighted Mean Square Error (wMSE) loss and 2) a Cross-Entropy loss incorporating soft labels. From our analysis of the DKL loss, we have identified two areas for improvement. Firstly, we address the limitation of DKL in scenarios like knowledge distillation by breaking its asymmetry property in training optimization. This modification ensures that the wMSE component is always effective during training, providing extra constructive cues. Secondly, we introduce global information into DKL for intra-class consistency regularization. **With these two enhancements, we derive the Improved KullbackLeibler (IKL) Di11 vergence loss and evaluate its effectiveness by conducting experiments on CIFAR12 10/100 and ImageNet datasets, focusing on adversarial training and knowledge dis13 tillation tasks. The proposed approach achieves new state-of-the-art performance on both tasks, demonstrating the substantial practical merits**.

![image](https://github.com/jiequancui/DKL/blob/main/figures/dkl.PNG)



## Results and Pretrained Models for Knowledge Distillation
### ImageNet

 | Method | Model-Teacher | Model-Student | Training Speed | Top-1 Acc(%) | link | log | 
 | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | ReviewKD      | ResNet-34 | ResNet18 | 0.319 s/iter | 71.61 | - | - | 
 | DKD           | ResNet-34 | ResNet18 | -            | 71.70 | - | - |
 | **IKL-KD**    | ResNet-34 | ResNet18 | **0.197 s/iter** | **71.91** | - | - |https://github.com/jiequancui/DKL/blob/main/Adv-training-dkl/README.md
 
 | Method | Model-Teacher | Model-Student | Training Speed | Top-1 Acc(%) | link | log | 
 | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | ReviewKD    | ResNet-50 | MobileNet | 0.526 s/iter | 72.56 | - | - | 
 | DKD         | ResNet-50 | MobileNet | -            | 72.05 | - | - |
 | **IKL-KD**  | ResNet-50 | MobileNet | **0.252 s/iter** | **72.84** | - | - |


## Adversarial Robustness
**By 2023/05/20**, with IKL loss, we achieve new state-of-the-art adversarial robustness under settings that **with/without augmentation strategies** on [auto-attack](https://robustbench.github.io/).

Please refer to [Adv-training-dkl](https://github.com/jiequancui/DKL/blob/main/Adv-training-dkl/README.md) for training and evaluation.

## Semi-supervised Learning

Please refer to [Semi-Supervised-Learning-dkl](https://github.com/jiequancui/DKL/blob/main/Semi-supervised-learning-dkl/README.md) for training and evaluation.


# Contact
If you have any questions, feel free to contact us through email (jiequancui@gmail.com) or Github issues. Enjoy!

# BibTex
If you find this code or idea useful, please consider citing our related work:
```
@article{cui2023decoupled,
  title={Decoupled Kullback-Leibler Divergence Loss},
  author={Cui, Jiequan and Tian, Zhuotao and Zhong, Zhisheng and Qi, Xiaojuan and Yu, Bei and Zhang, Hanwang},
  journal={arXiv preprint arXiv:2305.13948},
  year={2023}
}

@inproceedings{cui2021learnable,
  title={Learnable boundary guided adversarial training},
  author={Cui, Jiequan and Liu, Shu and Wang, Liwei and Jia, Jiaya},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15721--15730},
  year={2021}
}


```
