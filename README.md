# Decoupled Kullback-Leibler (DKL) Divergence Loss
This repository contains the implementation code for our arXiv paper **Decoupled Kullback-Leibler (DKL) Divergence Loss**.

## Overview
In this paper, we delve deeper into the KullbackLeibler (KL) Divergence loss and observe that it is equivalent to the Doupled Kullback-Leibler (DKL) Divergence loss that consists of 1) a weighted Mean Square Error (wMSE) loss and 2) a Cross4 Entropy loss incorporating soft labels. From our analysis of the DKL loss, we have identified two areas for improvement. Firstly, we address the limitation of DKL in scenarios like knowledge distillation by breaking its asymmetry property in training optimization. This modification ensures that the wMSE component is always effective during training, providing extra constructive cues. Secondly, we introduce global information into DKL for intra-class consistency regularization. **With these two enhancements, we derive the Improved KullbackLeibler (IKL) Di11 vergence loss and evaluate its effectiveness by conducting experiments on CIFAR12 10/100 and ImageNet datasets, focusing on adversarial training and knowledge dis13 tillation tasks. The proposed approach achieves new state-of-the-art performance on both tasks, demonstrating the substantial practical merits**.

![image](https://github.com/jiequancui/DKL/blob/main/figures/dkl.PNG)



## Results and Pretrained Models for Knowledge Distillation
### ImageNet

 | Method | Model-Teacher | Model-Student | Training Speed | Top-1 Acc(%) | link | log | 
 | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | ReviewKD      | ResNet-34 | ResNet18 | 0.319 s/iter | 71.61 | - | - | 
 | DKD           | ResNet-34 | ResNet18 | -            | 71.70 | - | - |
 | **IKL-KD**    | ResNet-34 | ResNet18 | **0.197 s/iter** | **71.91** | - | - |
 
 | Method | Model-Teacher | Model-Student | Training Speed | Top-1 Acc(%) | link | log | 
 | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | ReviewKD    | ResNet-50 | MobileNet | 0.526 s/iter | 72.56 | - | - | 
 | DKD         | ResNet-50 | MobileNet | -            | 72.05 | - | - |
 | **IKL-KD**  | ResNet-50 | MobileNet | **0.252 s/iter** | **72.64** | - | - |


## Results and Pretrained Models for Adversarial Robustness
**By 2023/05/20**, with IKL loss, we achieve new state-of-the-art adversarial robustness under settings that **with/without augmentation strategies** on [auto-attack](https://robustbench.github.io/).


### CIFAR-100 with basic data preprocessing (random crop and random horizontal flip)
| # | Method | Model | Natural Acc | Robust Acc (AutoAttack) | link | log | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| 1 | AWP                                              | WRN-34-10 | 60.38 | 28.86 | - | - |
| 2 | LBGAT                                            | WRN-34-10 | 62.31 | 30.44 | - | - |
| 3 | LAS-AT                                           | WRN-34-10 | 62.99 | 31.20 | - | - |
| 4 | ACAT                                             | WRN-34-10 | 65.75 | 30.23 | - | - |
| 5 | **IKL-AT**                                       | WRN-34-10 | **66.51** | **31.43** | - | - |
| 6 | **IKL-AT**                                       | WRN-34-10 | 64.08 | **31.65** | - | - |


### CIFAR-100 with synthesized data
| # | Method | Model | Natural Acc | Robust Acc (AutoAttack) | link | log | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| 1 | Wang et al. (better diffusion models) 1M         | WRN-28-10 | 68.06 | 35.65 | - | - |
| 2 | Wang et al. (better diffusion models) 50M        | WRN-28-10 | 72.58 | 38.83 | - | - |
| 3 | **IKL-AT** 1M                                    | WRN-28-10 | 68.99 | 35.89 | - | - |
| 4 | **IKL-AT** 50M                                   | WRN-28-10 | **73.85** | **39.18** | - | - |


### CIFAR-10 with basic data preprocessing (random crop and random horizontal flip)
| # | Method | Model | Natural Acc | Robust Acc (AutoAttack) | link | log | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| 1 | AWP                                              | WRN-34-10 | 85.36 | 56.17 | - | - |
| 2 | LBGAT                                            | WRN-34-20 | 88.70 | 53.57 | - | - |
| 3 | LAS-AT                                           | WRN-34-10 | 87.74 | 55.52 | - | - |
| 4 | ACAT                                             | WRN-34-10 | 82.41 | 55.36 | - | - |
| 5 | **IKL-AT**                                       | WRN-34-10 | 85.31 | **57.13** | - | - |



### CIFAR-10 with synthesized data
| # | Method | Model | Natural Acc | Robust Acc (AutoAttack) | link | log | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| 1 | Wang et al. (better diffusion models) 1M         | WRN-28-10 | 91.12 | 63.35 | - | - |
| 2 | Wang et al. (better diffusion models) 20M        | WRN-28-10 | 92.44 | 67.31 | - | - |
| 3 | **IKL-AT** 1M                                    | WRN-28-10 | 90.75 | 63.54 | - | - |
| 4 | **IKL-AT** 20M                                   | WRN-28-10 | 92.16 | **67.75** | - | - |

## Training Scripts
Full code will be released soon
