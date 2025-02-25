# Generalized Kullback-Leibler (DKL) Divergence Loss
This repository contains the implementation code for our **NeurIPS 2024** paper **Decoupled Kullback-Leibler (DKL) Divergence Loss**, [arXiv](https://arxiv.org/pdf/2305.13948) and our arXiv paper **Generalized Kullback-Leibler Divergence Loss**, [arXiv]().

## Overview
In this paper, we delve deeper into the Kullback–Leibler (KL) Divergence loss and mathematically prove that it is equivalent to the Decoupled Kullback-Leibler (DKL) Divergence loss that consists of 1) a weighted Mean Square Error ($\mathbf{w}$MSE) loss and 2) a Cross-Entropy loss incorporating soft labels. 
Thanks to the decoupled structure of DKL loss, we have identified two areas for improvement.
Firstly, we address the limitation of KL loss in scenarios like knowledge distillation by breaking its asymmetric optimization property along with a smoother weight function. This modification effectively alleviates convergence challenges in optimization, particularly for classes with high predicted scores in soft labels.
Secondly, we introduce class-wise global information into KL/DKL to reduce bias arising from individual samples.
With these two enhancements, we derive the Generalized Kullback–Leibler (GKL) Divergence loss and evaluate its effectiveness by conducting experiments on CIFAR-10/100, ImageNet, and vision-language datasets, focusing on adversarial training, and knowledge distillation tasks. Specifically, we achieve new state-of-the-art adversarial robustness on the public leaderboard --- **RobustBench** and competitive knowledge distillation performance across CIFAR/ImageNet models and CLIP models, demonstrating the substantial practical merits. 

![image](https://github.com/jiequancui/DKL/blob/main/DKLv2/figures/gkl.PNG)

## Environment Settings
Python==3.8.13      
Pytorch==1.8.1+cu111    
Numpy==1.23.1     
Open-sourced code from [DKD](https://github.com/megvii-research/mdistiller) is used for fair comparions.


## Knowledge Distillation

Please refer to [KD-dkl](https://github.com/jiequancui/DKL/blob/main/DKLv2/KD-dkl) for training and evaluation.


## Imbalanced Knowledge Distillation

Please refer to [Imbalanced-KD-dkl](https://github.com/jiequancui/DKL/blob/main/DKLv2/Imbalanced-KD-dkl) for training and evaluation.


## Adversarial Robustness
**By 2023/05/20**, with GKL loss, we achieve new state-of-the-art adversarial robustness under settings that **with/without augmentation strategies** on [auto-attack](https://robustbench.github.io/).

Please refer to [Adv-training-dkl](https://github.com/jiequancui/DKL/blob/main/DKLv2/Adv-training-dkl) for training and evaluation.





# Decoupled Kullback-Leibler (DKL) Divergence Loss
This repository contains the implementation code for our **NeurIPS 2024** paper **Decoupled Kullback-Leibler (DKL) Divergence Loss**, [arXiv](https://arxiv.org/pdf/2305.13948) and our arXiv paper **Generalized Kullback-Leibler Divergence Loss**, [arXiv]().

## Overview
In this paper, we delve deeper into the Kullback–Leibler (KL) Divergence loss and mathematically prove that it is equivalent to the Decoupled Kullback-Leibler (DKL) Divergence loss that consists of 1) a weighted Mean Square Error ($\mathbf{w}$MSE) loss and 2) a Cross-Entropy loss incorporating soft labels. 
Thanks to the decomposed formulation of DKL loss, we have identified two areas for improvement. 
Firstly, we address the limitation of KL/DKL in scenarios like knowledge distillation by breaking its asymmetric optimization property. This modification ensures that the $\mathbf{w}$MSE component is always effective during training, providing extra constructive cues. Secondly, we introduce class-wise global information into KL/DKL to mitigate bias from individual samples. With these two enhancements, we derive the Improved Kullback–Leibler (IKL) Divergence loss and evaluate its effectiveness by conducting experiments on CIFAR-10/100 and ImageNet datasets, focusing on adversarial training and knowledge distillation tasks. The proposed approach achieves new state-of-the-art adversarial robustness on the public leaderboard --- **RobustBench** and competitive performance on knowledge distillation, demonstrating the substantial practical merits.

![image](https://github.com/jiequancui/DKL/blob/main/DKLv1/figures/dkl.PNG)



## Knowledge Distillation

Please refer to [KD-dkl](https://github.com/jiequancui/DKL/blob/main/DKLv1/KD-dkl) for training and evaluation.


## Imbalanced Knowledge Distillation

Please refer to [Imbalanced-KD-dkl](https://github.com/jiequancui/DKL/blob/main/DKLv1/Imbalanced-KD-dkl) for training and evaluation.


## Adversarial Robustness
**By 2023/05/20**, with IKL loss, we achieve new state-of-the-art adversarial robustness under settings that **with/without augmentation strategies** on [auto-attack](https://robustbench.github.io/).

Please refer to [Adv-training-dkl](https://github.com/jiequancui/DKL/blob/main/DKLv1/Adv-training-dkl) for training and evaluation.

## Semi-supervised Learning

Please refer to [Semi-Supervised-Learning-dkl](https://github.com/jiequancui/DKL/blob/main/DKLv1/Semi-supervised-learning-dkl) for training and evaluation.


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

@ARTICLE{10130611,
  author={Cui, Jiequan and Zhong, Zhisheng and Tian, Zhuotao and Liu, Shu and Yu, Bei and Jia, Jiaya},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Generalized Parametric Contrastive Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TPAMI.2023.3278694}}


@inproceedings{cui2021parametric,
  title={Parametric contrastive learning},
  author={Cui, Jiequan and Zhong, Zhisheng and Liu, Shu and Yu, Bei and Jia, Jiaya},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={715--724},
  year={2021}
}


```
