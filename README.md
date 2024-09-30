# Decoupled Kullback-Leibler (DKL) Divergence Loss
This repository contains the implementation code for our **NeurIPS 2024** paper **Decoupled Kullback-Leibler (DKL) Divergence Loss**, [arXiv](https://arxiv.org/pdf/2305.13948v1.pdf).

## Overview
In this paper, we delve deeper into the Kullback–Leibler (KL) Divergence loss and mathematically prove that it is equivalent to the Decoupled Kullback-Leibler (DKL) Divergence loss that consists of 1) a weighted Mean Square Error ($\mathbf{w}$MSE) loss and 2) a Cross-Entropy loss incorporating soft labels. 
Thanks to the decomposed formulation of DKL loss, we have identified two areas for improvement. 
Firstly, we address the limitation of KL/DKL in scenarios like knowledge distillation by breaking its asymmetric optimization property. This modification ensures that the $\mathbf{w}$MSE component is always effective during training, providing extra constructive cues. Secondly, we introduce class-wise global information into KL/DKL to mitigate bias from individual samples. With these two enhancements, we derive the Improved Kullback–Leibler (IKL) Divergence loss and evaluate its effectiveness by conducting experiments on CIFAR-10/100 and ImageNet datasets, focusing on adversarial training and knowledge distillation tasks. The proposed approach achieves new state-of-the-art adversarial robustness on the public leaderboard --- **RobustBench** and competitive performance on knowledge distillation, demonstrating the substantial practical merits.

![image](https://github.com/jiequancui/DKL/blob/main/figures/dkl.PNG)



## Knowledge Distillation

Please refer to [KD-dkl](https://github.com/jiequancui/DKL/blob/main/KD-dkl) for training and evaluation.

## Adversarial Robustness
**By 2023/05/20**, with IKL loss, we achieve new state-of-the-art adversarial robustness under settings that **with/without augmentation strategies** on [auto-attack](https://robustbench.github.io/).

Please refer to [Adv-training-dkl](https://github.com/jiequancui/DKL/blob/main/Adv-training-dkl) for training and evaluation.

## Semi-supervised Learning

Please refer to [Semi-Supervised-Learning-dkl](https://github.com/jiequancui/DKL/blob/main/Semi-supervised-learning-dkl) for training and evaluation.


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
