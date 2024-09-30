# Decoupled Kullback-Leibler (DKL) Divergence Loss
This repository contains the implementation code for our NeurIPS 2024 paper **Decoupled Kullback-Leibler (DKL) Divergence Loss**, [arXiv](https://arxiv.org/pdf/2305.13948v1.pdf).


## Knowledge Distillation
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
