# Decoupled Kullback-Leibler (DKL) Divergence Loss
This repository contains the implementation code for our **NeurIPS 2024** paper **Decoupled Kullback-Leibler (DKL) Divergence Loss**, [arXiv](https://arxiv.org/pdf/2305.13948v1.pdf).


## Results and Pretrained Models for Adversarial Robustness
**By 2023/05/20**, with IKL loss, we achieve new state-of-the-art adversarial robustness under settings that **with/without augmentation strategies** on [auto-attack](https://robustbench.github.io/).

### CIFAR-100 with autoaug
| # | Method | Model | Natural Acc | Robust Acc (AutoAttack) | link | log | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| 1 | DAJAT   | WRN-34-10 | 68.74 | 31.30 | - | - |
| 2 | **IKL-AT**                                       | WRN-34-10 | 65.93 | **32.52** | [model](https://drive.google.com/file/d/1kQQpVVtlS9uBC4oZ3ei9OHly1yriU1UU/view?usp=sharing) | [log](https://drive.google.com/file/d/1wvD6I2yojK5SJ4gFnkU9uwDRkU7mgZPH/view?usp=sharing) |


### CIFAR-100 with basic data preprocessing (random crop and random horizontal flip)
| # | Method | Model | Natural Acc | Robust Acc (AutoAttack) | link | log | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| 1 | AWP                                              | WRN-34-10 | 60.38 | 28.86 | - | - |
| 2 | LBGAT                                            | WRN-34-10 | 62.31 | 29.33 | - | - |
| 3 | LAS-AT                                           | WRN-34-10 | 62.99 | 30.77 | - | - |
| 4 | ACAT                                             | WRN-34-10 | 65.75 | 30.23 | - | - |
| 5 | **IKL-AT**                                       | WRN-34-10 | **66.51** | **31.43** | [model](https://drive.google.com/file/d/1NaWPX5w32xTiny91kJ6SJxSejCzsD1dy/view?usp=sharing) | [log](https://drive.google.com/file/d/1GzRey51JGmYNZTV79M_qHCL03tIf6X1P/view?usp=sharing) |
| 6 | **IKL-AT**                                       | WRN-34-10 | 65.76 | **31.91** | [model](https://drive.google.com/file/d/1lgFnfmsCw4UxguAOWSLsg6xUDfGG-HRy/view?usp=sharing) | [log](https://drive.google.com/file/d/19WeOtNsJ-ot1sG80RKEA-iMRgq4sGvNt/view?usp=sharing) |


### CIFAR-100 with synthesized data
| # | Method | Model | Natural Acc | Robust Acc (AutoAttack) | link | log | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| 1 | Wang et al. (better diffusion models) 1M         | WRN-28-10 | 68.06 | 35.65 | - | - |
| 2 | Wang et al. (better diffusion models) 50M        | WRN-28-10 | 72.58 | 38.83 | - | - |
| 3 | **IKL-AT** 1M                                    | WRN-28-10 | 68.99 | 35.89 | - | - |
| 4 | **IKL-AT** 50M                                   | WRN-28-10 | **73.85** | **39.18** | [model](https://drive.google.com/file/d/1Leec2X9kGBnBSuTiYytdb4_wR50ibTE8/view?usp=sharing) | [log](https://drive.google.com/file/d/1BcfEOhqGigxkI4GUmWvE27614mXEZVZP/view?usp=sharing) |


### CIFAR-10 with basic data preprocessing (random crop and random horizontal flip)
| # | Method | Model | Natural Acc | Robust Acc (AutoAttack) | link | log | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| 1 | AWP                                              | WRN-34-10 | 85.36 | 56.17 | - | - |
| 2 | LBGAT                                            | WRN-34-20 | 88.70 | 53.57 | - | - |
| 3 | LAS-AT                                           | WRN-34-10 | 87.74 | 55.52 | - | - |
| 4 | ACAT                                             | WRN-34-10 | 82.41 | 55.36 | - | - |
| 5 | **IKL-AT**                                       | WRN-34-10 | 85.31 | **57.13** | [model](https://drive.google.com/file/d/1SFdNdKE6ezI6OsINWX-h74dGo2-9u3Ac/view?usp=sharing) | [log](https://drive.google.com/file/d/1Uz6EjNRthCHpJvIbrGHGcMqluHDe70Ix/view?usp=sharing) |



### CIFAR-10 with synthesized data
| # | Method | Model | Natural Acc | Robust Acc (AutoAttack) | link | log | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| 1 | Wang et al. (better diffusion models) 1M         | WRN-28-10 | 91.12 | 63.35 | - | - |
| 2 | Wang et al. (better diffusion models) 20M        | WRN-28-10 | 92.44 | 67.31 | - | - |
| 3 | **IKL-AT** 1M                                    | WRN-28-10 | 90.75 | 63.54 | - | - |
| 4 | **IKL-AT** 20M                                   | WRN-28-10 | 92.16 | **67.75** | [model](https://drive.google.com/file/d/1gEodZ4ushbRPaaVfS_vjJyldH3wJg4zV/view?usp=sharing) | [log](https://drive.google.com/file/d/1tVSeSeum-q2v2CnIBIwwH2xjI5bA2WYd/view?usp=sharing) |

## Training
More training scripts will be provided soon to reproduce our results on knowledge distillation and adversarial training tasks.
```
For the adversarial training task:
cd Adv-training-dkl 
bash sh/train_dkl_cifar100.sh
bash sh/train_dkl_cifar100_autoaug.sh
bash sh/train_dkl_cifar10.sh
```

## Evaluation
before running the evaluation with auto-attack, please download the pre-trained models.
```
cd Adv-training-dkl/auto_attacks
bash sh/eval.sh
```


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
