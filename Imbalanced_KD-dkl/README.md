# Decoupled Kullback-Leibler (DKL) Divergence Loss
This repository contains the implementation code for our NeurIPS 2024 paper **Decoupled Kullback-Leibler (DKL) Divergence Loss**, [arXiv](https://arxiv.org/pdf/2305.13948v1.pdf).


## Knowledge Distillation on ImageNet-LT

 | Method | Model-Teacher | Model-Student | Top-1 Acc(%) | link | log | 
 | :---: | :---: | :---: | :---: | :---: | :---: |
 | Baseline      | - | ResNet18 | 41.150 | - | - |
 | Baseline      | - | ResNet50 | 45.476 | - | - |
 | Baseline      | - | ResNeXt101-32x4d | 48.024 | - | - |
 | KL-KD         | ResNeXt101-32x4d | ResNet18 |  | 43.974 | - |
 | KL-KD         | ResNeXt101-32x4d | ResNet50 |  | 48.310 | - |
 | **IKL-KD**    | ResNeXt101-32x4d | ResNet18 | **44.880** | - | [log]() |
 | **IKL-KD**    | ResNeXt101-32x4d | ResNet50 | **49.224** | - | [log]() |

## Self-KD on ImageNet-LT

 | Method | Model-Teacher | Model-Student | Top-1 Acc(%) | link | log | 
 | :---: | :---: | :---: | :---: | :---: | :---: |
 | Baseline      | - | ResNet18 | 41.150 | - | - |
 | Baseline      | - | ResNet50 | 45.476 | - | - |
 | KL-KD         | ResNet18 | ResNet18 |  | - | - |
 | KL-KD         | ResNet50 | ResNet50 |  | 47.036 | - |
 | **IKL-KD**    | ResNet18 | ResNet18 | **-** | - | [log]() |
 | **IKL-KD**    | ResNet50 | ResNet50 | **47.504** | - | [log]() |


 
## Training 
Please refer to https://github.com/dvlab-research/Parametric-Contrastive-Learning for environment setup. More training scripts will be available.

```
cd DKL/Imbalanced-KD-dkl
bash sh/Distill/ImageNetLT_train_R50_baseline.sh
bash sh/Distill/ImageNetLT_train_X101_baseline.sh
bash sh/Distill/ImageNetLT_train_X101R50_distill.sh
bash sh/Distill/ImageNetLT_train_X101R50_distill_dkl.sh
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

@inproceedings{cui2021learnable,
  title={Learnable boundary guided adversarial training},
  author={Cui, Jiequan and Liu, Shu and Wang, Liwei and Jia, Jiaya},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15721--15730},
  year={2021}
}


```
