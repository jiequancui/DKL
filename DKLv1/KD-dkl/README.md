# Decoupled Kullback-Leibler (DKL) Divergence Loss
This repository contains the implementation code for our NeurIPS 2024 paper **Decoupled Kullback-Leibler (DKL) Divergence Loss**, [arXiv](https://arxiv.org/pdf/2305.13948v1.pdf).


## Results for Knowledge Distillation
### ImageNet

 | Method | Model-Teacher | Model-Student | Training Speed | Top-1 Acc(%) | link | log | 
 | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | ReviewKD      | ResNet-34 | ResNet18 | 0.319 s/iter | 71.61 | - | - | 
 | DKD           | ResNet-34 | ResNet18 | -            | 71.70 | - | - |
 | **IKL-KD**    | ResNet-34 | ResNet18 | **0.197 s/iter** | **71.91** | - | [log](https://drive.google.com/file/d/1uFTrIfPI-7BcTfutzulTgYD6xAL5kHH2/view?usp=sharing) |
 
 | Method | Model-Teacher | Model-Student | Training Speed | Top-1 Acc(%) | link | log | 
 | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
 | ReviewKD    | ResNet-50 | MobileNet | 0.526 s/iter | 72.56 | - | - | 
 | DKD         | ResNet-50 | MobileNet | -            | 72.05 | - | - |
 | **IKL-KD**  | ResNet-50 | MobileNet | **0.252 s/iter** | **72.84** | - | [log](https://drive.google.com/file/d/1aA5YDqnriNc3w_W-bY1VSCFZ6-OpAbKk/view?usp=sharing) |

## Training 
Please refer to https://github.com/megvii-research/mdistiller for environment setup. More training scripts will be available.

```
cd DKL/KD-dkl
bash sh/imagenet_r34_r18_ikl.sh
bash sh/imagenet_r50_mv_ikl.sh
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
