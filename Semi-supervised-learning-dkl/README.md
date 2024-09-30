# Decoupled Kullback-Leibler (DKL) Divergence Loss
This repository contains the implementation code for our **NeurIPS 2024** paper **Decoupled Kullback-Leibler (DKL) Divergence Loss**, [arXiv](https://arxiv.org/pdf/2305.13948v1.pdf).


## Results for Semi-supervised Learning 

### CIFAR-100 with 200 labeled-data 
| # | Method | Model | Acc | log | 
| :---: | :---: | :---: | :---: | :---: |
| 1 | FixMatch | ViT-tiny | 69.89 | [log](https://drive.google.com/file/d/1GYi8TaWfE9UDxRTi8BxPk20AQSKJqUwp/view?usp=sharing) |
| 2 | FixMatch-dkl | ViT-tiny | 70.57 | [log](https://drive.google.com/file/d/142S1LWILL1x0p-hTzlHDKzOvaAdFRfwJ/view?usp=sharing) |
| 3 | MeanTeacher | ViT-tiny | 67.49 | [log](https://drive.google.com/file/d/16SKOA7fQ2WQ6uY3rUXuytiA3mqK4xS1Y/view?usp=sharing) |
| 4 | MeanTeacher-dkl | ViT-tiny | 68.75 | [log](https://drive.google.com/file/d/1t4836seXFgm6Snfx1cRCV2QCe1c6Sq6T/view?usp=sharing) |



## Training
Please refer to https://github.com/microsoft/Semi-supervised-learning.git for environment setup.
```
For the semi-supervised learning task:
cd Semi-supervised-learning-dkl 
bash sh/train_fixmatch_ikl.sh
bash sh/train_meanteacher_ikl.sh
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
