# Decoupled Kullback-Leibler (DKL) Divergence Loss
This repository contains the implementation code for our arXiv paper **Decoupled Kullback-Leibler (DKL) Divergence Loss**.

## Overview
In this paper, we delve deeper into the KullbackLeibler (KL) Divergence loss and observe that it is equivalent to the Doupled Kullback-Leibler (DKL) Divergence loss that consists of 1) a weighted Mean Square Error (wMSE) loss and 2) a Cross4 Entropy loss incorporating soft labels. From our analysis of the DKL loss, we have identified two areas for improvement. Firstly, we address the limitation of DKL in scenarios like knowledge distillation by breaking its asymmetry property in training optimization. This modification ensures that the wMSE component is always effective during training, providing extra constructive cues. Secondly, we introduce global information into DKL for intra-class consistency regularization. With these two enhancements, we derive the Improved KullbackLeibler (IKL) Di11 vergence loss and evaluate its effectiveness by conducting experiments on CIFAR12 10/100 and ImageNet datasets, focusing on adversarial training and knowledge dis13 tillation tasks. The proposed approach achieves new state-of-the-art performance on both tasks, demonstrating the substantial practical merits.

![image](https://github.com/FPNAS/ResLT/blob/main/assets/reslt.jpg)
