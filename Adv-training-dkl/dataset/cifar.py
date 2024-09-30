import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.datasets as datasets

import numpy as np
from PIL import Image
import random
import math


class CIFAR10V2(torchvision.datasets.CIFAR10):
    cls_num = 10
    data_path="./data/cifar10"
    def __init__(self, root=None, train=True, transform=None, target_transform=None, download=True):
        root=self.data_path
        super(CIFAR10V2, self).__init__(root, train, transform, target_transform, download)
        
    def __getitem__(self, index):
          img, target = self.data[index], self.targets[index]
          img = Image.fromarray(img)
          if self.transform is not None:
              img_a = self.transform[0](img)
              img_n = self.transform[1](img)
          return [img_a, img_n], target 


class CIFAR100V2(torchvision.datasets.CIFAR100):
    cls_num = 100
    data_path="./data/cifar100"
    def __init__(self, root=None, train=True, transform=None, target_transform=None, download=True):
        root=self.data_path
        super(CIFAR100V2, self).__init__(root, train, transform, target_transform, download)
        
    def __getitem__(self, index):
          img, target = self.data[index], self.targets[index]
          img = Image.fromarray(img)
          if self.transform is not None:
              img_a = self.transform[0](img)
              img_n = self.transform[1](img)
          return [img_a, img_n], target 
