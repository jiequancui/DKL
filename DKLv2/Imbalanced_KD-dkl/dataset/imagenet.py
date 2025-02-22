import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImageNetLT(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                img_path = line.split()[0]
                #print(img_path, "before ")
                strs = img_path.split('/')
                img_path = img_path[:-5]+'_'+strs[1]+".JPEG"
                #print(img_path, "after ")
                self.img_path.append(os.path.join(root, img_path))
                self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label  # , index
