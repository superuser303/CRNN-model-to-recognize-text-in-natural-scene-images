import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class IIIT5KDataset(Dataset):
    def __init__(self, img_dir, anno_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        with open(anno_file, 'r') as f:
            lines = f.readlines()[1:]  # skip header
            for line in lines:
                img_name, label = line.strip().split(' ')[:2]
                self.samples.append((img_name, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label