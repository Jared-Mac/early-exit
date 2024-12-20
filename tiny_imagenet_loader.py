# -*- coding: utf-8 -*-
"""TinyImageNetLoader.ipynb

Automatically generated by Colaboratory.


"""

#loads images as 3*64*64 tensors 

# !wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
# !unzip -q tiny-imagenet-200.zip

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os, glob
from torchvision.io import read_image, ImageReadMode
import pytorch_lightning as pl
batch_size = 64

id_dict = {}
for i, line in enumerate(open('data/tiny-imagenet-200/wnids.txt', 'r')):
  id_dict[line.replace('\n', '')] = i

class TrainTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("data/tiny-imagenet-200/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
            image = read_image(img_path, ImageReadMode.RGB)
        class_id = img_path.split('/')[-3]
        label = self.id_dict[class_id]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

class TestTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("data/tiny-imagenet-200/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        val_annotations_path = 'data/tiny-imagenet-200/val/val_annotations.txt'
        with open(val_annotations_path, 'r') as f:
            for line in f:
                img_name, class_id = line.strip().split('\t')[:2]
                self.cls_dic[img_name] = self.id_dict[class_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
            image = read_image(img_path, ImageReadMode.RGB)
        img_filename = os.path.basename(img_path)
        label = self.cls_dic[img_filename]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

class TinyImageNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Normalize(
            (122.4786, 114.2755, 101.3963), 
            (70.4924, 68.5679, 71.8127)
        )
        self.train_dataset = None
        self.test_dataset = None
        self.setup()

    def setup(self, stage=None):
        if self.train_dataset is None:
            self.train_dataset = TrainTinyImageNetDataset(id=id_dict, transform=self.transform)
        if self.test_dataset is None:
            self.test_dataset = TestTinyImageNetDataset(id=id_dict, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

    def val_dataloader(self):
        return self.test_dataloader()