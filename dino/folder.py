import os
import glob
import math

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from . import utils

class ImageFolder(Dataset):

    MEAN = [0.485,
    0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    EIG_VALS = [0.2175, 0.0188, 0.0045]
    EIG_VECS = np.array([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203]
    ])

    def __init__(self, root, transform=None, max_indice=None, slice=None) -> None:
        super().__init__()
        self.root = root
        self.img_list = glob.glob(os.path.join(root, '*.png'))
        self.img_list += glob.glob(os.path.join(root, '*.jpg'))
        self.max_num = np.inf if max_indice is None else max_indice
        
        if slice is not None:
            n = len(self.img_list)
            self.img_list[math.floor(slice[0] * n): math.floor(slice[1] * n)]

        self.transforms = transform
    
    def __len__(self) -> int:
        return min(len(self.img_list), self.max_num)
    
    def __getitem__(self, index) -> torch.Tensor:
        if self.max_num is not None:
            n = len(self.img_list)
            mul = math.ceil(n / self.max_num)
            index = (index * mul) % n
        pil_img = Image.open(self.img_list[index]).convert("RGB")
        if self.transforms:
            img = self.transforms(pil_img)
        else:
            img = pil_img
        # return {
        #     'input': img,
        #     'instance_target': index
        # }
        return img, index


class KpImageFolder(ImageFolder):

    def __getitem__(self, index) -> torch.Tensor:
        if self.max_num is not None:
            n = len(self.img_list)
            mul = math.ceil(n / self.max_num)
            index = (index * mul) % n
        pil_img = Image.open(self.img_list[index]).convert("RGB")
        
        if self.transforms:
            img, _ = self.transforms(pil_img, index)
        else:
            raise RuntimeError('Need transforms to genreating keypoints')
        return img, index


class LitImgFolder(pl.LightningDataModule):

    def __init__(self, root_dir, batch_size=32, num_worker=16, split=0.01, step_per_epoch=100_000):
        self.root = root_dir
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.split = split
        self.steps = step_per_epoch
    
    def train_dataloader(self) -> DataLoader:
        slice_range = (0, 1 - self.split)
        train_dataset = ImageFolder(
            self.root,
            transform=[],
            slice=slice_range,
            max_indice=self.steps)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            worker_init_fn=utils.worker_init_fn)
        return train_loader
    
    def val_dataloader(self) -> DataLoader:
        slice_range = (1 - self.split, 1)
        val_dataset = ImageFolder(
            self.root,
            transform=[],
            slice=slice_range)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            worker_init_fn=utils.worker_init_fn)
        return val_loader