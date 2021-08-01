import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageFolder(Dataset):

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    EIG_VALS = [0.2175, 0.0188, 0.0045]
    EIG_VECS = np.array([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203]
    ])

    def __init__(self, root, transform=None) -> None:
        super().__init__()
        self.root = root
        self.img_list = glob.glob(os.path.join(root, '*.png'))
        self.img_list += glob.glob(os.path.join(root, '*.jpg'))

        self.transforms = transform
    
    def __len__(self) -> int:
        return min(len(self.img_list), 100_000)
    
    def __getitem__(self, index) -> torch.Tensor:
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
