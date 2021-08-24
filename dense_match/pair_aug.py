import itertools
import functools
from re import L
from typing import *
from dataclasses import dataclass
from collections import defaultdict

import cv2
from imgaug.augmenters.meta import OneOf
from imgaug.augmenters.size import CropToFixedSize
import torch
import numpy as np
import imgaug as ia
from PIL import Image
from imgaug import augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from torchvision.transforms import functional as tvtf


@dataclass
class MetaKP:
    aug_kp: Keypoint  # 0 ~ img width/height
    aug_grid_kp: Tuple[float]  # 0 ~ grid(feature map) width/height
    src_grid_kp: Tuple[float]  # 0 ~ grid(feature map) width/height
    src_img_id: int  # will come to handy when we blend two sample together
    src_img_size: Union[Tuple[int], List[int]]


class PairAug:

    def __init__(self, n_deriv=3, output_size=[320, 320], sample_rate=16, norm=True):
        self.n_derive = n_deriv
        self.output_size = output_size
        self.grid_size = output_size[0] // sample_rate
        self.grid_cell = output_size[0] // self.grid_size
        self.norm = norm
    
    @staticmethod
    def collect(batch: List[Dict[str, list]]):
        batch_cache = defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batch_cache[k].append(v)                    
        
        batched = {}
        for k, v in batch_cache.items():
            if type(v[0]) is torch.Tensor:
                if v[0].ndim == 3:
                    batched[k] = torch.stack(v)
                elif 6 > v[0].ndim > 3:
                    batched[k] = torch.cat(v)
                elif v[0].ndim == 1:
                    if 'img_idx' in k:
                        batched[k] = torch.cat(v)
                    else:
                        raise ValueError(f'{v[0].ndim}')
                else:
                    raise ValueError(f'{v[0].ndim}')
            elif type(v[0]) is list:
                batched[k] = functools.reduce(lambda a, b: a + b, v)
            else:
                raise RuntimeError()
        return batched
    
    @property
    def grid_kp(self):
        if not hasattr(self, '_grid_kp'):
            assert self.output_size[0] == self.output_size[1], 'Currently only support square image'
            imsize = self.output_size[0]
            sr = self.output_size[0] // self.grid_size

            self._grid_kp = KeypointsOnImage([
                Keypoint(
                    x=i * sr + sr // 2,
                    y=j * sr + sr // 2)
                for i, j in itertools.product(range(self.grid_size), range(self.grid_size))
                ],
                shape=(imsize, imsize, 3)
            )
        return self._grid_kp.deepcopy()

    @property
    def uni_size_transf(self) -> iaa.Augmenter:
        if not hasattr(self, '_uni_size_transf'):
            self._uni_size_transf = iaa.Sequential([
                iaa.CropToAspectRatio(1),
                iaa.OneOf([
                    iaa.Sequential([
                        iaa.CropToFixedSize(
                            width=self.output_size[0],
                            height=self.output_size[1]),
                        iaa.Resize(self.output_size),
                    ]),
                    iaa.Resize(self.output_size),
                    iaa.Sequential([
                        iaa.Resize((0.5, 1.0)),
                        iaa.Resize({
                            'width': self.output_size[0],
                            'height': self.output_size[1]
                        })
                    ])
                ])
            ])
        return self._uni_size_transf
    
    @property
    def gemo_transf(self) -> iaa.Augmenter:
        if not hasattr(self, '_gemo_trans'):
            self._gemo_trans = iaa.Sequential([
                iaa.Fliplr(p=0.5),
                iaa.Flipud(p=0.5),
                iaa.Sometimes(0.5, iaa.Rotate(rotate=(-90, 90))),
                iaa.Sometimes(0.5, iaa.OneOf([
                    iaa.PiecewiseAffine(scale=(0.01, 0.1)),
                    iaa.ShearX((-20, 20)),
                    iaa.ShearY((-20, 20)),
                    iaa.ScaleX((0.5, 1.5)),
                    iaa.ScaleY((0.5, 1.5)),
                ])),
                iaa.Sometimes(0.1, iaa.Jigsaw(nb_rows=8, nb_cols=8, max_steps=(3, 3))),
            ])
        return self._gemo_trans
    
    @property
    def color_transf(self) -> iaa.Augmenter:
        if not hasattr(self, '_color_transf'):
            self._color_transf = iaa.Sequential([
                iaa.OneOf([
                    iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
                    iaa.MultiplyHueAndSaturation(mul_hue=(0.5, 1.5)),
                    iaa.ChangeColorTemperature((1100, 10000)),
                ]),
                iaa.OneOf([
                    iaa.Sometimes(
                        0.4,
                        iaa.OneOf([
                            iaa.GammaContrast((0.5, 2.0)),
                            iaa.AllChannelsCLAHE(clip_limit=(1, 10)),
                            iaa.HistogramEqualization(),
                        ])
                    ),
                    iaa.Sometimes(
                        0.6,
                        iaa.OneOf([
                            iaa.BlendAlpha([0.25, 0.75], iaa.MedianBlur(13)),
                            iaa.GaussianBlur(sigma=(0.0, 3.0)),
                            iaa.MotionBlur(k=15)
                        ]),
                    ),
                    iaa.Sometimes(
                        0.2,
                        iaa.OneOf([
                            # iaa.BlendAlphaHorizontalLinearGradient(
                            #     iaa.TotalDropout(1.0),
                            #     min_value=0.2, max_value=0.8)
                            iaa.BlendAlphaHorizontalLinearGradient(
                                iaa.Lambda(lambda x, r, p, h: [cv2.filter2D(x[0],-1, np.ones([11, 11]) / 121)]),
                                start_at=(0.0, 1.0), end_at=(0.0, 1.0)),
                            iaa.Cartoon(),
                        ])
                    )
                ])
            ])
        return self._color_transf
    
    def filter_insert_kp_mtea(self, aug_kps: KeypointsOnImage, index: int) -> List[MetaKP]:
        assert len(aug_kps) == self.grid_size ** 2
        abs_top_right_coord = list(itertools.product(range(self.grid_size), range(self.grid_size)))
        abs_coord = [(x, y) for x, y in abs_top_right_coord]
        
        meta_kps = []
        for a, b in zip(aug_kps, abs_coord):
            if not a.is_out_of_image(tuple(self.output_size)):
                aug_grid_kp = (a.x // self.grid_cell, a.y // self.grid_cell)
                meta_kps.append(MetaKP(aug_kps, aug_grid_kp, b, index, self.output_size))
        return meta_kps

    def kp_to_4d_onehot(self, kps: List[MetaKP]) -> torch.Tensor:
        onehot = torch.zeros([self.grid_size,] * 4, dtype=torch.float32)
        for kp in kps:
            ax, ay = kp.aug_grid_kp
            x, y = kp.src_grid_kp
            onehot[y, x, round(ay), round(ax)] = 1
        return onehot
    
    def _norm_image(self, x):
        return tvtf.normalize(tvtf.to_tensor(x), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __call__(self, image: Image, index: int) -> Dict[str, torch.Tensor]:
        np_img = np.asarray(image)

        base_img = self.uni_size_transf(image=np_img)
        aug_imgs = []
        aug_kps = []
        target_corrs = []
        for _ in range(self.n_derive):
            deriv_img, kps = self.gemo_transf(image=base_img, keypoints=self.grid_kp)
            deriv_img, kps = self.color_transf(image=deriv_img, keypoints=kps)
            mkps = self.filter_insert_kp_mtea(kps, index)
            assert deriv_img.shape == base_img.shape, f"{deriv_img.shape} != {base_img.shape}"
            deriv_img = self._norm_image(deriv_img) if self.norm else torch.from_numpy(deriv_img)
            aug_imgs.append(deriv_img)
            aug_kps.append(mkps)
            target_corrs.append(self.kp_to_4d_onehot(mkps))
        
        base_img = self._norm_image(base_img) if self.norm else torch.from_numpy(base_img)
        
        return {
            "base_img": base_img,
            "aug_imgs": torch.stack(aug_imgs),
            "aug_kps": aug_kps,
            "target_corrs": torch.stack(target_corrs),
            "base_img_idx": torch.tensor([index]),
            "aug_img_idx": torch.tensor([index] * (self.n_derive)),
        }


if __name__ == '__main__':
    paug = PairAug()
    img = np.ones([720, 480, 3], dtype=np.uint8)
    img = Image.fromarray(img)
    d = paug(img, 0)
    # print(paug.collect([d, d]))
    
    for k, v in paug.collect([d, d]).items():
        if type(v) is torch.Tensor:
            print(k, v.shape)
        else:
            print(k, len(v))