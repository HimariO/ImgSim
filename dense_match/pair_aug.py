import itertools
from typing import *
from dataclasses import dataclass

import cv2
from imgaug.augmenters.meta import OneOf
from imgaug.augmenters.size import CropToFixedSize
import torch
import numpy as np
import imgaug as ia
from PIL import Image
from imgaug import augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage


@dataclass
class MetaKP:
    aug_kp: Keypoint  # 0 ~ img width/height
    aug_grid_kp: Tuple[float]  # 0 ~ grid(feature map) width/height
    src_grid_kp: Tuple[float]  # 0 ~ grid(feature map) width/height
    src_img_id: int  # will come to handy when we blend two sample together
    src_img_size: Union[Tuple[int], List[int]]


class PairAug:

    def __init__(self, n_deriv=3, output_size=[320, 320], sample_rate=16):
        self.n_derive = n_deriv
        self.output_size = output_size
        self.grid_size = output_size[0] // sample_rate
        self.grid_cell = output_size[0] // self.grid_size
    
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
                shape=[imsize, imsize, 3]
            )
        return self._grid_kp.deepcopy()

    @property
    def uni_size_transf(self) -> iaa.Augmenter:
        if not hasattr(self, '_uni_size_transf'):
            self._uni_size_transf = iaa.Sequential([
                iaa.CropToAspectRatio(1),
                iaa.OneOf([
                    iaa.CropToFixedSize(
                        width=self.output_size[0],
                        height=self.output_size[1]),
                    iaa.Resize(self.output_size),
                    iaa.Sequential([
                        iaa.Resize((0.5, 1.0)),
                        iaa.Reisze({
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
                iaa.OneOf([
                    iaa.PiecewiseAffine(scale=(0.01, 0.1)),
                    iaa.ShearX((-20, 20)),
                    iaa.ShearY((-20, 20)),
                    iaa.ScaleX((0.5, 1.5)),
                    iaa.ScaleY((0.5, 1.5)),
                ]),
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
                    iaa.OneOf([
                        iaa.GammaContrast((0.5, 2.0)),
                        iaa.AllChannelsCLAHE(clip_limit=(1, 10)),
                        iaa.HistogramEqualization(),
                    ]),
                    iaa.OneOf([
                        iaa.BlendAlpha([0.25, 0.75], iaa.MedianBlur(13)),
                        iaa.GaussianBlur(sigma=(0.0, 3.0)),
                        iaa.MotionBlur(k=15)
                    ]),
                    iaa.OneOf([
                        # iaa.BlendAlphaHorizontalLinearGradient(
                        #     iaa.TotalDropout(1.0),
                        #     min_value=0.2, max_value=0.8)
                        iaa.BlendAlphaHorizontalLinearGradient(
                            iaa.Lambda(lambda x, r, p, h: [cv2.filter2D(x[0],-1, np.ones([11, 11]) / 121)]),
                            start_at=(0.0, 1.0), end_at=(0.0, 1.0)),
                        iaa.Cartoon(),
                    ])
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
        onehot = torch.zeros([*self.grid_size, *self.grid_size], dtype=torch.float32)
        for kp in kps:
            ax, ay = kp.aug_grid_kp, kp.aug_grid_kp
            x, y = kp.src_grid_kp
            onehot[x, y, ax, ay] = 1
        return onehot

    def __call__(self, image: Image, index: int) -> Tuple[torch.Tensor]:
        np_img = np.asarray(image)

        base_img = self.uni_size_transf(np_img)
        aug_imgs = []
        aug_kps = []
        target_corrs = []
        for _ in range(self.n_derive):
            deriv_img, kps = self.gemo_transf(image=base_img, keypoints=self.grid_kp)
            deriv_img, kps = self.color_transf(image=deriv_img, keypoints=kps)
            mkps = self.filter_insert_kp_mtea(kps, index)
            
            aug_imgs.append(deriv_img)
            aug_kps.append(mkps)
            target_corrs.append(self.kp_to_4d_onehot(mkps))
        return base_img, aug_imgs, aug_kps, target_corrs
        