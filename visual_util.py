import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import imgaug as ia
import skimage
from PIL import Image
from imgaug import augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from skimage.filters import rank
from skimage.morphology import disk


def hsv_to_rgb(h, s, v):
    if s == 0.0: return (v, v, v)
    i = int(h*6.) # XXX assume int() truncates!
    f = (h*6.)-i; p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    if i == 5: return (v, p, q)


def grid_colors(n):
    
    def hsv_to_rgb(h, s, v):
        if s == 0.0: return (v, v, v)
        i = int(h*6.) # XXX assume int() truncates!
        f = (h*6.)-i; p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
        if i == 0: return (v, t, p)
        if i == 1: return (q, v, p)
        if i == 2: return (p, v, t)
        if i == 3: return (p, q, v)
        if i == 4: return (t, p, v)
        if i == 5: return (v, p, q)
        
    colors = [
        tuple(v for v in hsv_to_rgb(i / n**2, 1, 1))
        for i in range(n**2)
    ]
    return colors


class CorrVis:
    
    def __init__(self, grid_size):
        self.colors = grid_colors(grid_size)
        random.shuffle(self.colors)

    def show_corr_mapping(self, corr_map: torch.Tensor, base_img, aug_img):
        h, w = corr_map.shape[-2:]
        corr_map = corr_map.view(h, w, h * w)
        max_conf, flat_idx = corr_map.max(dim=-1)
        max_conf = torch.flatten(max_conf).cpu().numpy()

        x = flat_idx % w
        y = flat_idx // w
        x = torch.flatten(x).float().cpu().numpy() + 0.5
        y = torch.flatten(y).float().cpu().numpy() + 0.5

        mapping = [
            {
                'ref': (i % w / w, i // w / h),
                'que': (_x / w, _y / h, _c)
            }
            for i, _x, _y, _c in zip(range(len(x)), x, y, max_conf)
        ]
        mapping = mapping[::2]

        h, w = aug_img.shape[:2]
        merge_image = np.concatenate([base_img, aug_img], axis=1)
        mapping = [m for m in mapping if m['que'][-1] > .1]

        plt.figure(figsize=(30, 30))
        plt.imshow(merge_image)
        
        src_dot_xs = []
        src_dot_ys = []
        src_dot_c = []
        for k, m in enumerate(mapping):
            x1, y1 = m['ref']
            x2, y2, _ = m['que']
            x1 = (x1 * w)
            x2 = (x2 * w)
            y1 = (y1 * h)
            y2 = (y2 * h)
            plt.arrow(x1, y1, x2 + w - x1, y2 - y1,
                color=self.colors[k], width=.5, head_width=3, alpha=0.3)
            
            src_dot_xs.append(x1)
            src_dot_ys.append(y1)
            src_dot_c.append(self.colors[k])
            
            src_dot_xs.append(x2 + w)
            src_dot_ys.append(y2)
            src_dot_c.append(self.colors[k])
        plt.scatter(src_dot_xs, src_dot_ys, c=src_dot_c)
        plt.show()