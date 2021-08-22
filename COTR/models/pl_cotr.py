import math
from typing import *

import numpy as np
import torch
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from einops import rearrange, repeat
from easydict import EasyDict as edict

from COTR.utils import debug_utils, constants, utils
from .misc import (NestedTensor, nested_tensor_from_tensor_list)
from .backbone import build_backbone
from .transformer import build_halfformer
from .position_encoding import NerfPositionalEncoding, MLP


class NormMeanSquaredError(torchmetrics.MeanSquaredError):
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sum_squared_error = torch.nn.functional.mse_loss(preds, target)
        sum_squared_error = sum_squared_error.sum() / target.sum()
        n_obs = len(preds)

        self.sum_squared_error += sum_squared_error
        self.total += n_obs


class LitCOTR(pl.LightningModule):

    def __init__(self, trasnformer_args: dict, backbone_args: dict, sine_type='lin_sine'):
        super().__init__()
        self.transformer = build_halfformer(edict(trasnformer_args))
        hidden_dim = self.transformer.d_model
        self.corr_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.backbone = build_backbone(edict(backbone_args))
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
    
    def correlation(self, feat_map_a, feat_map_b):
        h, w = feat_map_a.shape[-2:]
        batched_feat_vec_a = rearrange(feat_map_a, "b c h w -> b (h w) c")
        batched_feat_vec_b = rearrange(feat_map_b, "b c h w -> b c (h w)")
        corr_volume = torch.bmm(batched_feat_vec_a, batched_feat_vec_b)
        corr_volume = rearrange(corr_volume, "b (h1 w1) (h2 w2) -> b h1 w1 h2 w2", h1=h, h2=h, w1=w, w2=w)
        return corr_volume

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(
            self.input_proj(src),
            mask,
            pos[-1]
        )
        return hs
    
    def forward_compare(img_a, img_b):
        pass
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        ref_hs = self.forward(NestedTensor(batch['base_img'], torch.zeros_like(batch['base_img'])))
        aug_hs = self.forward(NestedTensor(batch['aug_imgs'], torch.zeros_like(batch['aug_imgs'])))
        assert len(aug_hs) % len(ref_hs) == 0
        b, c, h ,w = ref_hs.shape
        n_der = len(aug_hs) // len(ref_hs)
        n_ref_hs = repeat(ref_hs, f"b c h w->b ({n_der} c) h w")
        n_ref_hs = n_ref_hs.view(b * n_der, c, h, w)
        
        pred_corr_volum = self.correlation(n_ref_hs, aug_hs)
        corr_loss = torch.nn.functional.mse_loss(pred_corr_volum, batch['target_corrs'])
        corr_loss = corr_loss.sum() / batch['target_corrs'].sum()
        
        self.log('train_mse_step', self.train_mse(pred_corr_volum, batch['target_corrs']), prog_bar=True)
        # if batch_idx % 10 == 0:
        #     self.logger.experiment.add_scalar()
        return corr_loss
    
    def validation_step(self, batch, batch_idx):
        ref_hs = self.forward(NestedTensor(batch['base_img'], torch.zeros_like(batch['base_img'])))
        aug_hs = self.forward(NestedTensor(batch['aug_imgs'], torch.zeros_like(batch['aug_imgs'])))
        assert len(aug_hs) % len(ref_hs) == 0
        b, c, h ,w = ref_hs.shape
        n_der = len(aug_hs) // len(ref_hs)
        n_ref_hs = repeat(ref_hs, f"b c h w->b ({n_der} c) h w")
        n_ref_hs = n_ref_hs.view(b * n_der, c, h, w)
        
        pred_corr_volum = self.correlation(n_ref_hs, aug_hs)
        corr_loss = torch.nn.functional.mse_loss(pred_corr_volum, batch['target_corrs'])
        corr_loss = corr_loss.sum() / batch['target_corrs'].sum()
        return {
            'pred_corr_volum': pred_corr_volum,
            'corr_loss': corr_loss,
        }

    def validation_epoch_end(self, validation_step_outputs):
        self.log('val_mse_epoch', self.val_mse.compute(), prog_bar=True)
        self.val_mse.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.00001)
