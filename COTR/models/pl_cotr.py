import math
from typing import *

import pysnooper
import numpy as np
import torch
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from einops import rearrange, repeat
from easydict import EasyDict as edict
from loguru import logger

from COTR.utils import debug_utils, constants, utils
from .misc import (NestedTensor, nested_tensor_from_tensor_list)
from .backbone import build_backbone
from .transformer import build_halfformer
from .position_encoding import NerfPositionalEncoding, MLP
from dense_match.margin import SampledMarginLoss

class NormMeanSquaredError(torchmetrics.MeanSquaredError):
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sum_squared_error = torch.nn.functional.mse_loss(preds, target)
        n_obs = 1  # NOTE: sum_squared_error is already batch-wise normalized

        self.sum_squared_error += sum_squared_error
        self.total += n_obs
        return sum_squared_error


class MovingSampleMargin(torchmetrics.MeanSquaredError):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sampling_args = kwargs.pop('sampling_args', {})
        margin_args = kwargs.pop('margin_args', {})
        self.loss_fn = SampledMarginLoss(sampling_args=sampling_args, margin_args=margin_args)
    
    def update(self, embed: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets.

        Args:
            embed: Predictions from model
            target: Ground truth values
        """
        sum_squared_error = self.loss_fn(embed, target)
        n_obs = 1  # NOTE: sum_squared_error is already batch-wise normalized

        self.sum_squared_error += sum_squared_error
        self.total += n_obs
        return sum_squared_error


class LitCOTR(pl.LightningModule):

    def __init__(self, trasnformer_args: dict, backbone_args: dict, sine_type='lin_sine', embed_dim=256):
        super().__init__()
        self.transformer = build_halfformer(edict(trasnformer_args))
        hidden_dim = self.transformer.d_model
        self.GoM = 2
        self.corr_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.backbone = build_backbone(edict(backbone_args))
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        self.embed_head = nn.Linear(hidden_dim, embed_dim, bias=False)
        self._reset_head_parameters()
        
        self.train_mse = NormMeanSquaredError()
        self.train_margin = MovingSampleMargin()
        self.val_mse = NormMeanSquaredError()
        self.val_margin = MovingSampleMargin()
        
        # self.sample_margin = SampledMarginLoss()
    
    def _reset_head_parameters(self):
        for p in self.embed_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def correlation(self, feat_map_a, feat_map_b):
        h, w = feat_map_a.shape[-2:]
        batched_feat_vec_a = rearrange(feat_map_a, "b c h w -> b (h w) c")
        batched_feat_vec_b = rearrange(feat_map_b, "b c h w -> b c (h w)")
        corr_volume = torch.bmm(batched_feat_vec_a, batched_feat_vec_b)
        corr_volume = rearrange(corr_volume, "b (h1 w1) (h2 w2) -> b h1 w1 h2 w2", h1=h, h2=h, w1=w, w2=w)
        return torch.clip(corr_volume, min=0)
    
    def norm_4d(self, x):
        h = x.flatten(-2, -1)
        h = h / (torch.linalg.norm(h, ord=2, dim=1, keepdim=True) + 1e-6)
        return h.view(*x.shape)

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs: torch.Tensor = self.transformer(
            self.input_proj(src),
            None,
            pos[-1]
        )
        embed = rearrange(hs, 'b c h w -> b (h w) c')
        embed = self.embed_head(rearrange(hs, 'b c h w -> b (h w) c'))
        norm_embed = embed / (torch.linalg.norm(embed, ord=2, dim=-1, keepdim=True) + 1e-6)
        norm_embed = rearrange(norm_embed, 'b (h w) c -> b c h w', h=hs.shape[-2], w=hs.shape[-1])
        return self.norm_4d(hs), norm_embed, src
    
    def forward_compare(img_a, img_b):
        pass
    
    # @pysnooper.snoop()
    def forward_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        nest_base = NestedTensor(batch['base_img'], torch.zeros_like(batch['base_img']))
        nest_aug = NestedTensor(batch['aug_imgs'], torch.zeros_like(batch['aug_imgs']))
        ref_hs, ref_emb, tmp1 = self.forward(nest_base)
        aug_hs, aug_emb, tmp2 = self.forward(nest_aug)
        assert len(aug_hs) % len(ref_hs) == 0
        b, c, h ,w = ref_hs.shape
        n_der = len(aug_hs) // len(ref_hs)
        n_ref_hs = repeat(ref_hs, f"b c h w->b ({n_der} c) h w")
        n_ref_hs = n_ref_hs.view(b * n_der, c, h, w)
        
        pred_corr_volum = self.correlation(n_ref_hs, aug_hs)
        corr_loss = self.train_mse(pred_corr_volum, batch['target_corrs'])
        margin_loss = self.train_margin(
            torch.cat([
                torch.mean(ref_emb, dim=[2, 3]),
                torch.mean(aug_emb, dim=[2, 3]),
            ]),
            torch.cat([
                batch['base_img_idx'],
                batch['aug_img_idx'],
            ])
        )
        return {
            'pred_corr_volum': pred_corr_volum,
            'ref_emb': ref_emb,
            'aug_emb': aug_emb,
            'corr_loss': corr_loss,
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx):
        nest_base = NestedTensor(batch['base_img'], torch.zeros_like(batch['base_img']))
        nest_aug = NestedTensor(batch['aug_imgs'], torch.zeros_like(batch['aug_imgs']))
        ref_hs, ref_emb, tmp1 = self.forward(nest_base)
        aug_hs, aug_emb, tmp2 = self.forward(nest_aug)
        assert len(aug_hs) % len(ref_hs) == 0
        b, c, h ,w = ref_hs.shape
        n_der = len(aug_hs) // len(ref_hs)
        n_ref_hs = repeat(ref_hs, f"b c h w->b ({n_der} c) h w")
        n_ref_hs = n_ref_hs.view(b * n_der, c, h, w)
        
        pred_corr_volum = self.correlation(n_ref_hs, aug_hs)
        # corr_loss = torch.nn.functional.mse_loss(pred_corr_volum, batch['target_corrs'])
        # corr_loss = corr_loss.sum() / batch['target_corrs'].sum()
        corr_loss = self.train_mse(pred_corr_volum, batch['target_corrs'])
        margin_loss = self.train_margin(
            torch.cat([
                torch.mean(ref_emb, dim=[2, 3]),
                torch.mean(aug_emb, dim=[2, 3]),
            ]),
            torch.cat([
                batch['base_img_idx'],
                batch['aug_img_idx'],
            ])
        )
        
        self.log('train_mse_step', self.train_mse.compute(), prog_bar=True)
        self.log('train_margin_step', self.train_margin.compute(), prog_bar=True)
        if batch_idx % 100 == 0:
            self.logger.experiment.add_histogram(
                'train/transformer/hs', torch.cat([tmp1, tmp2], dim=0), batch_idx)
        return corr_loss + margin_loss * 0.5
    
    def validation_step(self, batch, batch_idx):
        nest_base = NestedTensor(batch['base_img'], torch.zeros_like(batch['base_img']))
        nest_aug = NestedTensor(batch['aug_imgs'], torch.zeros_like(batch['aug_imgs']))
        ref_hs, ref_emb = self.forward(nest_base)[:2]
        aug_hs, aug_emb = self.forward(nest_aug)[:2]
        assert len(aug_hs) % len(ref_hs) == 0
        b, c, h ,w = ref_hs.shape
        n_der = len(aug_hs) // len(ref_hs)
        n_ref_hs = repeat(ref_hs, f"b c h w->b ({n_der} c) h w")
        n_ref_hs = n_ref_hs.view(b * n_der, c, h, w)
        
        pred_corr_volum = self.correlation(n_ref_hs, aug_hs)
        corr_loss = self.val_mse(pred_corr_volum, batch['target_corrs'])
        self.log('val_mse_step', self.val_mse.compute(), prog_bar=True)

        # NOTE: val sanity check will use the same target index for all samples, 
        #       margin loss is expected to fail in this situation.
        # with logger.catch(exception=ValueError):
        try:
            margin_loss = self.val_margin(
                torch.cat([
                    torch.mean(ref_emb, dim=[2, 3]),
                    torch.mean(aug_emb, dim=[2, 3]),
                ]),
                torch.cat([
                    batch['base_img_idx'],
                    batch['aug_img_idx'],
                ])
            )
            self.log('val_margin_step', self.val_margin.compute(), prog_bar=True)
        except ValueError as e:
            logger.exception(str(e))
        return {
            'pred_corr_volum': pred_corr_volum,
            'corr_loss': corr_loss,
        }

    def training_epoch_end(self, training_step_outputs):
        self.log('train_mse_epoch', self.train_mse.compute())
        self.train_mse.reset()
        self.log('train_margin_epoch', self.train_margin.compute())
        self.train_margin.reset()
    
    def validation_epoch_end(self, validation_step_outputs):
        self.log('val_mse_epoch', self.val_mse.compute(), prog_bar=True)
        self.val_mse.reset()
        self.log('val_margin_epoch', self.val_margin.compute(), prog_bar=True)
        self.val_margin.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0001)
