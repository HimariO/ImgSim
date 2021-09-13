import math
from typing import *

import pysnooper
import numpy as np
import torch
from torch.nn.modules import sparse
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
from .focal_loss import FocalLossV2
from .partial_fc import ArcMarginProduct, PartialFC
from dense_match.margin import SampledMarginLoss


class NormMeanSquaredError(torchmetrics.MeanSquaredError):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, squared=False, **kwargs)
        self.loss_fn = FocalLossV2(reduction='mean')
    
    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None):
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        if mask is None:
            # sum_squared_error = self.loss_fn(preds, target)
            sum_squared_error = torch.nn.functional.mse_loss(preds, target)
        else:
            sum_squared_error = mask * (preds - target)**2
            # sum_squared_error = mask * torch.nn.functional.binary_cross_entropy(preds, target.int(), reduction='none')
            sum_squared_error = sum_squared_error.sum() / mask.sum()
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
    
    def compute(self) -> torch.Tensor:
        return self.sum_squared_error / self.total

class MovingAverage(torchmetrics.MeanSquaredError):
    
    def update(self, loss: torch.Tensor, num_entry: torch.Tensor):
        """Update state with predictions and targets.

        Args:
            embed: Predictions from model
            target: Ground truth values
        """
        self.sum_squared_error += loss
        self.total += num_entry
    
    def compute(self) -> torch.Tensor:
        return self.sum_squared_error / self.total


class HybirdAdam(torch.optim.Optimizer):

    def __init__(self, params, sparse_params, lr=1e-4):
        self.adam = torch.optim.AdamW(params, lr=lr)
        self.sparse_adam = torch.optim.SparseAdam(sparse_params, lr=lr)
    
    def step(self, closure=None):
        self.adam.stpe(closure-closure)
        self.sparse_adam.stpe(closure-closure)
    
    def zero_grad(self, set_to_none=False):
        self.adam.zero_grad(set_to_none=set_to_none)
        self.sparse_adam.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self):
        return {
            'dense': self.adam.state_dict(),
            'sparse': self.sparse_adam.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.adam.load_state_dict(state_dict['dense'])
        self.sparse_adam.load_state_dict(state_dict['sparse'])
    



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
        # self.corr_norm_scale = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        # self.embed_norm_scale = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.spec_token = nn.Embedding(10, hidden_dim)
        self.use_cls_token = True
        self.cnn_corr = True
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
        src_proj = self.input_proj(src)
        assert mask is not None

        if not self.cnn_corr or self.use_cls_token:
            cls_embed = self.spec_token(torch.zeros(src.shape[0], dtype=torch.int64).to(src.device))
            hs, cls = self.transformer(
                src_proj,
                None,
                pos[-1],
                cls_embed=cls_embed if self.use_cls_token else None,
            )

        if self.cnn_corr:
            hs = src_proj
            
        if self.use_cls_token:
            embed = self.embed_head(cls)
            norm_embed = embed / (torch.linalg.norm(embed, ord=2, dim=-1, keepdim=True) + 1e-6)
        else:
            embed = rearrange(hs, 'b c h w -> b (h w) c')
            embed = self.embed_head(rearrange(hs, 'b c h w -> b (h w) c'))
            norm_embed = embed / (torch.linalg.norm(embed, ord=2, dim=-1, keepdim=True) + 1e-6)
            norm_embed = rearrange(norm_embed, 'b (h w) c -> b c h w', h=hs.shape[-2], w=hs.shape[-1])
            norm_embed = torch.mean(norm_embed, dim=[2, 3])
        # norm_embed = norm_embed * self.embed_norm_scale
        return (
            self.norm_4d(hs),
            norm_embed,
            src
        )
    
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
                ref_emb,
                aug_emb,
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
        corr_loss = self.train_mse(pred_corr_volum, batch['target_corrs'], mask=None)
        margin_loss = self.train_margin(
            torch.cat([
                ref_emb,
                aug_emb,
            ]),
            torch.cat([
                batch['base_img_idx'],
                batch['aug_img_idx'],
            ])
        )
        
        self.log('train_mse_step', self.train_mse.compute(), prog_bar=True)
        self.log('train_margin_step', self.train_margin.compute(), prog_bar=True)
        if self.global_step % 100 == 0:
            self.logger.experiment.add_histogram(
                'train/transformer/hs', torch.cat([tmp1, tmp2], dim=0), self.global_step)
            self.logger.experiment.add_histogram(
                'train/transformer/pred_corr', pred_corr_volum, self.global_step)
            # self.logger.experiment.add_scalar(
            #     'norm_scale/feat_map', self.corr_norm_scale, self.global_step)
            # self.logger.experiment.add_scalar(
            #     'norm_scale/embed_vec', self.embed_norm_scale, self.global_step)
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
                    ref_emb,
                    aug_emb,
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


class LitSelfCor(LitCOTR):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_self_mse = NormMeanSquaredError()

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
        self_corr_volum = self.correlation(n_ref_hs, n_ref_hs)
        corr_loss = self.train_mse(pred_corr_volum, batch['target_corrs'], mask=None)
        self_corr = torch.eye(h * w).view(h, w, h, w).to(pred_corr_volum.device)
        self_corr_loss = self.train_self_mse(self_corr_volum, self_corr, mask=None)
        margin_loss = self.train_margin(
            torch.cat([
                ref_emb,
                aug_emb,
            ]),
            torch.cat([
                batch['base_img_idx'],
                batch['aug_img_idx'],
            ])
        )
        
        self.log('train_mse_step', self.train_mse.compute(), prog_bar=True)
        self.log('train_self_mse_step', self.train_self_mse.compute(), prog_bar=True)
        self.log('train_margin_step', self.train_margin.compute(), prog_bar=True)
        
        if self.global_step % 100 == 0:
            self.logger.experiment.add_histogram(
                'train/transformer/hs', torch.cat([tmp1, tmp2], dim=0), self.global_step)
            self.logger.experiment.add_histogram(
                'train/transformer/pred_corr', pred_corr_volum, self.global_step)
        
        return corr_loss + self_corr_loss * 0.5 + margin_loss * 0.5


class LitArcCor(LitCOTR):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arc_fc = ArcMarginProduct(
            PartialFC(1_000_000, self.transformer.d_model)
        )
        self.train_arc_ce = MovingAverage()
        self.train_arc_acc = torchmetrics.Accuracy()
        self.automatic_optimization = False

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
        corr_loss = self.train_mse(pred_corr_volum, batch['target_corrs'], mask=None)
        embeds = torch.cat([ref_emb, aug_emb,])
        img_idx = torch.cat([
            batch['base_img_idx'],
            batch['aug_img_idx'],
        ])
        with torch.no_grad():
            margin_loss = self.train_margin(embeds, img_idx)

        arc_logits, sub_label = self.arc_fc(embeds, img_idx)
        arc_ce = nn.functional.cross_entropy(arc_logits, sub_label)
        self.train_arc_ce.update(arc_ce, 1)
        self.train_arc_acc.update(arc_logits, sub_label)
        
        self.log('train_mse_step', self.train_mse.compute(), prog_bar=True)
        self.log('train_margin_step', self.train_margin.compute(), prog_bar=True)
        self.log('train_arc_ce', self.train_arc_ce.compute(), prog_bar=True)
        self.log('train_arc_acc', self.train_arc_acc.compute(), prog_bar=True)
        
        if self.global_step % 100 == 0:
            self.logger.experiment.add_histogram(
                'train/transformer/hs', torch.cat([tmp1, tmp2], dim=0), self.global_step)
            self.logger.experiment.add_histogram(
                'train/transformer/pred_corr', pred_corr_volum, self.global_step)
        
        opts = self.optimizers()
        for opt in opts: opt.zero_grad()
        loss = corr_loss + arc_ce
        self.manual_backward(loss)
        for opt in opts: opt.step()
    
    def training_epoch_end(self, training_step_outputs):
        # self.log('train_mse_epoch', self.train_mse.compute())
        self.train_mse.reset()
        # self.log('train_margin_epoch', self.train_margin.compute())
        self.train_margin.reset()
        self.train_arc_ce.reset()
        self.train_arc_acc.reset()
 
    def configure_optimizers(self):
        sparse_param = list(self.arc_fc.parameters())
        # sp_data = [p.data for p in sparse_param]
        # param = [p for p in self.parameters() if p.data not in sp_data]
        param = (
            list(self.transformer.parameters()) +
            list(self.embed_head.parameters()) + 
            list(self.backbone.parameters()) +
            list(self.spec_token.parameters()) +
            list(self.input_proj.parameters()) +
            list(self.corr_embed.parameters())
        )
        return [
            torch.optim.Adam(param, lr=1e-4),
            torch.optim.SparseAdam(sparse_param, lr=1e-4),
        ]