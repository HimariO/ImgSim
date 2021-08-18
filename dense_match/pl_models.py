import torch
import pytorch_lightning as pl

from .models.GLUNet.GLU_Net import GLUNet_model, GLUNetCorr


class LitGLUNetCorr(pl.LightningModule):

    GLOBAL_GOCOR =  {
        'optim_iter': 3,
        'apply_query_loss': True,
        'reg_kernel_size': 3,
        'reg_inter_dim': 16,
        'reg_output_dim': 16
    }
    
    LOCAL_GOCOR = {
        'optim_iter': 3
    }

    MEAN_VECTOR = [0.485, 0.456, 0.406]
    STD_VECTOR = [0.229, 0.224, 0.225]

    def __init__(self, iterative_refinement=True, global_corr_type='GlobalGOCor',
                global_gocor_arguments=GLOBAL_GOCOR, normalize='leakyrelu',
                local_corr_type='LocalGOCor', local_gocor_arguments=LOCAL_GOCOR,
                same_local_corr_at_all_levels=True):
        self.model = GLUNetCorr(
            iterative_refinement=iterative_refinement,
            global_corr_type=global_corr_type,
            global_gocor_arguments=global_gocor_arguments,
            normalize=normalize,
            local_corr_type=local_corr_type,
            local_gocor_arguments=local_gocor_arguments,
            same_local_corr_at_all_levels=same_local_corr_at_all_levels)
    
    def normalize_image(self, images: torch.Tensor):
        images = images.float() / 255
        if not hasattr(self, '_mean'):
            self._mean = torch.as_tensor(self.MEAN_VECTOR, dtype=images.dtype, device=images.device)
        if not hasattr(self, '_std'):
            self._std = torch.as_tensor(self.STD_VECTOR, dtype=images.dtype, device=images.device)
        images = images - self._mean[None, :, None, None]
        images = images / self._std[None, :, None, None]
        return images
    
    def _forward(self, images: torch.Tensor):
        images = self.normalize_image(images)
        feat_x16 = self.model.pyramid(images)[4]
        return feat_x16
    
    def forward(self, ref_imgs: torch.Tensor, que_imgs: torch.Tensor):
        c1 = self._forward(ref_imgs)
        c2 = self._forward(ref_imgs)
        corr4 = self.model.get_global_correlation(c1, c2)


    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, validation_step_outputs):
        self.log('val_aucc_epoch', self.val_accuracy.compute(), prog_bar=True)
        self.val_accuracy.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.00001)