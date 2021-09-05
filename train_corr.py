import sys
import glob
import time

import fire
import numpy as np
import torch
import pytorch_lightning as pl
from loguru import logger
from einops import rearrange


from COTR.models.cotr_model import COTR, build
from COTR.models.pl_cotr import LitCOTR
from COTR.options.options import *
from COTR.options.options_utils import *
from dino import folder
from dense_match.pair_aug import PairAug, CacheAuged
from visual_util import CorrVis
from nali_dev import build_pairaug_iterator


def _cotr():
    parser = argparse.ArgumentParser()
    set_COTR_arguments(parser)
    parser.add_argument('--out_dir', type=str, default=general_config['out'], help='out directory')
    parser.add_argument('--load_weights', type=str, default=None, help='load a pretrained set of weights, you need to provide the model id')
    parser.add_argument('--max_corrs', type=int, default=100, help='number of correspondences')

    opt = parser.parse_args()
    opt.command = ' '.join(sys.argv)

    layer_2_channels = {'layer1': 256,
                        'layer2': 512,
                        'layer3': 1024,
                        'layer4': 2048, }
    opt.dim_feedforward = layer_2_channels[opt.layer]
    return build(opt)

def build_cotr(ckpt=None):
    args = {
        "backbone": 'resnet50',
        "hidden_dim": 256,
        "dilation": False,
        "dropout": 0.1,
        "nheads": 8,
        "layer": 'layer3',
        "enc_layers": 6,
        "dec_layers": 6,
        "position_embedding": 'lin_sine',
        "cat_img": False,
        # "lr_backbone": 1e-6,  # NOTE: using this arg will unfreeze backbone, potentialy cause NaN in the training process as CNN feature scale going up.
    }
    layer_2_channels = {
        'layer1': 256,
        'layer2': 512,
        'layer3': 1024,
        'layer4': 2048,
    }
    args['dim_feedforward'] = layer_2_channels[args['layer']]

    if ckpt is not None:
        logger.info(f"Load checkpoint: {os.path.basename(ckpt)}")
        cotr = LitCOTR.load_from_checkpoint(
            ckpt,
            trasnformer_args=args,
            backbone_args=args,
            strict=False)
    else:
        cotr = LitCOTR(args, args)
        state = torch.load('../COTR/out/default/checkpoint.pth.tar', map_location='cpu')['model_state_dict']
        cotr.load_state_dict(state, strict=False)

    return cotr


def debug_dataset(lit_img: folder.LitImgFolder):
    train_loader = lit_img.train_dataloader()
    val_loader = lit_img.val_dataloader()
    print(len(train_loader.dataset))
    print(len(train_loader.dataset.img_list))
    print(len(val_loader.dataset))
    print(len(val_loader.dataset.img_list))

    A = B= time.time()
    N = 128
    for i, data in enumerate(val_loader):
        D = time.time() - A
        A = time.time()
        logger.info(f"[{i}] {D:.5f}")
        # print('base_img: ', data['base_img'].shape, 'aug_imgs: ', data['aug_imgs'].shape)
        # print(data['base_img_idx'])
        # print(data['aug_img_idx'])
        if i > N: break
    D = time.time() - B
    speed = D / N
    logger.info(f"{speed} = {D} / {N}")
    sys.exit(0)


def model_profiling(model: LitCOTR, lit_img: folder.LitImgFolder):
    model = model.cuda()
    
    val_loader = lit_img.val_dataloader()
    batches = []
    for i, data in enumerate(val_loader):
        logger.info(f'Load batch: {i}')
        data = {k: v.cuda() for k, v in data.items() if type(v) is torch.Tensor}
        batches.append(data)
        if i > 1: break
    
    start_time = time.time()
    iters = 1024
    for i in range(iters):
        # loss = model.training_step(batches[0], 0)
        logger.info(f"{i}/{iters}")
        loss = model.forward_step(batches[0], 0)
        
    elapsed = time.time() - start_time
    speed = elapsed / iters
    print(f"{speed:.4f} = {elapsed:.4f} / {iters}")

@logger.catch(reraise=True)
def train(ckpt=None, overfit=False):
    p_aug = PairAug(n_deriv=3, output_size=[256, 256])
    lit_img = folder.LitImgFolder(
        '/home/ron/Downloads/fb-isc/train',
        p_aug,
        batch_size=96,
        num_worker=24)
    
    model = build_cotr(ckpt=ckpt)

    if overfit:
        logger.debug('Try to overfit on batch')
        trainer = pl.Trainer(
            accumulate_grad_batches=1,
            val_check_interval=1.0,
            checkpoint_callback=False,
            callbacks=[],
            default_root_dir='checkpoints/overfit',
            gpus=1,
            precision=16,
            max_steps=1000,
            overfit_batches=128,
            limit_val_batches=0,
        )

        """
        manual overfit rountine
        """
        _loader = lit_img.train_dataloader()
        _batch = next(iter(_loader))
        batch = {k: v.cuda() for k, v in _batch.items() if type(v) is torch.Tensor}
        adam = model.configure_optimizers()
        model = model.cuda()
        
        for step in range(64):
            adam.zero_grad()
            pred = model.forward_step(batch, 0)
            pred['corr_loss'].backward()
            adam.step()
            logger.debug(f"step-{step} loss: {pred['corr_loss']}")
        
        fit_res = {
            'batch': batch,
            'predict': {k: v.detach().cpu() for k, v in pred.items()}
        }
        torch.save(fit_res, 'debug.pth')
    else:
        trainer = pl.Trainer(
            accumulate_grad_batches=1,
            val_check_interval=1.0,
            checkpoint_callback=True,
            callbacks=[],
            default_root_dir='checkpoints/train',
            gpus=1,
            precision=16,
            terminate_on_nan=True,
        )
        trainer.fit(model, datamodule=lit_img)


@logger.catch(reraise=True)
def train_dali(ckpt=None, overfit=False):
    train_iter = build_pairaug_iterator(
        '/home/ron/Downloads/fb-isc/train',
        batch_size=96,
        output_size=[256, 256],
        max_iter=100_000)
    val_iter = build_pairaug_iterator(
        '/home/ron/Downloads/fb-isc/query',
        batch_size=96,
        output_size=[256, 256],
        max_iter=10_000)
    
    model = build_cotr(ckpt=ckpt)

    if overfit:
        logger.debug('Try to overfit on batch')
        trainer = pl.Trainer(
            accumulate_grad_batches=1,
            val_check_interval=1.0,
            checkpoint_callback=False,
            callbacks=[],
            default_root_dir='checkpoints/overfit',
            gpus=1,
            precision=16,
            max_steps=1000,
            overfit_batches=128,
            limit_val_batches=0,
        )

        """
        manual overfit rountine
        """
        _batch = next(iter(val_iter))
        batch = {k: v.cuda() for k, v in _batch.items() if type(v) is torch.Tensor}
        adam = model.configure_optimizers()
        model = model.cuda()
        
        for step in range(64):
            adam.zero_grad()
            pred = model.forward_step(batch, 0)
            pred['corr_loss'].backward()
            adam.step()
            logger.debug(f"step-{step} loss: {pred['corr_loss']}")
        
        fit_res = {
            'batch': batch,
            'predict': {k: v.detach().cpu() for k, v in pred.items()}
        }
        torch.save(fit_res, 'debug.pth')
    else:
        trainer = pl.Trainer(
            accumulate_grad_batches=1,
            val_check_interval=1.0,
            checkpoint_callback=True,
            callbacks=[],
            default_root_dir='checkpoints/train',
            gpus=1,
            precision=16,
            terminate_on_nan=True,
        )
        trainer.fit(model, train_dataloader=train_iter, val_dataloaders=val_iter)


@logger.catch
def debug():
    p_aug = PairAug(n_deriv=3, norm=True, output_size=[256, 256])
    lit_img = folder.LitImgFolder(
        '/home/ron/Downloads/fb-isc/train',
        p_aug,
        batch_size=96,
        num_worker=24)
    
    debug_dataset(lit_img)

    # model = build_cotr()
    # model_profiling(model, lit_img)


@logger.catch(reraise=True)
def vis_corr(ckpt):
    p_aug = PairAug(n_deriv=2, norm=False, output_size=[256, 256])
    lit_img = folder.LitImgFolder(
        '/home/ron/Downloads/fb-isc/train',
        p_aug,
        batch_size=9,
        num_worker=16)
    model = build_cotr(ckpt=ckpt).cuda()

    vis = CorrVis(p_aug.grid_size)
    val_loader = lit_img.train_dataloader()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            data = {
                k: (p_aug._norm_image(rearrange(v, 'b h w c->b c h w').float()).cuda()
                    if k in ['aug_imgs', 'base_img']
                    else v.cuda())
                for k, v in batch.items()
                if type(v) is torch.Tensor
            }
            # data['aug_imgs'] = p_aug._norm_image(data['aug_imgs'])
            # data['base_img'] = p_aug._norm_image(data['base_img'])
            pred_dict = model.forward_step(data, i)
            logger.info(f"corr_loss: {pred_dict['corr_loss']}")

            corrs = pred_dict['pred_corr_volum'].cpu()
            for j, aug_img in enumerate(batch['aug_imgs'].cpu().numpy()):
                base = batch['base_img'][j // p_aug.n_derive]
                corr = corrs[j]
                vis.show_corr_mapping(corr, base, aug_img)
                input('Press Enter to show next')

            if i > 1:
                break


if __name__ == '__main__':
    fire.Fire({
        'debug': debug,
        'train': train,
        'train_dali': train_dali,
        'vis': vis_corr,
    })