import math
import sys
import glob
import time
from pprint import pprint

import fire
import numpy as np
import torch
import pytorch_lightning as pl
from loguru import logger
from einops import rearrange
from torch.functional import norm
from torch.utils.data import dataloader


from COTR.models.cotr_model import COTR, build
from COTR.models.pl_cotr import LitCOTR, LitSelfCor, LitArcCor, Baseline
from COTR.options.options import *
from COTR.options.options_utils import *
from dino import folder
from dense_match.pair_aug import PairAug, CacheAuged, EZPairAug
from visual_util import CorrVis
from nali_dev import build_pairaug_iterator


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
        "lr_backbone": 1e-5,  # NOTE: using this arg will unfreeze backbone, potentialy cause NaN in the training process as CNN feature scale going up.
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
        cotr = LitArcCor.load_from_checkpoint(
            ckpt,
            trasnformer_args=args,
            backbone_args=args,
            strict=False)
    else:
        cotr = LitArcCor(args, args)
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
def train_dali(ckpt=None, overfit=False, resume=False):
    train_iter = build_pairaug_iterator(
        '/home/ron/Downloads/fb-isc/train',
        batch_size=96,
        output_size=[256, 256],
        max_iter=100_000 // 96)
    val_iter = build_pairaug_iterator(
        '/home/ron/Downloads/fb-isc/query',
        batch_size=96,
        output_size=[256, 256],
        max_iter=10_000 // 96)
    
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
        
        for step in range(128):
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
            default_root_dir='checkpoints/self_cor',
            gpus=1,
            precision=32,
            terminate_on_nan=True,
            resume_from_checkpoint=ckpt if resume else None,
        )
        # trainer.validate(model, dataloaders=[val_iter])
        trainer.fit(model, train_dataloader=train_iter, val_dataloaders=val_iter)


@logger.catch(reraise=True)
def train_baseline(ckpt=None, state_dict=None, overfit=False, resume=False):
    from nali_dev import CacheDataset

    train_imgs = glob.glob('/home/ron/Downloads/fb-isc/train/*.jpg')
    train_imgs = sorted(train_imgs)[:100_000]
    # # pprint(train_imgs[:3])
    # # sys.exit(0)
    train_iter = build_pairaug_iterator(
        train_imgs,
        batch_size=32,
        output_size=[256, 256],
        # max_iter=4000,
        unit_norm=False,
        easy=True)
    # val_iter = build_pairaug_iterator(
    #     '/home/ron/Downloads/fb-isc/query',
    #     batch_size=32,
    #     output_size=[256, 256],
    #     max_iter=1_000)

    # cache = CacheDataset('./dali_cache.pth')
    # train_iter = torch.utils.data.DataLoader(
    #     cache, batch_size=1,  num_workers=0, 
    #     shuffle=True, collate_fn=CacheDataset.debatch)

    p_aug = EZPairAug(n_deriv=3, output_size=[256, 256])
    lit_img = folder.LitImgFolder(
        train_imgs,
        p_aug,
        batch_size=64,
        num_worker=12,
        split=0.1)
    
    hparam = {
        "partial_fc": True,
        "lr": 1e-5,
    }
    if ckpt:
        assert os.path.exists(ckpt)
        logger.info(f"load_from_checkpoint: {ckpt}")
        model = Baseline.load_from_checkpoint(ckpt, strict=True, **hparam)
    else:
        model = Baseline(embed_dim=256, num_classes=len(train_imgs), **hparam)
    
    if state_dict:
        addition = ", this will override model parameter from ckpt!" if ckpt else ""
        logger.info(f"load_state_dict: {state_dict} {addition}")
        model.load_state_dict(torch.load(state_dict))

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
            overfit_batches=8,
            limit_val_batches=0,
        )

        """
        manual overfit rountine
        """
        _batch = next(iter(train_iter))
        batch = {k: v.cuda() for k, v in _batch.items() if type(v) is torch.Tensor}
        # adams = model.configure_optimizers()
        model = model.cuda()
        
        for step in range(128):
            # for adam in adams: adam.zero_grad()
            pred = model.training_step(batch, 0)
            # pred['corr_loss'].backward()
            # for adam in adams: adam.step()
            logger.debug(f"step-{step} loss: {pred['loss']}")
        
        fit_res = {
            'batch': batch,
            'predict': {k: v.detach().cpu() for k, v in pred.items()}
        }
        torch.save(fit_res, 'debug.pth')
    else:
        if resume:
            logger.warning(f'Resume from checkpoint: {ckpt}')
        trainer = pl.Trainer(
            accumulate_grad_batches=1,
            val_check_interval=1.0,
            checkpoint_callback=True,
            callbacks=[],
            default_root_dir='checkpoints/baseline',
            gpus=1,
            precision=16,
            terminate_on_nan=True,
            resume_from_checkpoint=ckpt if resume else None,
            max_epochs=100 if resume else 10,
        )
        # trainer.validate(model, dataloaders=[val_iter])
        try:
            # trainer.fit(model, train_dataloader=train_iter, val_dataloaders=None)
            trainer.fit(model, datamodule=lit_img)
        except RuntimeError as e:
            logger.error(e)
        
        # debug_ckpt = f"{trainer.global_step}-step.pth"
        # torch.save(model.state_dict(), debug_ckpt)
        try:
            _, emb = model.cuda()(torch.ones([1, 3, 256, 256]).cuda())
            torch.save(emb, f"{trainer.global_step}-emb.pth")
        except:
            breakpoint()
            print('end')



@logger.catch
def debug():
    # p_aug = PairAug(n_deriv=3, norm=True, output_size=[256, 256])
    # lit_img = folder.LitImgFolder(
    #     '/home/ron/Downloads/fb-isc/train',
    #     p_aug,
    #     batch_size=96,
    #     num_worker=24)
    
    # debug_dataset(lit_img)

    # model = build_cotr()
    # model_profiling(model, lit_img)

    # A = torch.ones(
    #     [1000, 32], requires_grad=True,
    #     device="cuda")
    # adam = torch.optim.Adam([A], lr=1e-2)
    
    # A = torch.nn.Embedding(1000, 32, device='cuda', sparse=True)
    # adam = torch.optim.SparseAdam(list(A.parameters()), lr=1e-2)
    
    # B = torch.normal(
    #     mean=torch.zeros([8, 32]),
    #     std=torch.ones([8, 32]))
    # Y = torch.zeros([8, 4])

    # # A = A.cuda()
    # B = B.cuda()
    # Y = Y.cuda()
    # indice = torch.tensor([1,4,6,2], dtype=torch.long, device='cuda')

    # for _ in range(100):
    #     sub_A = A(indice)

    #     adam.zero_grad()
    #     C = torch.nn.functional.linear(B, sub_A)
    #     loss = torch.abs(Y - C).sum()
    #     loss.backward()
    #     adam.step()
    #     breakpoint()
    # print(A(indice))
    # print(A(indice).mean(dim=0))
    # A_exclu = A[[i for i in range(100) if i not in [1,4,6,2]], :]
    # print(A_exclu.shape)
    # print(A_exclu.mean(dim=0))

    Baseline()


@logger.catch(reraise=True)
def vis_corr(ckpt):
    p_aug = PairAug(n_deriv=2, norm=False, output_size=[256, 256])
    lit_img = folder.LitImgFolder(
        '/home/ron/Downloads/fb-isc/train',
        p_aug,
        batch_size=9,
        num_worker=16)
    model = build_cotr(ckpt=ckpt).cuda()
    model.eval()

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



def vis_batch():
    import matplotlib.pyplot as plt
    
    train_imgs = glob.glob('/home/ron/Downloads/fb-isc/train/*.jpg')
    train_imgs = sorted(train_imgs)[:1000]

    p_aug = EZPairAug(n_deriv=3, output_size=[256, 256], norm=False)
    lit_img = folder.LitImgFolder(
        train_imgs,
        p_aug,
        batch_size=32,
        num_worker=0,
        split=0.1)
    
    iter = lit_img.train_dataloader()
    
    for i, batch in enumerate(iter):
        # batch[0]['images'].shape
        print(i)
        print(type(batch['base_img']), batch['base_img'].shape, batch['base_img'].dtype)
        print(type(batch['aug_imgs']), batch['aug_imgs'].shape, batch['aug_imgs'].dtype)
        print(type(batch['target_corrs']), batch['target_corrs'].shape, batch['target_corrs'].dtype)
        print(batch['base_img_idx'])
        print(batch['aug_img_idx'])

        # for j, aug_img in enumerate(batch['aug_imgs'].cpu().numpy()):
        #     base = batch['base_img'][j // 3].cpu()
        batch_size = len(batch['base_img'])
        for b, id, img in zip(range(batch_size), batch['base_img_idx'], batch['base_img']):
            plt.subplot(math.ceil(batch_size**0.5), math.ceil(batch_size**0.5), b + 1)
            plt.title(str(int(id)))
            plt.imshow(img.cpu())
        plt.show()
        
        for b, id, img in zip(range(batch_size * 3), batch['aug_img_idx'], batch['aug_imgs']):
            plt.subplot(math.ceil(
                (batch_size * 3)**0.5),
                math.ceil((batch_size * 3)**0.5),
                b + 1)
            plt.title(str(int(id)))
            plt.imshow(img.cpu())
        plt.show()
        
        input('Press Enter to show next')


if __name__ == '__main__':
    fire.Fire({
        'debug': debug,
        'train': train,
        'train_dali': train_dali,
        'train_baseline': train_baseline,
        'vis': vis_corr,
        'vis_batch': vis_batch,
    })