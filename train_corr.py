import sys
import glob
import time

import fire
import numpy as np
import torch
import pytorch_lightning as pl
from loguru import logger


from COTR.models.cotr_model import COTR, build
from COTR.models.pl_cotr import LitCOTR
from COTR.options.options import *
from COTR.options.options_utils import *
from dino import folder
from dense_match.pair_aug import PairAug


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

def cotr():
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
    # cotr = LitCOTR(args, args)
    # state = torch.load('../COTR/out/default/checkpoint.pth.tar', map_location='cpu')['model_state_dict']
    # cotr.load_state_dict(state, strict=False)
    cotr = LitCOTR.load_from_checkpoint(
        '/home/ron/Projects/ImgSim/checkpoints/train/lightning_logs/version_26/checkpoints/epoch=1-step=8333.ckpt',
        trasnformer_args=args,
        backbone_args=args)
    return cotr


def debug_dataset(lit_img: folder.LitImgFolder):
    train_loader = lit_img.train_dataloader()
    val_loader = lit_img.val_dataloader()
    print(len(train_loader.dataset))
    print(len(train_loader.dataset.img_list))
    print(len(val_loader.dataset))
    print(len(val_loader.dataset.img_list))

    A = time.time()
    for i, data in enumerate(val_loader):
        D = time.time() - A
        A = time.time()
        print(i, f"{D:.5f}", 'base_img: ', data['base_img'].shape, 'aug_imgs: ', data['aug_imgs'].shape)
        print(data['base_img_idx'])
        print(data['aug_img_idx'])
        if i > 8: sys.exit(0)


with logger.catch():
    p_aug = PairAug(n_deriv=3, output_size=[256, 256])
    lit_img = folder.LitImgFolder(
        '/home/ron/Downloads/fb-isc/train',
        p_aug,
        batch_size=96,
        num_worker=30)
    
    # debug_dataset(lit_img)

    model = cotr()

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
    # trainer = pl.Trainer(
    #     accumulate_grad_batches=1,
    #     val_check_interval=1.0,
    #     checkpoint_callback=False,
    #     callbacks=[],
    #     default_root_dir='checkpoints/overfit',
    #     gpus=1,
    #     precision=16,
    #     max_steps=1000,
    #     overfit_batches=128,
    #     limit_val_batches=0,
    # )

    trainer.fit(model, datamodule=lit_img)