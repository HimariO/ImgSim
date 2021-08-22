import glob
import time

import fire
import numpy as np
import torch
import pytorch_lightning as pl


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
    }
    layer_2_channels = {
        'layer1': 256,
        'layer2': 512,
        'layer3': 1024,
        'layer4': 2048,
    }
    args['dim_feedforward'] = layer_2_channels[args['layer']]
    cotr = LitCOTR(args, args)
    return cotr

p_aug = PairAug(n_deriv=3, output_size=[320, 320])
lit_img = folder.LitImgFolder(
    '/home/ron/Downloads/fb-isc/query',
    p_aug,
    batch_size=32,
    num_worker=20)

# A = time.time()
# for i, data in enumerate(lit_img.train_dataloader()):
#     D = time.time() - A
#     A = time.time()
#     print(i, f"{D:.5f}", 'base_img: ', data['base_img'].shape, 'aug_imgs: ', data['aug_imgs'].shape)
#     if i > 64: break

model = cotr()
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

trainer.fit(model, datamodule=lit_img)