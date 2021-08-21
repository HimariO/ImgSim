import glob

import fire
import numpy as np
import torch
import pytorch_lightning as pl


from COTR.models.cotr_model import COTR, build
from COTR.options.options import *
from COTR.options.options_utils import *

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
build(opt)