# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import sys
import os
import platform
import argparse
import time
from collections import OrderedDict, defaultdict

import numpy as np
import h5py
import faiss
from PIL import Image
from pretrainedmodels.models import pnasnet
from loguru import logger

import torch
import torchvision
import torchvision.transforms
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import functional as tvtf

from isc.io import write_hdf5_descriptors
from isc.pnasnet import pnasnet5large
from isc.vision_transformer import VisionTransformer, vit_base
from COTR.models.pl_cotr import LitArcCor, LitCOTR
from COTR.models.misc import NestedTensor


def setup_parser(parser):
    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('feature extraction options')
    aa('--transpose', default=-1, type=int, help="one of the 7 PIL transpose options ")
    aa('--train_pca', default=False, action="store_true", help="run PCA training")
    aa('--pca_file', default="", help="File with PCA descriptors")
    aa('--pca_dim', default=1500, type=int, help="output dimension for PCA")
    aa('--device', default="cuda:0", help='pytroch device')
    aa('--batch_size', default=64, type=int, help="max batch size to use for extraction")
    aa('--num_workers', default=20, type=int, help="nb of dataloader workers")

    group = parser.add_argument_group('model options')
    aa('--model', default='multigrain_resnet50', help="model to use")
    aa('--checkpoint', default='data/multigrain_joint_3B_0.5.pth', help='override default checkpoint')
    aa('--GeM_p', default=7.0, type=float, help="Power used for GeM pooling")
    aa('--scales', default="1.0", help="scale levels")
    aa('--imsize', default=512, type=int, help="max image size at extraction time")

    group = parser.add_argument_group('dataset options')
    aa('--file_list', required=True, help="CSV file with image filenames")
    aa('--image_dir', default="", help="search image files in these directories")
    aa('--n_train_pca', default=10000, type=int, help="nb of training vectors for the PCA")
    aa('--i0', default=0, type=int, help="first image to process")
    aa('--i1', default=-1, type=int, help="last image to process + 1")

    group = parser.add_argument_group('output options')
    aa('--o', default="/tmp/desc.hdf5", help="write trained features to this file")
    return parser


def load_model(name, checkpoint_file):
    if name == "zoo_resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        model.eval()
        return model

    if name == "multigrain_resnet50":
        model = torchvision.models.resnet50(pretrained=False)
        st = torch.load(checkpoint_file)
        state_dict = OrderedDict([
            (name[9:], v)
            for name, v in st["model_state"].items() if name.startswith("features.")
        ])
        model.fc
        model.fc = None
        model.load_state_dict(state_dict)
        model.eval()
        return model

    if name == 'pnasnet':
        model = pnasnet5large(pretrained='imagenet')
        if checkpoint_file:
            st = torch.load(checkpoint_file)

            if 'model_state' in st:
                state_dict = OrderedDict([
                    (name.replace('features.0.base_net.', ''), v)
                    for name, v in st["model_state"].items() if name.startswith("features.")
                ])
            else:
                raise RuntimeError("Wrong checkpoint format!")

            print(f"load states: {len(state_dict)}")
            # model.fc
            # model.fc = None
            model.load_state_dict(state_dict)
        model.eval()
        return model
    
    if name == 'vit':
        model = vit_base(patch_size=16, drop_path_rate=0.1)
        st = torch.load(checkpoint_file)
        if 'student' in st:
            st = OrderedDict([
                (name.replace('module.backbone.', ''), v)
                for name, v in st["student"].items()
            ])
        print(f"load states: {len(st)}")
        model.load_state_dict(st, strict=False)
        model.eval()
        return model
    
    if name == 'cotr':
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
            "lr_backbone": 1e-6,  # NOTE: using this arg will unfreeze backbone, potentialy cause NaN in the training process as CNN feature scale going up.
        }
        layer_2_channels = {
            'layer1': 256,
            'layer2': 512,
            'layer3': 1024,
            'layer4': 2048,
        }
        args['dim_feedforward'] = layer_2_channels[args['layer']]
        model = LitCOTR.load_from_checkpoint(
            checkpoint_file,
            trasnformer_args=args,
            backbone_args=args,
            strict=False)
        logger.info(f"load COTR checkpoint: {checkpoint_file}")
        return model

    assert False



def resnet_activation_map(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x


def pnasnet_activation_map(net: pnasnet.PNASNet5Large, x):
    h = net.features(x)
    h = torch.nn.functional.relu(h)
    return h


def vit_extract_feat(model: VisionTransformer, samples):
    feats = model.get_intermediate_layers(samples, n=1)[0].clone()

    cls_output_token = feats[:, 0, :]  #  [CLS] token
    # GeM with exponent 4 for output patch tokens
    b, h, w, d = len(samples), int(samples.shape[-2] / model.patch_embed.patch_size), int(samples.shape[-1] / model.patch_embed.patch_size), feats.shape[-1]
    feats = feats[:, 1:, :].reshape(b, h, w, d)
    feats = feats.clamp(min=1e-6).permute(0, 3, 1, 2)
    feats = nn.functional.avg_pool2d(feats.pow(4), (h, w)).pow(1. / 4).reshape(b, -1)
    # concatenate [CLS] token and GeM pooled patch tokens
    feats = torch.cat((cls_output_token, feats), dim=1)
    return feats


def cotr_feat(model: LitCOTR, samples):
    
    def _norm_image(x):
        if not type(x) is torch.Tensor:
            x = tvtf.to_tensor(x)
        return tvtf.normalize(x, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    nt = NestedTensor(_norm_image(samples), torch.zeros_like(samples))
    norm_feat_map, norm_embed, _ = model(nt)
    return norm_embed


def gem_npy(x, p=3, eps=1e-6):
    x = np.clip(x, a_min=eps, a_max=np.inf)
    x = x ** p
    x = x.mean(axis=0)
    return x ** (1. / p)


class ImageList(Dataset):

    def __init__(self, image_list, imsize=None, transform=None):
        Dataset.__init__(self)
        self.image_list = image_list
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        x = Image.open(self.image_list[i])
        x = x.convert("RGB")
        if self.imsize is not None:
            x.thumbnail((self.imsize, self.imsize), Image.ANTIALIAS)
        if self.transform is not None:
            x = self.transform(x)
        return x


def main():

    parser = argparse.ArgumentParser()
    parser = setup_parser(parser)
    args = parser.parse_args()
    args.scales = [float(x) for x in args.scales.split(",")]

    print("args=",)
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    print("reading image names from", args.file_list)

    if args.device == "cpu":
        if 'Linux' in platform.platform():
            os.system(
                'echo hardware_image_description: '
                '$( cat /proc/cpuinfo | grep ^"model name" | tail -1 ), '
                '$( cat /proc/cpuinfo | grep ^processor | wc -l ) cores'
            )
        else:
            print("hardware_image_description:", platform.machine(), "nb of threads:", args.nproc)
    else:
        print("hardware_image_description:", torch.cuda.get_device_name(0))

    image_list = [l.strip() for l in open(args.file_list, "r")]

    if args.i1 == -1:
        args.i1 = len(image_list)
    image_list = image_list[args.i0:args.i1]

    # add jpg suffix if there is none
    image_list = [
        fname if "." in fname else fname + ".jpg"
        for fname in image_list
    ]

    # full path name for the image
    image_dir = args.image_dir
    if not image_dir.endswith('/'):
        image_dir += "/"

    image_list = [image_dir + fname for fname in image_list]

    print(f"  found {len(image_list)} images")

    if args.train_pca:
        rs = np.random.RandomState(123)
        image_list = [
            image_list[i]
            for i in rs.choice(len(image_list), size=args.n_train_pca, replace=False)
        ]
        print(f"subsampled {args.n_train_pca} vectors")

    # transform without resizing
    mean, std = [0.485, 0.456, 0.406],[0.229, 0.224, 0.225]

    transforms = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ]

    if args.transpose != -1:
        # transforms.insert(TransposeTransform(args.transpose), 0)
        raise RuntimeError('Not sure what this line will do')

    transforms = torchvision.transforms.Compose(transforms)

    im_dataset = ImageList(image_list, transform=transforms, imsize=args.imsize)

    print("loading model")
    net = load_model(args.model, args.checkpoint)
    net.to(args.device)

    print("computing features")

    t0 = time.time()

    with torch.no_grad():
        if args.batch_size == 1:
            raise RuntimeError('!?')
        else:
            all_desc = [None] * len(im_dataset)
            ndesc = [0]
            buckets = defaultdict(list)

            def handle_bucket(bucket):
                ndesc[0] += len(bucket)
                x = torch.stack([xi for no, xi in bucket])
                x = x.to(args.device)
                
                print(f"ndesc {ndesc[0]} / {len(all_desc)} handle bucket of shape {x.shape}\r", end="", flush=True)
                feats = []
                for s in args.scales:
                    xs = nn.functional.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False)
                    
                    if 'resnet' in args.model:
                        o = resnet_activation_map(net, xs)
                    elif 'vit' == args.model:
                        o = vit_extract_feat(net, xs)
                    elif 'cotr' == args.model:
                        o = cotr_feat(net, xs)
                    else:
                        o = pnasnet_activation_map(net, xs)
                    
                    o = o.cpu().numpy()    # B, C, H, W
                    feats.append(o)

                for i, (no, _) in enumerate(bucket):
                    feats_i = np.vstack([
                        f[i].reshape(f[i].shape[0], -1).T
                        for f in feats
                    ])
                    if args.model in ['resnet', 'pnasnet']:
                        gem = gem_npy(feats_i, p=args.GeM_p)
                    else:
                        # import pdb; pdb.set_trace()
                        gem = np.mean(feats_i, axis=0)
                        # gem = feats_i
                    all_desc[no] = gem

            max_batch_size = args.batch_size

            dataloader = torch.utils.data.DataLoader(
                im_dataset, batch_size=1, shuffle=False,
                num_workers=args.num_workers
            )

            for no, x in enumerate(dataloader):
                x = x[0]  # don't batch
                buckets[x.shape].append((no, x))

                if len(buckets[x.shape]) >= max_batch_size:
                    handle_bucket(buckets[x.shape])
                    del buckets[x.shape]

            for bucket in buckets.values():
                handle_bucket(bucket)

    all_desc = np.vstack(all_desc)

    t1 = time.time()

    print()
    print(f"image_description_time: {(t1 - t0) / len(image_list):.5f} s per image")

    if args.train_pca:
        d = all_desc.shape[1]
        pca = faiss.PCAMatrix(d, args.pca_dim, -0.5)
        print(f"Train PCA {pca.d_in} -> {pca.d_out}")
        pca.train(all_desc)
        print(f"Storing PCA to {args.pca_file}")
        faiss.write_VectorTransform(pca, args.pca_file)
    elif args.pca_file:
        print("Load PCA matrix", args.pca_file)
        pca = faiss.read_VectorTransform(args.pca_file)
        print(f"Apply PCA {pca.d_in} -> {pca.d_out}")
        all_desc = pca.apply_py(all_desc)

    print("normalizing descriptors")
    faiss.normalize_L2(all_desc)

    if not args.train_pca:
        print(f"writing descriptors to {args.o}")
        write_hdf5_descriptors(all_desc, image_list, args.o)


if __name__ == "__main__":
    """

    python baselines/GeM_baseline.py \
         --file_list list_files/train \
         --image_dir ~/Downloads/fb-isc/train \
         --pca_file data/pca_multigrain.vt \
         --n_train_pca 10000 \
         --train_pca

    python baselines/GeM_baseline.py \
    --file_list list_files/subset_1_queries \
    --image_dir  ~/Downloads/fb-isc/query \
    --o data/subset_1_queries_multigrain.hdf5 \
    --pca_file data/pca_multigrain.vt \
    --model pnasnet \
    --checkpoint ./data/pnasnet5large-finetune500.pth

    python baselines/GeM_baseline.py \
    --file_list list_files/subset_1_references \
    --image_dir ~/Downloads/fb-isc/reference \
    --o data/subset_1_references_multigrain.hdf5 \
    --pca_file data/pca_multigrain.vt \
    --model pnasnet \
    --checkpoint ./data/pnasnet5large-finetune500.pth

    python scripts/score_normalization.py \
    --query_descs data/subset_1_queries_multigrain.hdf5 \
    --db_descs data/subset_1_references_multigrain.hdf5 \
    --train_descs data/train_{0..19}_multigrain.hdf5 \
    --factor 2.0 --n 10 \
    --o data/predictions_dev_queries_25k_normalized.csv

    python scripts/compute_metrics.py \
    --query_descs data/subset_1_queries_multigrain.hdf5 \
    --db_descs data/subset_1_references_multigrain.hdf5 \
    --gt_filepath list_files/subset_1_ground_truth.csv \
    --track2 \
    --max_dim 2000



    for i in {0..19}; do
        python baselines/GeM_baseline.py \
            --file_list list_files/train \
            --i0 $((i * 50000)) --i1 $(((i + 1) * 50000)) \
            --image_dir ~/Downloads/fb-isc/train \
            --o data/train_${i}_multigrain.hdf5 \
            --pca_file data/pca_multigrain.vt
    done

    python scripts/score_normalization.py \
        --query_descs data/subset_1_queries_multigrain.hdf5 \
        --db_descs data/subset_1_references_multigrain.hdf5 \
        --train_descs data/train_{0..19}_multigrain.hdf5 \
        --factor 2.0 --n 10 \
        --o data/predictions_dev_queries_25k_normalized.csv
    python scripts/score_normalization.py \
        --query_descs data/dev_queries_multigrain.hdf5 \
        --db_descs data/references_{0..19}_multigrain.hdf5 \
        --train_descs data/res50_train_feat/train_{0..19}_multigrain.hdf5 \
        --factor 2.0 --n 10 \
        --o data/predictions_dev_queries_normalized.csv
    
    python scripts/compute_metrics.py \
        --preds_filepath data/predictions_dev_queries_25k_normalized.csv \
        --gt_filepath list_files/subset_1_ground_truth.csv \
            
        --query_descs data/subset_1_queries_multigrain.hdf5 \
        --db_descs data/subset_1_references_multigrain.hdf5 \
        --track2 \
        --write_predictions res50_submit.csv
    """
    main()



