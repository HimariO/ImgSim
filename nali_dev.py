import os
import math
import glob
import time
import itertools
import random
from typing import *
from numpy.random import shuffle

import torch
import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
from torch.utils.data import Dataset
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch.fn import torch_python_function
from torchvision.transforms import functional as tvtf

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from loguru import logger
from einops import rearrange


class DALIPairAug:

    @staticmethod
    def show(images, aug_images, landmarks):
        if hasattr(images, "as_cpu"):
            images = images.as_cpu()
            aug_images = aug_images.as_cpu()
        batch_size = len(images)

        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize = (16,14))
        plt.suptitle(None)
        columns = 3
        rows = int(math.ceil(batch_size / columns))
        gs = gridspec.GridSpec(rows, columns)
        for i in range(batch_size):
            ax = plt.subplot(gs[i])
            plt.axis("off")
            plt.title('')
            img = images.at(i)
            aug_img = aug_images.at(i)
            img = np.concatenate([aug_img, img], axis=1)
            r = 0.002 * max(img.shape[0], img.shape[1])
            for p in landmarks.at(i):
                circle = patches.Circle(p, r, color=(0,1,0,1))
                ax.add_patch(circle)
            plt.imshow(img)

    @staticmethod
    def encoded_images_sizes(jpegs):
        shapes = fn.peek_image_shape(jpegs)  # the shapes are HWC
        h = fn.slice(shapes, 0, 1, axes=[0]) # extract height...
        w = fn.slice(shapes, 1, 1, axes=[0]) # ...and width...
        return fn.cat(w, h)                  # ...and concatenate

    @staticmethod
    def kp_to_target(
            kps: torch.Tensor, src_kps: torch.Tensor,
            img_shape: torch.Tensor, grid_size: torch.Tensor,
            vflip: torch.Tensor, hflip: torch.Tensor,
            grid_cells=None):
        h, w = img_shape[0], img_shape[1]
        grid_cells = img_shape[0] // grid_size if grid_cells is None else grid_cells

        if (vflip > 0).any():
            kps[:, 1] = img_shape[1] - 1 - kps[:, 1]
        if (hflip > 0).any():
            kps[:, 0] = img_shape[0] - 1 - kps[:, 0]

        # print("kps:", kps.shape, src_kps.shape, img_shape, grid_size)
        bound_mask = torch.logical_and(0 <= kps, kps <= img_shape[None, :]).all(dim=-1)
        kps_pair = torch.cat([
            kps / (img_shape[None, :] / grid_size),
            src_kps / (img_shape[None, :] / grid_size)
            ],
            dim=-1)  # (n, 2) -> (n, 4) {aug_x, aug_y, src_x, src_y}
        kps_in = kps_pair[bound_mask]
        kps_in = torch.clip(torch.floor(kps_in).long(), max=grid_cells - 1)

        if grid_cells is None:
            onehot = torch.zeros([
                h // grid_size, w // grid_size,
                h // grid_size, w // grid_size], dtype=torch.float32)
        else:
            onehot = torch.zeros([grid_cells, grid_cells, grid_cells, grid_cells], dtype=torch.float32)
            
        onehot[kps_in[:, 3], kps_in[:, 2], kps_in[:, 1], kps_in[:, 0]] = 1
        
        mask = torch.normal(mean=torch.zeros_like(onehot), std=1 - onehot)
        mask = (mask > 1.5).float()  # NOTE: keep around 10% of negative sample
        mask += onehot
        return onehot, mask

    @staticmethod
    def random_transform(mt, p=0.5):
        coin = fn.random.coin_flip(probability=p)
        mtid = np.float32([
                [1, 0, 0],
                [0, 1, 0],])
        return mt * coin + (1 - coin) * mtid

    @staticmethod
    def random_choise(image_pool):
        n = np.array(len(image_pool))
        p = np.array(1 / n)
        rand = fn.random.uniform(range=(0., 1.))
        final = [
            image_pool[i] * (
                (fn.constant(fdata=i * p) <= rand) & (rand < fn.constant(fdata=(i + 1) * p))
            )
            for i in range(n)]
        return sum(final)

    @staticmethod
    def random_apply(img, aug_img, p=0.5):
        coin = fn.random.coin_flip(probability=p)
        return aug_img * coin + (1 - coin) * img

    @staticmethod
    def build_pipeline(src, batch_size=64, output_size=[256, 256], grid_size=16, n_deriv=3, debug=False):
        assert all(o % grid_size == 0 for o in output_size)
        assert batch_size % (n_deriv + 1) == 0
        
        pipe = dali.pipeline.Pipeline(
            batch_size=batch_size, num_threads=16, device_id=1,
            exec_pipelined=False, exec_async=False)
        
        with pipe:
            # jpegs, _ = fn.readers.file(file_root=root_dir, files=image_files)
            jpegs, indice, drive_indice = fn.external_source(source=src, num_outputs=3)
            indice = fn.cast(indice, dtype=dali.types.INT64)
            images = fn.decoders.image(jpegs, device="mixed")

            size = DALIPairAug.encoded_images_sizes(jpegs)
            center = np.array(output_size) / 2

            coord = [
                [grid_size * (i + .5), grid_size * (j + .5)]
                for i, j in itertools.product(
                    range(0, output_size[0] // grid_size),
                    range(0, output_size[1] // grid_size))]
            src_keypoints = np.array(coord, dtype=np.int32)
            
            # mt = fn.transforms.combine(tr1, rot, np.float32([[1, 1, 0],
            #                                            [0, 1, 0]]), tr2)
            rcrop = fn.random_resized_crop(images, size=output_size, random_aspect_ratio=[0.8, 1.2], random_area=[0.2, 1.0])
            
            aug_imgs = []
            aug_kps = []
            targets = []
            targets_mask =[]
            for _ in range(n_deriv):
                tr1 = fn.transforms.translation(offset=-center)
                tr2 = fn.transforms.translation(offset=center)
                rot = fn.transforms.rotation(angle=fn.random.uniform(range=(-45, 45)))
                shr = fn.transforms.shear(shear=fn.random.uniform(range=(-1, 1), shape=[2]))
                rand_tr = fn.transforms.translation(offset=
                    fn.random.uniform(
                        range=(-max(output_size)/5, max(output_size)/5),
                        shape=[2])
                )
                mt = fn.transforms.combine(
                    tr1,
                    DALIPairAug.random_transform(rot, p=0.5),
                    DALIPairAug.random_transform(shr, p=0.5),
                    tr2,
                    DALIPairAug.random_transform(rand_tr, p=0.5)
                )  # 2x3 affine mtx
                
                aug_image = fn.warp_affine(rcrop, matrix = mt, fill_value=0, inverse_map=False)
                keypoints = fn.coord_transform(src_keypoints, MT = mt)

                hflip = fn.random.coin_flip(probability=0.5)
                vflip = fn.random.coin_flip(probability=0.5)
                aug_image = fn.flip(
                    aug_image,
                    horizontal=hflip,
                    vertical=vflip)
                aug_image = fn.brightness_contrast(
                    aug_image,
                    brightness=fn.random.uniform(range=(0.4, 1.0)),
                    contrast=fn.random.uniform(range=(0.8, 1.0))
                )
                grayscale = fn.color_space_conversion(aug_image, image_type=dali.types.RGB, output_type=dali.types.GRAY)
                grayscale = fn.cat(grayscale, grayscale, grayscale, axis=2)
                aug_image = DALIPairAug.random_apply(
                    aug_image,
                    DALIPairAug.random_choise([
                        fn.color_space_conversion(aug_image, image_type=dali.types.RGB, output_type=dali.types.BGR),
                        fn.color_space_conversion(aug_image, image_type=dali.types.RGB, output_type=dali.types.YCbCr),
                        grayscale,
                    ]),
                    p=0.3,
                )
                
                aug_imgs.append(aug_image)
                aug_kps.append(keypoints)

                onehot, mask = torch_python_function(
                    keypoints, src_keypoints, output_size, grid_size, vflip, hflip,
                    function=DALIPairAug.kp_to_target, batch_processing=False, num_outputs=2)
                targets.append(onehot)
                targets_mask.append(mask)
            
            if debug:
                pipe.set_outputs(aug_imgs[0], rcrop, aug_kps[0], indice, targets[0], drive_indice)
            else:
                aug_imgs = fn.stack(*aug_imgs)
                aug_kps = fn.stack(*aug_kps)
                targets = fn.stack(*targets)
                targets_mask = fn.stack(*targets_mask)
                pipe.set_outputs(aug_imgs, rcrop, aug_kps, indice, drive_indice, targets, targets_mask)

        pipe.build()

        if debug:    
            images, croped, keypoints, _, target, _ = pipe.run()
            print(images, keypoints)
            DALIPairAug.show(croped, images, keypoints)
            plt.show()

            # show(croped, keypoints)
            # plt.show()
        return pipe
    
    @staticmethod
    def build_simple_pipeline(src, batch_size=64, output_size=[256, 256], grid_size=16, n_deriv=3, debug=False):
        assert all(o % grid_size == 0 for o in output_size)
        assert batch_size % (n_deriv + 1) == 0
        
        pipe = dali.pipeline.Pipeline(
            batch_size=batch_size, num_threads=16, device_id=1,
            exec_pipelined=False, exec_async=False)
        
        with pipe:
            # jpegs, _ = fn.readers.file(file_root=root_dir, files=image_files)
            jpegs, indice, drive_indice = fn.external_source(source=src, num_outputs=3)
            indice = fn.cast(indice, dtype=dali.types.INT64)
            drive_indice = fn.cast(drive_indice, dtype=dali.types.INT64)
            images = fn.decoders.image(jpegs, device="mixed")

            size = DALIPairAug.encoded_images_sizes(jpegs)
            center = np.array(output_size) / 2

            coord = [
                [grid_size * (i + .5), grid_size * (j + .5)]
                for i, j in itertools.product(
                    range(0, output_size[0] // grid_size),
                    range(0, output_size[1] // grid_size))]
            src_keypoints = np.array(coord, dtype=np.int32)
            
            # mt = fn.transforms.combine(tr1, rot, np.float32([[1, 1, 0],
            #                                            [0, 1, 0]]), tr2)
            rcrop = fn.random_resized_crop(images, size=output_size, random_aspect_ratio=[0.8, 1.2], random_area=[0.5, 1.0])
            
            aug_imgs = []
            aug_kps = []
            targets = []
            targets_mask =[]
            for _ in range(n_deriv):
                tr1 = fn.transforms.translation(offset=-center)
                tr2 = fn.transforms.translation(offset=center)
                rot = fn.transforms.rotation(angle=fn.random.uniform(range=(-20, 20)))
                rand_tr = fn.transforms.translation(offset=
                    fn.random.uniform(
                        range=(-max(output_size)/5, max(output_size)/5),
                        shape=[2])
                )
                mt = fn.transforms.combine(
                    tr1,
                    DALIPairAug.random_transform(rot, p=0.5),
                    tr2,
                    DALIPairAug.random_transform(rand_tr, p=0.5),
                )  # 2x3 affine mtx
                
                aug_image = fn.warp_affine(rcrop, matrix = mt, fill_value=0, inverse_map=False)
                keypoints = fn.coord_transform(src_keypoints, MT = mt)

                hflip = fn.random.coin_flip(probability=0.5)
                aug_image = fn.brightness_contrast(
                    aug_image,
                    brightness=fn.random.uniform(range=(0.4, 1.2)),
                    contrast=fn.random.uniform(range=(0.8, 1.2))
                )
                grayscale = fn.color_space_conversion(aug_image, image_type=dali.types.RGB, output_type=dali.types.GRAY)
                grayscale = fn.cat(grayscale, grayscale, grayscale, axis=2)
                aug_image = DALIPairAug.random_apply(
                    aug_image,
                    DALIPairAug.random_choise([
                        fn.color_space_conversion(aug_image, image_type=dali.types.RGB, output_type=dali.types.BGR),
                        # fn.color_space_conversion(aug_image, image_type=dali.types.RGB, output_type=dali.types.YCbCr),
                        grayscale,
                    ]),
                    p=0.3,
                )
                
                aug_imgs.append(aug_image)
                aug_kps.append(keypoints)

                onehot, mask = torch_python_function(
                    keypoints, src_keypoints, output_size, grid_size, 0, hflip,
                    function=DALIPairAug.kp_to_target, batch_processing=False, num_outputs=2)
                targets.append(onehot)
                targets_mask.append(mask)
            
            if debug:
                pipe.set_outputs(aug_imgs[0], rcrop, aug_kps[0], indice, targets[0], drive_indice)
            else:
                aug_imgs = fn.stack(*aug_imgs)
                aug_kps = fn.stack(*aug_kps)
                targets = fn.stack(*targets)
                targets_mask = fn.stack(*targets_mask)
                pipe.set_outputs(aug_imgs, rcrop, aug_kps, indice, drive_indice, targets, targets_mask)

        pipe.build()

        if debug:    
            images, croped, keypoints, _, target, _ = pipe.run()
            print(images, keypoints)
            DALIPairAug.show(croped, images, keypoints)
            plt.show()
        return pipe




class ExternalIterator(object):
    
    def __init__(self, root_or_files: Union[str, List[str]], batch_size, 
                n_drive=3, max_iter=None, shuffle=True, debug=False):
        self.batch_size = batch_size
        self.img_paths = (
            sorted(glob.glob(os.path.join(root_or_files, '*.jpg')))
            if not type(root_or_files) is list
            else root_or_files )
        self.indices = list(range(len(self.img_paths)))
        self.n_drive = n_drive
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.debug = debug
        if shuffle: random.shuffle(self.indices)
        logger.info(f"[ExternalIterator] imgs: {len(self.img_paths)}, batch_size: {batch_size}, shuffle: {shuffle}")
    
    def __len__(self):
        return (
            len(self.img_paths)
            if self.max_iter is None 
            else min(self.max_iter, len(self.img_paths))
        )

    def __iter__(self):
        self.i = 0
        self.n = len(self)
        if self.shuffle: random.shuffle(self.indices)
        return self

    def __next__(self):
        batch = []
        sample_idx = []
        drive_idx = []
        img_paths = []
        for _ in range(self.batch_size):
            index = self.indices[self.i]
            with open(self.img_paths[index], mode='rb') as f:
                batch.append(np.frombuffer(f.read(), dtype = np.uint8))
            sample_idx.append(np.array([index], dtype=np.int64))
            drive_idx.append(np.array([index for _ in range(self.n_drive)], dtype=np.int64))
            
            self.i = (self.i + 1) 
            if self.i >= len(self.img_paths): random.shuffle(self.indices)
            self.i %= len(self.img_paths)

            if self.debug:
                img_paths.append(self.img_paths[index])
        if self.debug:
            return (batch, sample_idx, drive_idx, img_paths)
        else:
            return (batch, sample_idx, drive_idx)


class PairAugDaliIterator(DALIGenericIterator):

    def __init__(self, *args, norm_img=True, unit_norm=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_img = norm_img
        self.unit_norm = unit_norm

    def _norm_image(self, x):
        if not type(x) is torch.Tensor:
            x = tvtf.to_tensor(x)
        if self.unit_norm:
            # NOTE map 0~255 to -1~1, used for efficientnet v2
            if x.max() > 1 or x.dtype == torch.uint8:
                return (x.float() -128) / 128
            else:
                return (x - 0.5) * 2
        return tvtf.normalize(x, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __next__(self):
        gpus_batch_data = super().__next__ ()
        _gpus_batch_data = []
        for i, batch_data in enumerate(gpus_batch_data):
            _batch_data = {}
            for k, v in batch_data.items():
                if k == 'target_corrs':
                    _batch_data[k] = rearrange(v, 'b n h1 w1 h2 w2->(b n) h1 w1 h2 w2')
                elif k == 'target_mask':
                    _batch_data[k] = rearrange(v, 'b n h1 w1 h2 w2->(b n) h1 w1 h2 w2')
                elif k == 'base_img':
                    if self.norm_img:
                        _batch_data['base_img'] = rearrange(v.float() / 255, 'b h w c->b c h w')
                        _batch_data['base_img'] = self._norm_image(_batch_data['base_img'])
                    else:
                        _batch_data[k] = v
                elif k == 'aug_imgs':
                    if self.norm_img:
                        _batch_data['aug_imgs'] = rearrange(v.float() / 255, 'b n h w c->(b n) c h w')
                        _batch_data['aug_imgs'] = self._norm_image(_batch_data['aug_imgs'])
                    else:
                        _batch_data['aug_imgs'] = rearrange(v, 'b n h w c->(b n) h w c')
                elif k == 'aug_img_idx' or k == 'base_img_idx':
                    _batch_data[k] = rearrange(v, 'b n->(b n)')
                else:
                    _batch_data[k] = v
            _gpus_batch_data.append(_batch_data)
        return _gpus_batch_data[0]
    
    def __len__(self):
        batch_count = self._size // (self._num_gpus * self.batch_size)
        # last_batch = 1 if self._fill_last_batch else 1
        return batch_count



class CacheDataset(Dataset):

    def __init__(self, cache_file):
        self.cache = torch.load(cache_file)
    
    def __getitem__(self, index):
        return self.cache[index]
    
    def __len__(self):
        return len(self.cache)
    
    @staticmethod
    def debatch(batch_items):
        return batch_items[0]


def build_pairaug_iterator(root_dir_files: Union[str, List[str]], batch_size=64, output_size=[256, 256],
                            grid_size=16, n_deriv=3, max_iter=None, easy=False, unit_norm=False) -> PairAugDaliIterator:
    eit = ExternalIterator(root_dir_files, batch_size // (1 + n_deriv), max_iter=max_iter)
    if not easy:
        pipe = DALIPairAug.build_pipeline(
            eit,
            batch_size=batch_size // (1 + n_deriv),
            output_size=output_size,
            grid_size=grid_size,
            n_deriv=n_deriv,)
    else:
        pipe = DALIPairAug.build_simple_pipeline(
            eit,
            batch_size=batch_size // (1 + n_deriv),
            output_size=output_size,
            grid_size=grid_size,
            n_deriv=n_deriv)
    iter = PairAugDaliIterator(
        pipe, output_map=[
            'aug_imgs',
            'base_img',
            'aug_kps',
            'base_img_idx',
            'aug_img_idx',
            'target_corrs',
            'target_mask'
        ],
        size=len(eit),
        auto_reset=True,
        unit_norm=unit_norm,
        fill_last_batch=False)
    return iter


def inspect_dataset_iter():
    from visual_util import CorrVis

    # dali_extra_dir = os.environ["DALI_EXTRA_PATH"]
    # root_dir = os.path.join(dali_extra_dir, 'db', 'face_landmark')
    root_dir = "/home/ron/Downloads/fb-isc/query"

    # images are in JPEG format
    image_files = glob.glob(os.path.join(root_dir, '*.jpg'))
    image_files = image_files[:16]
    # keypoints are in NumPy files
    keypoint_files = ["{}.npy".format(i) for i in range(6)]

    with logger.catch():
        eit = ExternalIterator(image_files, 16 // 4)
        pipe = DALIPairAug.build_simple_pipeline(eit, batch_size=16 // 4)

        dataset_size = 2048*1
        iter = PairAugDaliIterator(
            pipe, output_map=[
                'aug_imgs',
                'base_img',
                'aug_kps',
                'base_img_idx',
                'aug_img_idx',
                'target_corrs',
                'target_mask'
            ],
            size=dataset_size, auto_reset=True, fill_last_batch=False, norm_img=False)
        
        vis = CorrVis(16)
        
        A = time.time()
        for i, batch in enumerate(iter):
            # batch[0]['images'].shape
            print(i)
            print(type(batch['base_img']), batch['base_img'].shape, batch['base_img'].dtype)
            print(type(batch['aug_imgs']), batch['aug_imgs'].shape, batch['aug_imgs'].dtype)
            print(type(batch['target_corrs']), batch['target_corrs'].shape, batch['target_corrs'].dtype)
            print(batch['base_img_idx'])
            print(batch['aug_img_idx'])

            if True:
                for j, aug_img in enumerate(batch['aug_imgs'].cpu().numpy()):
                    base = batch['base_img'][j // 3].cpu()
                    corr = batch['target_corrs'][j].cpu()
                    vis.show_corr_mapping(corr, base, aug_img)
                    input('Press Enter to show next')
                # plt.imshow(batch['images'][0].cpu())
                # plt.show()
                # if i > 2: break
        D = time.time() - A
        speed = D/(i+1)
        print(f"{D} / {(i+1)} = {speed:.6f}")


def inspect_image_class():
    from visual_util import CorrVis

    # dali_extra_dir = os.environ["DALI_EXTRA_PATH"]
    # root_dir = os.path.join(dali_extra_dir, 'db', 'face_landmark')
    root_dir = "/home/ron/Downloads/fb-isc/query"

    # images are in JPEG format
    image_files = glob.glob(os.path.join(root_dir, '*.jpg'))
    image_files = image_files[:16]
    # keypoints are in NumPy files
    keypoint_files = ["{}.npy".format(i) for i in range(6)]

    with logger.catch():
        eit = ExternalIterator(image_files, 16)
        pipe = DALIPairAug.build_simple_pipeline(eit, batch_size=16)

        dataset_size = 2048*1
        iter = PairAugDaliIterator(
            pipe, output_map=[
                'aug_imgs',
                'base_img',
                'aug_kps',
                'base_img_idx',
                'aug_img_idx',
                'target_corrs',
                'target_mask'
            ],
            size=dataset_size, auto_reset=True, fill_last_batch=False, norm_img=False)
        
        vis = CorrVis(16)
        A = time.time()
        
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
        
        D = time.time() - A
        speed = D/(i+1)
        print(f"{D} / {(i+1)} = {speed:.6f}")


def debug_iter_consistancy():
    with logger.catch():
        record = {}
        
        for e in range(3):
            eit = ExternalIterator("/home/ron/Downloads/fb-isc/query", 64 // 4, shuffle=True, debug=True)
            record[e] = {}
            for i, data in zip(range(len(eit)), iter(eit)):
                batch, sample_idx, drive_idx, img_paths = data
                for j, p in zip(sample_idx, img_paths):
                    record[e][int(j)] = p
        
        for _ in range(10):
            rand_sample = random.randint(0, len(record[0]) - 1)
            print(f"sample - {rand_sample}")
            for e in range(3):
                print(record[e][rand_sample])
        breakpoint()
        print('a')


def cache_iter_output():
    root_dir = "/home/ron/Downloads/fb-isc/query"

    # images are in JPEG format
    image_files = glob.glob(os.path.join(root_dir, '*.jpg'))
    image_files = image_files[:1000]

    with logger.catch():
        eit = ExternalIterator(image_files, 8)
        pipe = DALIPairAug.build_simple_pipeline(eit, batch_size=8)

        dataset_size = 2048*1
        iter = PairAugDaliIterator(
            pipe, output_map=[
                'aug_imgs',
                'base_img',
                'aug_kps',
                'base_img_idx',
                'aug_img_idx',
                'target_corrs',
                'target_mask'
            ],
            size=dataset_size, auto_reset=True, fill_last_batch=False, norm_img=True)
        
        cache = {}
        for i, batch in zip(range(len(iter)), iter):
            print(i)
            cache[i] = batch
        torch.save(cache, 'dali_cache.pth')


if __name__ == '__main__':
    # cache_iter_output()
    inspect_image_class()