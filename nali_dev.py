import os
import math
import glob
import time
import itertools
import random

import torch
import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch.fn import torch_python_function

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from loguru import logger

# dali_extra_dir = os.environ["DALI_EXTRA_PATH"]
# root_dir = os.path.join(dali_extra_dir, 'db', 'face_landmark')
root_dir = "/home/ron/Downloads/fb-isc/query"

# images are in JPEG format
image_files = glob.glob(os.path.join(root_dir, '*.jpg'))
# keypoints are in NumPy files
keypoint_files = ["{}.npy".format(i) for i in range(6)]

def show(images, landmarks):
    if hasattr(images, "as_cpu"):
        images = images.as_cpu()
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
        r = 0.002 * max(img.shape[0], img.shape[1])
        for p in landmarks.at(i):
            circle = patches.Circle(p, r, color=(0,1,0,1))
            ax.add_patch(circle)
        plt.imshow(img)


def encoded_images_sizes(jpegs):
    shapes = fn.peek_image_shape(jpegs)  # the shapes are HWC
    h = fn.slice(shapes, 0, 1, axes=[0]) # extract height...
    w = fn.slice(shapes, 1, 1, axes=[0]) # ...and width...
    return fn.cat(w, h)                  # ...and concatenate


def kp_to_target(kps: torch.Tensor, src_kps: torch.Tensor, img_shape, grid_size):
    h, w = img_shape
    # print("kps:", kps.shape, src_kps.shape, img_shape, grid_size)
    bound_mask = torch.logical_and(0 <= kps, kps <= img_shape[None, :]).all(dim=-1)
    kps_pair = torch.cat([
        kps / img_shape[None, :],
        src_kps / img_shape[None, :]],
        dim=-1)  # (n, 2) -> (n, 4)
    # breakpoint()
    kps_in = kps_pair[bound_mask]
    kps_in = torch.round(kps_in).long()

    onehot = torch.zeros([
        h // grid_size, w // grid_size,
        h // grid_size, w // grid_size], dtype=torch.float32)
    onehot[kps_in[:, 2], kps_in[:, 3], kps_in[:, 0], kps_in[:, 1]] = 1
    
    mask = torch.normal(mean=torch.zeros_like(onehot), std=1 - onehot)
    mask = (mask > 1.5).float()  # NOTE: keep around 10% of negative sample
    mask += onehot
    return onehot, mask


def random_transform(mt, p=0.5):
    coin = fn.random.coin_flip(probability=p)
    mtid = np.float32([
            [1, 0, 0],
            [0, 1, 0],])
    return mt * coin + (1 - coin) * mtid


def pipeline_example(src, batch_size=64, output_size=[256, 256], grid_size=16):
    assert all(o % grid_size == 0 for o in output_size)
    pipe = dali.pipeline.Pipeline(
        batch_size=batch_size, num_threads=16, device_id=1,
        exec_pipelined=False, exec_async=False)
    
    with pipe:
        # jpegs, _ = fn.readers.file(file_root=root_dir, files=image_files)
        jpegs, indice = fn.external_source(source=src, num_outputs=2)
        indice = fn.cast(indice, dtype=dali.types.INT64)
        images = fn.decoders.image(jpegs, device="mixed")

        size = encoded_images_sizes(jpegs)
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
        for _ in range(3):
            tr1 = fn.transforms.translation(offset=-center)
            tr2 = fn.transforms.translation(offset=center)
            rot = fn.transforms.rotation(angle=fn.random.uniform(range=(-90, 90)))
            rand_tr = fn.transforms.translation(offset=
                fn.random.uniform(
                    range=(-max(output_size)/5, max(output_size)/5),
                    shape=[2])
            )
            mt = fn.transforms.combine(
                tr1,
                random_transform(rot, p=0.5),
                tr2,
                random_transform(rand_tr, p=0.5)
            )  # 2x3 affine mtx
            
            # breakpoint()
            
            aug_image = fn.warp_affine(rcrop, matrix = mt, fill_value=0, inverse_map=False)
            keypoints = fn.coord_transform(src_keypoints, MT = mt)
            aug_imgs.append(aug_image)
            aug_kps.append(keypoints)

            onehot, mask = torch_python_function(
                keypoints, src_keypoints, output_size, grid_size,
                function=kp_to_target, batch_processing=False, num_outputs=2)
            targets.append(onehot)
            targets_mask.append(mask)
        
        pipe.set_outputs(aug_imgs[0], rcrop, aug_kps[0], indice, targets[0], mt)
        # aug_imgs = fn.stack(*aug_imgs)
        # aug_kps = fn.stack(*aug_kps)
        # pipe.set_outputs(aug_imgs, rcrop, aug_kps, indice)

    pipe.build()
    
    images, croped, keypoints, _, target, mt = pipe.run()
    print(images, keypoints, mt.at(0).shape)
    show(images, keypoints)
    plt.show()

    # show(croped, keypoints)
    # plt.show()
    return pipe




class ExternalIterator(object):
    
    def __init__(self, root, batch_size):
        self.batch_size = batch_size
        self.img_paths = glob.glob(os.path.join(root, '*.jpg'))
        self.indices = list(range(len(self.img_paths)))
        random.shuffle(self.indices)

    def __iter__(self):
        self.i = 0
        self.n = len(self.img_paths)
        return self

    def __next__(self):
        batch = []
        sample_idx = []
        for _ in range(self.batch_size):
            index = self.indices[self.i]
            with open(self.img_paths[index], mode='rb') as f:
                batch.append(np.frombuffer(f.read(), dtype = np.uint8))
            sample_idx.append(np.array([index], dtype = np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch, sample_idx)


class ImagenetteValPipeline(dali.pipeline.Pipeline):
    
    def __init__(self, source, batch_size=32, num_threads=8, device_id=1):
        super(ImagenetteValPipeline, self).__init__(batch_size, num_threads, device_id, seed=42)
        
        self.input = ops.ExternalSource(source=source)
        self.decode = ops.nvJPEGDecoder(device='mixed', output_type=types.RGB)
        # Not possible to center crop with DALI, so I use the entire image.
        self.resize = ops.RandomResizedCrop(size=[320, 320], random_aspect_ratio=[0.8, 1.2], random_area=[0.2, 1.0])
        # Convert tensor format from NHWC to NCHW and normalize


        
    def define_graph(self):
        self.jpegs, self.index = self.input()
        images = self.decode(self.jpegs)
        images = self.resize(images)
        # images = self.normperm(images)
        return (images, self.index)


if __name__ == '__main__':
    with logger.catch():
        eit = ExternalIterator(root_dir, 64 // 4)
        pipe = pipeline_example(eit, batch_size=64 // 4)

        # dataset_size = 2048*8
        # iter = DALIGenericIterator(
        #     pipe, output_map=['images', 'crops', 'keypoints', 'sample_idx'],
        #     size=dataset_size, auto_reset=True, fill_last_batch=False)
        
        # A = time.time()
        # for i, batch in enumerate(iter):
        #     # batch[0]['images'].shape
        #     print(i)
        #     print(type(batch[0]['images']), batch[0]['images'].shape, batch[0]['images'].dtype)
        #     # plt.imshow(batch[0]['images'][0].cpu())
        #     # plt.show()
        #     # if i > 2: break
        # D = time.time() - A
        # speed = D/(i+1)
        # print(f"{D} / {(i+1)} = {speed:.6f}")