import os
import math
import glob
import itertools

import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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


pipe = dali.pipeline.Pipeline(batch_size=6, num_threads=3, device_id=1)
with pipe:
    jpegs, _ = fn.readers.file(file_root=root_dir, files=image_files)
    images = fn.decoders.image(jpegs, device="mixed")

    size = encoded_images_sizes(jpegs)
    center = size / 2

    coord = [[20 * i, 20 * j] for i, j in itertools.product(range(1, 10), range(1, 10))]
    keypoints = np.array(coord, dtype=np.int32)
    tr1 = fn.transforms.translation(offset=-center)
    tr2 = fn.transforms.translation(offset=center)
    rot = fn.transforms.rotation(angle=fn.random.uniform(range=(-90, 90)))
    mt = fn.transforms.combine(tr1, rot, tr2)
    # mt = fn.transforms.combine(tr1, rot, np.float32([[1, 1, 0],
    #                                            [0, 1, 0]]), tr2)
    
    rcrop = fn.random_resized_crop(images, random_aspect_ratio=[0.8, 1.2], random_area=[0.2, 1.0])
    images = fn.warp_affine(images, matrix = mt, fill_value=0, inverse_map=False)
    keypoints = fn.coord_transform(keypoints, MT = mt)
    pipe.set_outputs(images, keypoints)

pipe.build()
images, keypoints = pipe.run()
print(images, keypoints)
show(images, keypoints)
plt.show()