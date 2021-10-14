import os
from pprint import pprint

import cv2
import numpy as np
import torch
import timm
from einops import rearrange
from PIL import Image
from timm.models.resnet import ResNet
from timm.models.efficientnet import EfficientNet, default_cfgs
from timm.data import ImageDataset, create_loader

from dino.folder import ImageFolder


def get_imagenet_labels(filename):
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            labels.append(line.split('\t')[1][:-1])  # split and remove line break.
    return labels


def view_model():
    # print('>> pretrained')
    # for i, name in enumerate(timm.list_models(pretrained=True)):
    #     print(f"[{i}] {name}")

    # print('>> not pretrained')
    # for i, name in enumerate(timm.list_models(pretrained=False)):
    #     print(f"[{i}] {name}")

    labels = get_imagenet_labels('./imagenet1k_labels.txt')
    m = timm.create_model(
        'tf_efficientnetv2_l',
        pretrained=True,
        # checkpoint_path="checkpoints/tf_efficientnetv2_l-d664b728.pth"
        num_classes=0,
        global_pool=''
    )
    # st = torch.load("checkpoints/tf_efficientnetv2_l-d664b728.pth")
    # m.load_state_dict(st)
    m.eval()
    # m = timm.create_model('hrnet_w32', pretrained=True)

    print(m(torch.zeros([1, 3, 320, 320])).shape)
    pprint(default_cfgs['tf_efficientnetv2_l'])

    if False:
        img = np.asarray(Image.open('/home/ron/Pictures/gloden_retriever.jpg').resize([384, 384]))
        # img = np.asarray(Image.open('/home/ron/Pictures/cat.jpg').resize([224, 224]))
        # img = torch.tensor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img = torch.tensor(img)
        img = rearrange(img, 'h w c -> 1 c h w').float() / 128 - 1

        pred = m(img)
        print(pred.shape)
        print(pred.argmax(dim=-1))
        print(pred.max(dim=-1).values)
        print(labels[pred.argmax(dim=-1)[0]])
        # breakpoint()


def dataset_softlabel(img_dir):
    assert os.path.exists(img_dir)
    
    dataset = ImageFolder(img_dir)
    model = timm.create_model(
        'tf_efficientnetv2_l',
        pretrained=True,
        # num_classes=0,
        # global_pool=''
    )
    
    model, test_time_pool = (model, False)
    config = timm.data.resolve_data_config({'bacch_size': 32, 'input_size': 320}, model=model)


    model = model.cuda()

    loader = timm.data.create_loader(
        dataset,
        input_size=config['input_size'],
        batch_size=32,
        use_prefetcher=True,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=8,
        crop_pct=1.0)

    model.eval()

    k = 3
    topk_ids = []
    with torch.no_grad():
        for batch_idx, (input, sample_idx) in enumerate(loader):
            input = input.cuda()
            labels = model(input)


if __name__ == '__main__':
    view_model()