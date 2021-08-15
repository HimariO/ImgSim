import os
from PIL import Image

import numpy
import torch
import timm

from dino.folder import ImageFolder


def view_model():
    print('>> pretrained')
    for i, name in enumerate(timm.list_models(pretrained=True)):
        print(f"[{i}] {name}")

    print('>> not pretrained')
    for i, name in enumerate(timm.list_models(pretrained=False)):
        print(f"[{i}] {name}")


    m = timm.create_model(
        'tf_efficientnetv2_l',
        pretrained=True,
        num_classes=0,
        global_pool=''
    )
    # m = timm.create_model('hrnet_w32', pretrained=True)
    print(m)
    print(m(torch.zeros([1, 3, 320, 320])).shape)


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