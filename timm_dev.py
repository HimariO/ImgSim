import torch
import timm

print('>> pretrained')
for i, name in enumerate(timm.list_models(pretrained=True)):
    print(f"[{i}] {name}")

print('>> not pretrained')
for i, name in enumerate(timm.list_models(pretrained=False)):
    print(f"[{i}] {name}")


m = timm.create_model(
    'tf_efficientnetv2_m',
    pretrained=True,
    num_classes=0,
    global_pool=''
)
# m = timm.create_model('hrnet_w32', pretrained=True)
print(m)
print(m(torch.zeros([1, 3, 320, 320])).shape)