import torch as tr
import torchvision as tv
from config import get_config

cfg = get_config()

# From https://forums.fast.ai/t/image-normalization-in-pytorch/7534/3, Ramesh (nov 17th)
normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# From https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3 by Kaican Li
reverse_normalize = tv.transforms.Compose([
    tv.transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])
])

randomcrop = tv.transforms.Compose([
    tv.transforms.RandomCrop(96),
    tv.transforms.ToTensor()
])

scale = tv.transforms.Compose([
    tv.transforms.ToPILImage(),
    tv.transforms.Resize(24),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

def get_dataset(path):
    return tv.datasets.ImageFolder(root=path, transform=randomcrop)