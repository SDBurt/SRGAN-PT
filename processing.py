import torch as tr
import torchvision as tv
import PIL
from PIL import Image, ImageFilter
from config import get_config

cfg = get_config()

# From https://forums.fast.ai/t/image-normalization-in-pytorch/7534/3, Ramesh (nov 17th)
normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# From https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3 by Kaican Li
reverse_normalize = tv.transforms.Compose([
    tv.transforms.Normalize(mean=[-2.118, -2.036, -1.804], std=[4.367, 4.464, 4.444])
])

randomcrop = tv.transforms.Compose([
    tv.transforms.RandomCrop(96),
    tv.transforms.ToTensor()
])

def downsample(img):
    res = to_pil(img)
    res = res.filter(ImageFilter.GaussianBlur(radius=5))
    res = scale(res)
    return res

to_pil = tv.transforms.ToPILImage()

scale = tv.transforms.Compose([
    tv.transforms.Resize(24, Image.BICUBIC),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

def get_dataset(path):
    print(f'Loading images from: {path}')
    return tv.datasets.ImageFolder(root=path, transform=randomcrop)