import torchvision as tv
from PIL import Image, ImageFilter
from config import get_config

cfg = get_config()

tt = tv.transforms.ToTensor()

pil = tv.transforms.ToPILImage()

# From https://forums.fast.ai/t/image-normalization-in-pytorch/7534/3, Ramesh (nov 17th)
normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# From https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3 by Kaican Li
reverse_normalize = tv.transforms.Compose([
    tv.transforms.Normalize(mean=[-2.118, -2.036, -1.804], std=[4.367, 4.464, 4.444])
])


# ToTensor() normalizes image between [0,1]
randomcrop = tv.transforms.Compose([
    tv.transforms.RandomCrop(cfg.cropsize),
    tt
])


def downsample(img):
    '''downsample the input image by applying a gaussian blur followed by scaling'''
    res = pil(img)
    res = res.filter(ImageFilter.GaussianBlur(radius=1))
    res = scale(res)
    return res

    

# Remove normalize if you want data to be between [0,1]
# Update logger and train()
scale = tv.transforms.Compose([
    tv.transforms.Resize(cfg.cropsize//cfg.factor, Image.BICUBIC),
    tt,
    normalize,
])

def get_dataset(path):
    '''Use torchvision Imagefolder to receive and randomly crop all input images'''
    if not path.endswith("/"):
        path = path + "/"
    print(f'Loading images from: {path}')
    return tv.datasets.ImageFolder(root=path, transform=randomcrop)