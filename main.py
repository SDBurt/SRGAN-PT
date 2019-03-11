# Libraries
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import argparse as ap
from glob import glob

from tqdm import trange
from PIL import Image

# From files
from config import get_config

def import_images(args):

    path = args.path + '*.jpg' 
    print("Collecting images from: {}".format(path))
    image_list = []
    for filename in glob(path): #assuming gif
        im = np.array(Image.open(filename))
        image_list.append(im)
    return np.array(image_list)

def main(images, cfg):
    print("Number of images from input: {}".format(images.shape[0]))



if __name__ == "__main__":


    cfg = get_config()
    images = import_images(cfg)

    

    main(images, cfg)

