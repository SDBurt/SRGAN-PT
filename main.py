import torch as tr
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from pathlib import Path
from datetime import datetime
from tensorboardX import SummaryWriter
from config import get_config
from models import Generator, Discriminator


device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
cfg = get_config()

class SRGAN(object):

    def __init__(self, cfg):
        super(SRGAN, self).__init__()

        self.generator = Generator(cfg).to(device)
        self.discriminator = Discriminator(cfg).to(device)
    
    def preprocessing(self):
        data = tv.datasets.ImageFolder(root='/path/to/your/data/trn', transform=generic_transform)


def main():
    cfg = get_config()
    srgan = SRGAN(cfg)

if __name__ == "__main__":
    main()

