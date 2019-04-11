import torch as tr
import torchvision as tv
import cv2
import glob
import random
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

        self.preprocessing()

    def preprocessing(self):
        high_res = [cv2.imread(file) for file in glob.glob(cfg.data_dir+"*.jpg")]
        low_res = self.downsample(high_res)
        self.images = np.array(list(zip(high_res, low_res)))

    def downsample(self, data, factor=4):
        '''
        Downsample list of images by applying a gaussian blur followed by a resize of (1 / factor).
        factor parameter defaults to 4 as per papers downsampling factor
        '''
        assert len(data) > 0
        low_res = []
        for img in data:
            blur_img = cv2.GaussianBlur(img,(5,5),0)
            low_res.append(cv2.resize(blur_img, None, fx=(1/factor),fy=(1/factor), interpolation = cv2.INTER_CUBIC))
        return low_res

    def batch(self):
        '''return a batch from images'''

        size = len(self.images)
        # select batch
        if size < cfg.batch_size:
            batch = random.sample(list(self.images), size)
        else:
            batch = random.sample(list(self.images), cfg.batch_size)

        return batch

    def train(self):
        for epoch in range(cfg.epochs):
            batch = self.batch()
            for hr, lr in batch:
                sr = self.generator(tr.tensor(lr))
        
        
        
def main():
    srgan = SRGAN(cfg)
    srgan.train()

if __name__ == "__main__":
    main()

