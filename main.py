import torch as tr
import torchvision as tv
import numpy as np
import cv2, random, os, h5py
from tqdm import trange
from pathlib import Path
from time import datetime
from tensorboardX import SummaryWriter
from config import get_config
from models import Generator, Discriminator
from preprocessing import package_data


device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
cfg = get_config()

class SRGAN(object):

    def __init__(self, cfg):
        super(SRGAN, self).__init__()

        self.global_step = 0

        self.generator = Generator(cfg).to(device)
        self.discriminator = Discriminator(cfg).to(device)

        self.preprocessing()
        self.build_writers()

    def preprocessing(self):
        if cfg.package_data:
            # Package data into H5 format
            package_data(cfg)

        # Load data
        cwd = os.getcwd()
        f = h5py.File(cwd + cfg.data_dir + '/data.h5', 'r')
        lr = f['lr'][:]
        hr = f['hr'][:]
        ds = f['ds'][:]
        f.close()

        self.data_tr = list(zip(hr, ds))
        self.size = len(self.data_tr)

    def build_writers(self):
        if not Path(cfg.save_dir).is_dir():
            os.mkdir(cfg.save_dir)
        if cfg.extension is None:
            cfg.extension = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        self.save_path = cfg.save_dir + '/' + cfg.extension

        log_path = cfg.log_dir + '/' + cfg.extension
        self.writer = SummaryWriter(log_path)

    def logger(self, tape, loss):
        if self.global_step % cfg.log_freq == 0:
            # Log vars
            self.writer.add_scalar('loss', loss, self.global_step)

    def log_state(self, state, name):
        if self.global_step % (cfg.log_freq * 10) == 0:
            self.writer.add_image(name, state, self.global_step)

    def get_batch(self):
        # select batch
        if self.size < cfg.batch_size:
            batch = random.sample(self.data_tr, self.size)
        else:
            batch = random.sample(self.data_tr, cfg.batch_size)

        return batch

    def train(self):
        if cfg.extension is not None:
            self.generator.load_state_dict(tr.load(self.save_path))
        for epoch in trange(cfg.epochs):
            batch = self.get_batch()
            for hr, ds in batch:
                # HWC -> NCHW, make type torch.cuda.float32
                ds = tr.tensor(ds[None], dtype=tr.float32).permute(0, 3, 1, 2).to(device)
                sr = self.generator(ds)

                if epoch % cfg.save_freq == 0:
                    tr.save(self.generator.state_dict(), self.save_path)


def main():
    srgan = SRGAN(cfg)
    srgan.train()

if __name__ == '__main__':
    main()

