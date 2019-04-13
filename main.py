import torch as tr
import torchvision as tv
import numpy as np
import cv2, random, os, h5py
from tqdm import trange
from pathlib import Path
from datetime import datetime
from tensorboardX import SummaryWriter
from config import get_config
from models import Generator, Discriminator
from preprocessing import package_data
from torchvision.models.vgg import vgg13
from vgg import LossNetwork



device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
#device = 'cpu'
cfg = get_config()

class SRGAN(object):
    def __init__(self, cfg):
        super(SRGAN, self).__init__()

        self.global_step = 0

        self.generator = Generator(cfg)
        self.discriminator = Discriminator(cfg)
        # self.vgg = vgg19(pretrained=True).features.to(device).eval()

        self.optim = tr.optim.Adam(list(self.generator.parameters()) + list(self.discriminator.parameters()), cfg.learning_rate)
        
        self.preprocessing()
        self.build_writers()

        # vgg_model = vgg13(pretrained=True).to(device)
        # self.loss_network = LossNetwork(vgg_model)
        # self.loss_network.eval()
        self.mse_loss = tr.nn.MSELoss()

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

        self.save_path = cfg.save_dir + cfg.extension

        log_path = cfg.log_dir + cfg.extension
        self.writer = SummaryWriter(log_path)

    def logger(self, loss):
        if self.global_step % cfg.log_freq == 0:
            # Log vars
            self.writer.add_scalar('loss', loss, self.global_step)

    def log_state(self, state, name):
        if self.global_step % (cfg.log_freq * 10) == 0:
            self.writer.add_image(name, state, self.global_step)
    
    def update(self, hr, ds):

        sr = self.generator(ds)
        
        generated = self.discriminator(sr)
        truth = self.discriminator(hr)

        # print("-- VGG Generated")
        # print(f"sr.shape: {sr.shape}")
        # vgg_sr = self.loss_network(sr)

        # print("-- VGG Truth")
        # print(f"hr.shape: {hr.shape}")
        # vgg_hr = self.loss_network(hr)

        # Euclidean distance between features
        # content_loss = self.mse_loss(vgg_hr, vgg_sr)
        content_loss = self.mse_loss(sr, hr)
        
        adversarial_loss = -np.log(generated)

        # Perceptual Loss (VGG loss)
        loss = content_loss + (1e-3 * adversarial_loss)
        self.logger(loss)

        loss.backward()
        self.optim.step()
        
        

    def get_batch(self):
        # select batch
        if self.size < cfg.batch_size:
            batch = random.sample(self.data_tr, self.size)
        else:
            batch = random.sample(self.data_tr, cfg.batch_size)

        return batch

    def train(self):
        # Load model
        if os.path.isfile(self.save_path):
            self.generator.load_state_dict(tr.load(self.save_path))
        for epoch in trange(cfg.epochs):
            batch = self.get_batch()
            for hr, ds in batch:
                # HWC -> NCHW, make type torch.cuda.float32
                
                ds = tr.tensor(ds[None], dtype=tr.float32).permute(0, 3, 1, 2)
                hr = tr.tensor(hr[None], dtype=tr.float32).permute(0, 3, 1, 2)
                self.update(hr, ds)
                self.global_step += 1
            if epoch % cfg.save_freq == 0:
                tr.save({'model': self.generator.state_dict(), 'optim': self.optim.state_dict(), 'global_step': self.global_step}, self.save_path)


def main():
    srgan = SRGAN(cfg)
    srgan.train()

if __name__ == '__main__':
    main()
