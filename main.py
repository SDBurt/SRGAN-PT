import torch as tr
import torchvision as tv
import numpy as np
import cv2, random, os, h5py, math
from tqdm import trange
from pathlib import Path
from datetime import datetime
from tensorboardX import SummaryWriter
from config import get_config
from models import Generator, Discriminator
from preprocessing import package_data
from torchvision.models.vgg import vgg19
from vgg import LossNetwork, FeatureExtractor

from processing import get_dataset, normalize, reverse_normalize, scale

device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
cfg = get_config()

class SRGAN(object):
    def __init__(self, cfg):
        super(SRGAN, self).__init__()

        # Networks
        self.generator = Generator(cfg)
        self.discriminator = Discriminator(cfg)

        # Optimizers
        self.optim = tr.optim.Adam(list(self.generator.parameters()) + list(self.discriminator.parameters()), cfg.learning_rate)
          
        # loss models
        self.mse_loss = tr.nn.MSELoss()
        self.feature_extractor = FeatureExtractor(vgg19(pretrained=True))

        #self.preprocessing()
        cwd = os.getcwd()
        data_path = cwd + cfg.data_dir + "/BSDS300"
        print(data_path)
        dataset = get_dataset(data_path)
        self.dataloader = tr.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        # For logging results to tensorboard
        self.global_step = 0
        self.build_writers()


    def build_writers(self):
        if not Path(cfg.save_dir).is_dir():
            os.mkdir(cfg.save_dir)
        if cfg.extension is None:
            cfg.extension = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        self.save_path = cfg.save_dir + cfg.extension

        log_path = cfg.log_dir + cfg.extension
        self.writer = SummaryWriter(log_path)


    def logger(self, name, loss):
        if self.global_step % cfg.log_freq == 0:
            # Log vars
            self.writer.add_scalar(name, loss, self.global_step)


    def log_state(self, name, state):
        if self.global_step % (cfg.log_freq * 5) == 0:
            print(state.shape)
            image = reverse_normalize(state[0])
            self.writer.add_image(name, image, self.global_step)


    def train(self):
        # Load model
        if os.path.isfile(self.save_path):
            self.generator.load_state_dict(tr.load(self.save_path))

        ds = tr.FloatTensor(cfg.batch_size, 3, 24, 24)

        for epoch in trange(2):
            
            # Batch
            for i, data in enumerate(self.dataloader):

                # Generate data
                hr, _ = data

                if hr.size(0) < cfg.batch_size:
                    break

                # Downsample images to low resolution and normalize
                for j in range(cfg.batch_size):
                    ds[j] = scale(hr[j])
                    hr[j] = normalize(hr[j])



                # Generate the super resolution image
                sr = self.generator(ds)

                # pixelwise MSE
                # print(f"ds.shape: {ds.shape}, hr.shape: {hr.shape}, sr.shape: {sr.shape}")
                loss = self.mse_loss(sr, hr)

                loss.backward()
                self.optim.step()
        
                self.logger("loss", loss)
                self.log_state("Generated", sr)
                self.log_state("Original", hr)
                self.global_step += 1

            if epoch % cfg.save_freq == 0:
                tr.save({'model': self.generator.state_dict(), 'optim': self.optim.state_dict(), 'global_step': self.global_step}, self.save_path)

        for epoch in trange(cfg.epochs):

            # Batch
            for i, data in enumerate(self.dataloader):
                # Generate data
                hr, _ = data

                if hr.size(0) < cfg.batch_size:
                    break

                # Downsample images to low resolution and normalize
                for j in range(cfg.batch_size):
                    ds[j] = scale(hr[j])
                    hr[j] = normalize(hr[j])

                # Generate the super resolution image
                sr = self.generator(ds)

                # discriminator for natural and generated image
                generated = self.discriminator(sr)
                truth = self.discriminator(hr)

                # From https://github.com/aitorzip/PyTorch-SRGAN/blob/master/train
                true_features = self.feature_extractor(hr)
                generated_features = self.feature_extractor(sr)

                # determine perceptual loss
                content_loss = self.mse_loss(true_features, generated_features)
                adversarial_loss = - math.log(generated) - math.log(truth)
                loss = content_loss + (1e-3 * adversarial_loss)


                loss.backward()
                self.optim.step()
        
                self.logger("loss", loss)
                self.logger("content_loss", content_loss)
                self.logger("adversarial_loss", adversarial_loss)
                self.log_state("Generated", sr)
                self.log_state("Original", hr)

                self.global_step += 1

            if epoch % cfg.save_freq == 0:
                tr.save({'model': self.generator.state_dict(), 'optim': self.optim.state_dict(), 'global_step': self.global_step}, self.save_path)




def main():
    srgan = SRGAN(cfg)
    srgan.train()

if __name__ == '__main__':
    main()
