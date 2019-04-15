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
from vgg import FeatureExtractor

from processing import get_dataset, normalize, reverse_normalize, downsample

device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
cfg = get_config()

class SRGAN(object):
    def __init__(self, cfg):
        super(SRGAN, self).__init__()

        # Networks
        self.generator = Generator(cfg)
        self.discriminator = Discriminator(cfg)

        # Optimizers
        self.optim_gen = tr.optim.Adam(self.generator.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))
        self.optim_disc = tr.optim.Adam(self.discriminator.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))
          
        # loss models
        self.mse_loss = tr.nn.MSELoss()
        self.bce_loss = tr.nn.BCELoss()
        self.feature_extractor = FeatureExtractor(vgg19(pretrained=True))

        #self.preprocessing()
        cwd = os.getcwd()
        data_path = cwd + cfg.data_dir
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
            image = reverse_normalize(state[0])
            self.writer.add_image(name, image, self.global_step)

    def pretrain(self):

        print('Pretraining Generator')
        
        ds = tr.FloatTensor(cfg.batch_size, cfg.num_channels, cfg.cropsize//cfg.factor, cfg.cropsize//cfg.factor)

        for epoch in trange(cfg.pretrain_epochs):
            
            # Batch
            for i, data in enumerate(self.dataloader):

                # Generate data
                hr, _ = data

                if hr.size(0) < cfg.batch_size:
                    break

                # Downsample images to low resolution and normalize
                for j in range(cfg.batch_size):
                    ds[j] = downsample(hr[j])
                    hr[j] = normalize(hr[j])

                # Generate the super resolution image
                sr = self.generator(ds)

                # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
                self.generator.zero_grad()

                loss = self.mse_loss(sr, hr)
                loss.backward()
                self.optim_gen.step()
        
                self.logger('Pretrain/Generator loss', loss)

                self.log_state('Pretrain/Downsampled', ds)
                self.log_state('Pretrain/Generated', sr)
                self.log_state('Pretrain/Original', hr)
                self.global_step += 1

            if epoch % cfg.save_freq == 0:
                tr.save({
                    'model_gen': self.generator.state_dict(),
                    'optim_gen': self.optim_gen.state_dict(),
                    }, './pretrained_models/pretrained.pt')

    def train(self):

        if os.path.isfile('./pretrained_models/pretrained.pt') and cfg.pretrain is False:
            checkpoint = tr.load('./pretrained_models/pretrained.pt')
            self.generator.load_state_dict(checkpoint['model_gen'])
            self.optim_gen.load_state_dict(checkpoint['optim_gen'])
            self.generator.train()

        print('Training SRGAN')

        ds = tr.FloatTensor(cfg.batch_size, cfg.num_channels, cfg.cropsize//cfg.factor, cfg.cropsize//cfg.factor)

        # Restart global step for SRGAN tape
        self.global_step = 0

        real_label = tr.ones((cfg.batch_size,1))
        fake_label = tr.ones((cfg.batch_size,1))

        for epoch in trange(cfg.epochs):

            # Mini-batchs
            for i, data in enumerate(self.dataloader):
                # Generate data
                hr, _ = data

                # Prevent error if the dataloader runs out of images
                if hr.size(0) < cfg.batch_size:
                    break

                # Downsample/Normalize images
                for j in range(cfg.batch_size):
                    ds[j] = downsample(hr[j])
                    hr[j] = normalize(hr[j])

                # Generate the super resolution image
                sr = self.generator(ds)

                ## Begin Training Discriminator
                # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
                self.discriminator.zero_grad()

                # discriminator for natural and generated image
                generated = self.discriminator(sr)
                truth = self.discriminator(hr)

                # Calculate Discriminator Loss
                # BCELoss use explained here https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
                loss_disc = self.bce_loss(truth, real_label) + self.bce_loss(generated, fake_label)

                # Optimize Discriminator
                loss_disc.backward(retain_graph=True)
                self.optim_disc.step()

                ## Begin Training Generator
                self.generator.zero_grad()

                # Calculate Perceptual Loss
                # From https://github.com/aitorzip/PyTorch-SRGAN/blob/master/train
                true_features = self.feature_extractor(hr)
                generated_features = self.feature_extractor(sr)
                content_loss = self.mse_loss(generated_features, true_features.detach())
                adversarial_loss = tr.sum(-tr.log(generated))
                loss_gen = content_loss + (1e-3 * adversarial_loss)

                # Optimize Generator
                loss_gen.backward()
                self.optim_gen.step()

                # Log Losses
                self.logger('SRGAN/Content Loss', content_loss)
                self.logger('SRGAN/Adversarial Loss', adversarial_loss)
                self.logger('SRGAN/Discriminator Loss', loss_disc)
                self.logger('SRGAN/Generator Loss', loss_gen)

                # Log Images
                self.log_state('SRGAN/Original', hr)
                self.log_state('SRGAN/Downsampled', ds)
                self.log_state('SRGAN/Generated', sr)
                
                # Increment Tape
                self.global_step += 1

            # if epoch % cfg.save_freq == 0:
            #     tr.save({
            #         'model_gen': self.generator.state_dict(),
            #         'model_disc': self.discriminator.state_dict(),
            #         'optim_gen': self.optim_gen.state_dict(),
            #         'optim_disc': self.optim_disc.state_dict(),
            #         'global_step': self.global_step
            #         }, '.pretrained_models/pretrained_20epochs.pt')


def main():
    srgan = SRGAN(cfg)

    if cfg.pretrain:
        srgan.pretrain()

    srgan.train()

if __name__ == '__main__':
    main()
