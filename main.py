import torch as tr
import torchvision as tv
import numpy as np
import random, os, math
from tqdm import trange
from pathlib import Path
from datetime import datetime
from tensorboardX import SummaryWriter
from config import get_config
from models import Generator, Discriminator
from vgg import FeatureExtractor

from processing import get_dataset, downsample, reverse_normalize, normalize, randomcrop, tt

device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
cfg = get_config()

class SRGAN(object):
    def __init__(self, cfg):
        super(SRGAN, self).__init__()

        # Networks
        self.generator = Generator(cfg).to(device)
        self.discriminator = Discriminator(cfg).to(device)

        # Optimizers
        self.optim_gen = tr.optim.Adam(self.generator.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))
        self.optim_disc = tr.optim.Adam(self.discriminator.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, 0.999))
          
        # loss models
        self.mse_loss = tr.nn.MSELoss()
        self.bce_loss = tr.nn.BCELoss(reduction='sum')
        self.feature_extractor = FeatureExtractor()

        # Get image dataset (cropped)
        dataset = get_dataset(cfg.data_dir)
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
            image = reverse_normalize(state)
            self.writer.add_image(name, image, self.global_step)

    def pretrain(self):
        '''Pretrain the generator network using a MSE loss function. This is done so that SRGAN generator does not fall into a local optima'''
        
        if os.path.isfile('./pretrained_models/pretrained.pt'):
            print("loaded_checkpoint: ./pretrained_models/pretrained.pt")
            checkpoint = tr.load('./pretrained_models/pretrained.pt')
            self.generator.load_state_dict(checkpoint['generator_state'])
            self.optim_gen.load_state_dict(checkpoint['generator_optimizer'])
            self.generator.train()
        
        print('Pretraining Generator')

        ds = tr.FloatTensor(cfg.batch_size, cfg.num_channels, cfg.cropsize//cfg.factor, cfg.cropsize//cfg.factor)

        for epoch in trange(cfg.pretrain_epochs):
            
            # Batch
            for i, data in enumerate(self.dataloader):

                # mini-batch data
                hr, _ = data
                

                if hr.size(0) < cfg.batch_size:
                    break

                # Downsample images to low resolution
                # Removing NORMALIZE -> [0,1] instead of ~ N(0.485, 0.29)
                # remove in downscale if so
                for j in range(cfg.batch_size):
                    ds[j] = downsample(hr[j])
                    hr[j] = normalize(hr[j])

                # Generate the super resolution image
                sr = self.generator(ds.to(device))

                # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
                self.generator.zero_grad()

                loss = self.mse_loss(sr, hr)
                loss.backward()
                self.optim_gen.step()
        
                self.logger('Pretrain/Generator loss', loss)

                self.log_state('Pretrain/Downsampled', ds[0])
                self.log_state('Pretrain/Generated', sr[0])
                self.log_state('Pretrain/Original', hr[0])
                self.global_step += 1

            if epoch % cfg.save_freq == 0:
                tr.save({
                    'generator_state': self.generator.state_dict(),
                    'generator_optimizer': self.optim_gen.state_dict()
                }, './pretrained_models/pretrained.pt')

    def train(self):
        '''Train the SRGAN using the just pretrained or past pretrained generator.'''


        if os.path.isfile('./pretrained_models/pretrained.pt'):
            print("loaded_checkpoint: ./pretrained_models/pretrained.pt")
            checkpoint = tr.load('./pretrained_models/pretrained.pt')
            self.generator.load_state_dict(checkpoint['generator_state'])
            self.optim_gen.load_state_dict(checkpoint['generator_optimizer'])
            self.generator.train()

        print('Training SRGAN')

        ds = tr.FloatTensor(cfg.batch_size, cfg.num_channels, cfg.cropsize//cfg.factor, cfg.cropsize//cfg.factor)

        # Restart global step for SRGAN tape
        self.global_step = 0

        real_label = tr.ones((cfg.batch_size,1)).to(device)
        fake_label = tr.zeros((cfg.batch_size,1)).to(device)

        for epoch in trange(cfg.epochs):

            # Mini-batchs
            for i, data in enumerate(self.dataloader):
                # Generate data
                hr, _ = data

                # Prevent error if the dataloader runs out of images
                if hr.size(0) < cfg.batch_size:
                    break

                # mini-batch data (ToTensor() normalizes between [0,1])
                # randn() instead of rand() for normalized input [-1,1]
                nz = tr.rand((hr.size(0),hr.size(1),hr.size(2)//cfg.factor, hr.size(3)//cfg.factor))
                for j in range(cfg.batch_size):
                    ds[j] = downsample(hr[j])
                    hr[j] = normalize(hr[j])
                    nz[j] = normalize(nz[j])

                ## Begin Training Discriminator
                # ----------------------------------------------------------------
                self.discriminator.zero_grad()
                # Why zero_grad
                # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

                # BCELoss use explained here
                # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

                # train real
                truth = self.discriminator(hr.to(device))
                loss_truth = self.bce_loss(truth, real_label)
                loss_truth.backward() 

                # train fake
                noise = self.generator(nz.to(device))
                fake = self.discriminator(noise.detach())
                loss_fake = self.bce_loss(fake, fake_label)
                loss_fake.backward() 

                loss_disc = loss_truth + loss_fake
                self.optim_disc.step()

                # Begin Training Generator
                # ----------------------------------------------------------------
                self.generator.zero_grad()

                # Generate the super resolution image and judge it
                sr = self.generator(ds)
                generated = self.discriminator(sr)

                # Calculate Perceptual Loss
                true_features = self.feature_extractor(hr)
                generated_features = self.feature_extractor(sr)
                content_loss = self.mse_loss(generated_features, true_features)
                adversarial_loss = self.bce_loss(generated, real_label)
                loss_gen = content_loss + (10e-3 * adversarial_loss)

                # Optimize Generator
                loss_gen.backward()
                self.optim_gen.step()

                # Log Losses
                self.logger('SRGAN/Content Loss', content_loss)
                self.logger('SRGAN/Adversarial Loss', adversarial_loss)
                self.logger('SRGAN/Discriminator Loss', loss_disc)
                self.logger('SRGAN/Generator Loss', loss_gen)

                # Log Images
                self.log_state('SRGAN/Original', hr[0])
                self.log_state('SRGAN/Downsampled', ds[0])
                self.log_state('SRGAN/Generated', sr[0])
                
                # Increment Tape
                self.global_step += 1

            if epoch % cfg.save_freq == 0:
                tr.save({
                    'generator_state': self.generator.state_dict(),
                    'generator_optimizer': self.optim_gen.state_dict()
                }, self.save_path+'.pt')

            # Get image dataset (cropped)
            dataset = get_dataset(cfg.data_dir)
            self.dataloader = tr.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)  

def main():
    srgan = SRGAN(cfg)

    if cfg.pretrain is True:
        srgan.pretrain()

    srgan.train()


if __name__ == '__main__':
    main()
