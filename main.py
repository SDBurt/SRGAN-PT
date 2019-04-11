import torch as tr
import torchvision as tv
import cv2
import glob
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
<<<<<<< HEAD
=======
from PIL import Image
import cv2
import os

# Torch and Torchvision libraries
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as dset
import torchvision.utils as vutils

# Tensorboard and other needed libraries
>>>>>>> c2c2e00037c2e9db859737395dcce077aa174a11
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

<<<<<<< HEAD
        self.generator = Generator(cfg).to(device)
        self.discriminator = Discriminator(cfg).to(device)
=======
        # Configuration file
        self.cfg = cfg

        # Networks
        self.generator = GeneratorNetwork(self.cfg).to(device)
        self.discriminator = DiscriminatorNetwork(self.cfg).to(device)
        
        # Apply weights
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

        # Content loss (perceptual)
        # self.content_loss = LossNetwork(self.generator)

        # Adversarial loss
        # TODO: Apparently, add generative component of the GAN to the perceptual loss?
        # Probabilities of the discriminator over all training samples
        self.adversarial_loss = None

        # Optimizer
        # INFO:
        #   Optimized using adam with beta1 = 0.9.
        #   Learning rate is 10^-4 and update iteration is 10^6
        #   TODO: SRGAN variants used alternative rates
        beta1 = 0.9
        self.optimizerG = optim.Adam(self.generator.parameters(),
                            lr=self.cfg.learning_rate, betas=(beta1, 0.999))
        self.optimizerD = optim.Adam(self.discriminator.parameters(),
                            lr=self.cfg.learning_rate, betas=(beta1, 0.999))

        # Discriminator Variables
        self.real_label = 1
        self.fake_label = 0

        # build writer variables
        self.saver = None
        self.writer = None
        self.save_path = None
        self.ckpt_prefix = None

        # Global step
        self.global_step = 0

        # Build writiers
        self.build_writers()

        # Image Arrays
        self.hr_images = []
        self.lr_images = []

    # TODO
    # Configure to work for this project
    # Add paths to gitignore
    def build_writers(self):

        if not Path(cfg.save_dir).is_dir():
            os.mkdir(cfg.save_dir)

        if cfg.extension is None:
            cfg.extension = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        self.save_path      = cfg.save_dir + cfg.extension
        self.ckpt_prefix    = self.save_path + '/ckpt'

        # self.saver = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, optimizer_step=self.global_step)
        log_path            = cfg.log_dir + cfg.extension
        self.writer         = SummaryWriter(log_path)

>>>>>>> c2c2e00037c2e9db859737395dcce077aa174a11
    
    def preprocessing(self):
        high_res = [cv2.imread(file) for file in glob.glob(cfg.data_dir+"*.jpg")]
        low_res = self.downsample(high_res)

    def downsample(self, data, factor=4):
        '''
        Downsample list of images by applying a gaussian blur followed by a resize of (1 / factor).
        factor parameter defaults to 4 as per papers downsampling factor
        '''
<<<<<<< HEAD
        low_res = []
        for img in data:
            blur_img = cv2.GaussianBlur(img,(5,5),0)
            low_res.append(cv2.resize(blur_img, None, fx=(1/factor),fy=(1/factor), interpolation = cv2.INTER_CUBIC))
        return low_res
=======
        resized_data = data
        for img in resized_data:
            
            # img(c,h,w)
            img = T.Resize(img, int(img.size(2)/factor), Image.BICUBIC)
        
        return resized_data

    def batch(self, sample_size=16):
        combined = np.array(list(zip(self.lr_images, self.hr_images)))
        batch = random.sample(combined, sample_size)
        print(batch.shape)
        return zip(*batch)


    def train(self, cfg, dataloader):
        '''
        Train the NN
        '''

        # Need high resolution and low resolution images here
        # Low resolution computed using Gaussian filter and then downsampling operation
        # downsample()

        # For each epoch
        # TODO
        # determine how many epochs were used
        # for epoch in trange(cfg.episodes):
            # For each batch in dataloader
            # TODO
            # determine batch size / if batches were used
            # for i, data in enumerate(dataloader, 0):
                


 
        

def main(srgan, cfg):

    # from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    # TODO
    # Look at resizing and use the following as parameter to imagefolder
    # transform=transforms.Compose([
    #                           transforms.Resize(image_size),
    #                           transforms.CenterCrop(image_size),
    #                           transforms.ToTensor(),
    #                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    dataset = dset.ImageFolder(root=cfg.dataset, transform=T.Compose([
        T.CenterCrop(cfg.image_size), 
        T.ToTensor()
        ]))


    srgan.hr_images = dset.ImageFolder(root=cfg.dataset, transform=T.Compose([
        T.CenterCrop(cfg.image_size), 
        T.ToTensor()
        ]))
    
    scale_factor = 4

    srgan.lr_images = dset.ImageFolder(root=cfg.dataset, transform=T.Compose([
        T.CenterCrop(cfg.image_size), 
        T.Resize(int(cfg.image_size/scale_factor), Image.BICUBIC),
        T.ToTensor()
        ]))

    #srgan.lr_images = srgan.downsample(srgan.hr_images)

    #low_res, high_res = srgan.batch()
    # Create the dataloader
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # test plot to see images
    hr_dataloader = torch.utils.data.DataLoader(srgan.hr_images, batch_size=cfg.batch_size, shuffle=False)
    lr_dataloader = torch.utils.data.DataLoader(srgan.lr_images, batch_size=cfg.batch_size, shuffle=False)

    real_batch = next(iter(hr_dataloader))
    downsampled_batch = next(iter(lr_dataloader))

    plt.figure(figsize=(8,16))
    plt.subplot(2,1,1)
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    
    plt.subplot(2,1,2)
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(downsampled_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

    # srgan.train()
>>>>>>> c2c2e00037c2e9db859737395dcce077aa174a11

def main():
    srgan = SRGAN(cfg)
    srgan.preprocessing()

if __name__ == "__main__":
    main()

