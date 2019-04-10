# Utility Libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image

# Torch and Torchvision libraries
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as dset
import torchvision.utils as vutils

# Tensorboard and other needed libraries
from pathlib import Path
from datetime import datetime
from tensorboardX import SummaryWriter

# Our Libraries
from config import get_config
from models import DiscriminatorNetwork, GeneratorNetwork, LossNetwork

# Decide which device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# If todo is added for later work
TODO = None


# Implementation of Photo-Realistic Single Image Super-Resolution using a Generative Adversarial Network
# http://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf
class SRGAN(object):

    def __init__(self, cfg):
        super(SRGAN, self).__init__()

        # Configuration file
        self.cfg = cfg

        # Networks
        self.generator = GeneratorNetwork(self.cfg).to(device)
        self.discriminator = DiscriminatorNetwork(self.cfg).to(device)
        
        # Apply weights
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

        # Content loss (perceptual)
        self.content_loss = LossNetwork(self.generator)

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

    

    # From https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    # TODO
    # Setup for project parameters
    # custom weights initialization called on netG and netD
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02) # Need this
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02) # Need this
            nn.init.constant_(m.bias.data, 0)

    def downsample(self, data, factor=4):
        '''
        Downsample `data` by a specific factor using bicubic interpolation over a 4x4 pixel neighborhood.
            Data is the input array containing the images to downsample, while factor is the scaling to downsample.
            The factor parameter defaults to 4.
        '''
        resized_data = data[:]
        for img in resized_data:
            img = cv2.resize(img,fx=(1/factor),fy=(1/factor), interpolation=cv2.INTER_CUBIC)
        
        return resized_data

    def batch(self, sample_size=16):
        combined = np.array(list(zip(self.lr_images, self.hr_images)))
        batch = random.sample(combined, sample_size)
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
    dataset = dset.ImageFolder(root=cfg.path, transform=T.Compose([
        T.CenterCrop(cfg.image_size), 
        T.ToTensor()
        ]))


    srgan.hr_images = dataset
    srgan.lr_images = srgan.downsample(dataset)

    low_res, high_res = srgan.batch()
    # Create the dataloader
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # test plot to see images
    #real_batch = next(iter(dataloader))

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

    # srgan.train()


if __name__ == "__main__":


    cfg = get_config()
    srgan = SRGAN(cfg)
    main(srgan, cfg)

