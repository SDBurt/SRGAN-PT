# Utility Libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image

# Torch and Torchvision libraries
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as dset
import torchvision.utils as vutils

# Our Libraries
from config import get_config


# Decide which device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# From https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# TODO
# Setup for project parameters
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(cfg, dataloader):

    '''
    for epoch in trange(cfg.episodes):
        for i, data in enumerate(dataloader, 0):
            


    '''     
    

def main(cfg):

 # from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    # TODO
    # Look at resizing and use the following as parameter to imagefolder
    # transform=transforms.Compose([
    #                           transforms.Resize(image_size),
    #                           transforms.CenterCrop(image_size),
    #                           transforms.ToTensor(),
    #                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    dataset = dset.ImageFolder(root=cfg.path, transform=T.Compose([
        T.Resize(cfg.image_size),
        T.CenterCrop(cfg.image_size), 
        T.ToTensor()
        ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

    # Build networks
    # gen_net = GeneratorNetwork()
    # gen_net.apply(weights_init)
    # print(gen_net)
    # 
    # desc_net = DiscriminatorNetwork()
    # desc_net.apply(weights_init)
    # print(desc_net)

    # TODO
    # Loss function

    real_label = 1
    fake_label = 0

    # TODO
    # Optimizers

    train()


if __name__ == "__main__":


    cfg = get_config()

    main(cfg)

