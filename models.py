import torch as tr
from layers import Residual, Flatten, Conv2dSame


class Generator(tr.nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        num_blks = 3
        num_filters = 64

        blocks = []
        blocks.extend([
            Conv2dSame(cfg.num_channels, num_filters, 9),
            tr.nn.PReLU()
        ])

        # Add k resnet blocks
        for k in range(num_blks):
            blocks.append(Residual(num_filters))
        
        blocks.extend([
            Conv2dSame(num_filters, num_filters, 3),
            tr.nn.BatchNorm2d(num_filters)
        ])

        self.block0 = tr.nn.Sequential(*blocks)
        self.block0.apply(self.init_weights)

        self.block1 = tr.nn.Sequential(
            Conv2dSame(num_filters, num_filters*4, 3),
            tr.nn.PixelShuffle(2),
            tr.nn.PixelShuffle(2),
            tr.nn.PReLU(),
            Conv2dSame(num_filters//4, num_filters*4, 3),
            tr.nn.PixelShuffle(2),
            tr.nn.PixelShuffle(2),
            tr.nn.PReLU(),
            Conv2dSame(num_filters//4, cfg.num_channels, 9)
        )
        self.block1.apply(self.init_weights)

        self.pad = Conv2dSame(cfg.num_channels, num_filters, 1)

    def forward(self, x_in):
        print(x_in.shape)
        x_out = self.block0(x_in)
        x_pad = self.pad(x_in)
        return self.block1(x_out + x_pad)
    
    def init_weights(self, layer):
        if type(layer) in [tr.nn.Conv2d, tr.nn.Linear]:
            tr.nn.init.xavier_uniform_(layer.weight)


class Discriminator(tr.nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        num_filters = 64
        # resolution divided by 4 strides of 2
        lr_hw_flat = int(cfg.lr_resolution[0] / 2**4)
        hr_hw_flat = int(cfg.hr_resolution[0] / 2**4)
        num_fc = 1024

        self.model = tr.nn.Sequential(
            # Channels in, channels out, filter size, stride, padding
            tr.nn.Conv2d(cfg.num_channels, num_filters, 3, 1, 1),
            tr.nn.LeakyReLU(),
            tr.nn.Conv2d(num_filters, num_filters, 3, 2, 1),
            tr.nn.BatchNorm2d(num_filters),
            tr.nn.LeakyReLU(),
            tr.nn.Conv2d(num_filters, num_filters*2, 3, 1, 1),
            tr.nn.BatchNorm2d(num_filters*2),
            tr.nn.LeakyReLU(),
            tr.nn.Conv2d(num_filters*2, num_filters*2, 3, 2, 1),
            tr.nn.BatchNorm2d(num_filters*2),
            tr.nn.LeakyReLU(),
            tr.nn.Conv2d(num_filters*2, num_filters*4, 3, 1, 1),
            tr.nn.BatchNorm2d(num_filters*4),
            tr.nn.LeakyReLU(),
            tr.nn.Conv2d(num_filters*4, num_filters*4, 3, 2, 1),
            tr.nn.BatchNorm2d(num_filters*4),
            tr.nn.LeakyReLU(),
            tr.nn.Conv2d(num_filters*4, num_filters*8, 3, 1, 1),
            tr.nn.BatchNorm2d(num_filters*8),
            tr.nn.LeakyReLU(),
            tr.nn.Conv2d(num_filters*8, num_filters*8, 3, 2, 1),
            tr.nn.BatchNorm2d(num_filters*8),
            tr.nn.LeakyReLU(),
            Flatten(),
            tr.nn.Linear(lr_hw_flat * lr_hw_flat * num_filters*8, num_fc),
            tr.nn.LeakyReLU(),
            tr.nn.Linear(num_fc, 1),
            tr.nn.Sigmoid()
        )
        self.model.apply(self.init_weights)

    def forward(self, x_in):
        x_out = self.model(x_in)
        return x_out
    
    def init_weights(self, layer):
        if type(layer) in [tr.nn.Conv2d, tr.nn.Linear]:
            tr.nn.init.xavier_uniform_(layer.weight)

