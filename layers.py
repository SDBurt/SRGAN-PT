import torch as tr
import math


class Flatten(tr.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


class Conv2dSame(tr.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(Conv2dSame, self).__init__()
        self.F = kernel_size
        self.S = stride
        self.D = dilation
        self.layer = tr.nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=dilation)

    def forward(self, x_in):
        print(x_in.shape)
        N, C, H, W = x_in.shape
        H2 = math.ceil(H / self.S)
        W2 = math.ceil(W / self.S)
        Pr = (H2 - 1) * self.S + (self.F - 1) * self.D + 1 - H
        Pc = (W2 - 1) * self.S + (self.F - 1) * self.D + 1 - W
        x_pad = tr.nn.ZeroPad2d((Pr//2, Pr - Pr//2, Pc//2, Pc - Pc//2))(x_in)
        x_out = self.layer(x_pad)
        return x_out


class Residual(tr.nn.Module):

    def __init__(self, num_filters):
        super(Residual, self).__init__()
        self.layer = tr.nn.Sequential(
            # Channels in, channels out, filter size, stride, padding
            Conv2dSame(num_filters, num_filters, 3),
            tr.nn.BatchNorm2d(num_filters),
            tr.nn.PReLU(),
            Conv2dSame(num_filters, num_filters, 3),
            tr.nn.BatchNorm2d(num_filters)
        )
        self.PReLU = tr.nn.PReLU()

    def forward(self, x_in):
        x_out = self.layer(x_in)
        return self.PReLU(x_out + x_in)
 
