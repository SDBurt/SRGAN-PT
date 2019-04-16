import torch as tr
import math


class Flatten(tr.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)

class ConvLayer(tr.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = tr.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = tr.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class Conv2dSame(tr.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(Conv2dSame, self).__init__()
        self.F = kernel_size
        self.S = stride
        self.D = dilation
        self.layer = tr.nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=dilation)

    def forward(self, x_in):
        
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
            tr.nn.Conv2d(num_filters, num_filters, 3, padding=1),
            tr.nn.BatchNorm2d(num_filters),
            tr.nn.PReLU(),
            tr.nn.Conv2d(num_filters, num_filters, 3, padding=1),
            tr.nn.BatchNorm2d(num_filters)
        )

    def forward(self, x_in):
        x_out = self.layer(x_in)
        return x_out + x_in
 
