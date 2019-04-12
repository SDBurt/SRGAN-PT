import torch as tr


class Residual(tr.nn.Module):

    def __init__(self, num_filters):
        super(Residual, self).__init__()
        self.layer = tr.nn.Sequential(
            # Channels in, channels out, filter size, stride, padding
            tr.nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            tr.nn.BatchNorm2d(num_filters),
            tr.nn.PReLU(),
            tr.nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            tr.nn.BatchNorm2d(num_filters)
        )
        self.PReLU = tr.nn.PReLU()

    def forward(self, x_in):
        x_out = self.layer(x_in)
        return self.PReLU(x_out + x_in)

class Flatten(tr.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)

class Conv2dSame(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=tr.nn.ReflectionPad2d):
        super(Conv2dSame, self).__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.layer = torch.nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x_in):
        x_out = self.layer(x_in)
        return x_out

