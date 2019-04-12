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

    def forward(self, x_in):
        x_out = self.layer(x_in)
        return tr.nn.PReLU(x_out + x_in)

class Flatten(tr.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)
