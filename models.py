import torch
import torch.nn as nn

from collections import namedtuple

# Class for VGG Loss
# TODO: Currently this is not optimized for our implementation
# Source: https://towardsdatascience.com/pytorch-implementation-of-perceptual-losses-for-real-time-style-transfer-8d608e2e9902
LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
    
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ResidualBlock(nn.Module):
    def __init__(self, inplane, outplane, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplane, outplane, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outplane)
        self.conv2 = nn.Conv2d(outplane, outplane, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outplane)

        self.relu = nn.ReLU(inplace=True)
        self.prelu =  nn.PReLU()
        self.downsample = downsample

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        # TODO
        # Check up on the prelu in the paper
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            res = self.downsample(x)

        x += res
        x = self.relu(x)
        return x

class GeneratorNetwork(nn.Module):
    def __init__(self, cfg):
        super(GeneratorNetwork, self).__init__()

        self.conv1 = nn.Conv2d(cfg.num_channels, 64, kernel_size=9, stride=1)
        self.prelu = nn.PReLU()
        self.layer = self.block(64, 64, 2, stride=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=1)
        self.pixelshuffle = nn.PixelShuffle(2) # This needs to be confirmed for upscale factor  
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(256, 3, kernel_size=9, stride=1)

    def block(self, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(ResidualBlock(inplanes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu(x)
        res = x

        # ResNet Blocks
        x = self.layer(x)
        x = self.layer(x)
        x = self.layer(x)
        x = self.layer(x)
        x = self.layer(x)
        x = self.layer(x)
        x = self.layer(x)
        x = self.layer(x)
        x = self.layer(x)
        x = self.layer(x)
        x = self.layer(x)
        x = self.layer(x)
        x = self.layer(x)
        x = self.layer(x)
        x = self.layer(x)
        x = self.layer(x) # 16


        x = self.conv2(x)
        x = self.bn2(x)
        x += res

        x = self.conv3(x)
        x = self.pixelshuffle(x)
        x = self.pixelshuffle(x)
        x = self.prelu(x)

        x = self.conv4(x)
        x = self.pixelshuffle(x)
        x = self.pixelshuffle(x)
        x = self.prelu(x)

        x = self.conv5(x)

        return x

class DiscriminatorNetwork(nn.Module):
    def __init__(self, cfg):
        super(DiscriminatorNetwork, self).__init__()

        # image_size default is 64 as per paper spec
        self.conv1 = nn.Conv2d(cfg.num_channels, 64, kernel_size=3, stride=1)
        self.lrelu = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2)
        self.bn8 = nn.BatchNorm2d(512)

        self.flatten = Flatten()

        # pytorch does not have a good flatten into dense
        # method so this is a TODO
        self.fc1 = nn.Linear(3872, 1024)

        self.fc2 = nn.Linear(1024, 1)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        # k3 n64 s1
        x = self.conv1(x)
        x = self.lrelu(x)
        # k3 n64 s2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        # k3 n128 s1
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu(x)
        # k3 n128 s2
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu(x)
        # k3 n256 s1
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.lrelu(x)
        # k3 n256 s2
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.lrelu(x)
        # k3 n512 s1
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.lrelu(x)
        # k3 n512 s2
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.lrelu(x)
        # flatten, dense(1024), lrelu, dense(1) sigmoid
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.fc2(x)
        
        return self.sigm(x)
