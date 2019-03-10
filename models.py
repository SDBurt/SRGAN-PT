import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, inplane, outplane, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplane, outplane, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outplane)
        self.conv2 = nn.Conv2d(outplane, outplane, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            res = self.downsample(x)

        out += res
        out = self.relu(out)
        return out


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class GeneratorNetwork(nn.Module):
    def __init__(self, input, num_classes):
        super(GeneratorNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.block(16, 32, 1, stride=2)
        self.layer2 = self.block(32, 32, 1, stride=2)
        #self.avg_pool =  nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.fc1 = nn.Linear(3872, 256)
        self.fc2 = nn.Linear(256, num_classes)
        #self.fc2 = nn.Linear(32, num_classes)

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
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class DiscriminatorNetwork(nn.Module):
    def __init__(self, input, num_classes):
        super(DiscriminatorNetwork, self).__init__()

        self.conv1 = nn.Conv2d(input, 64, kernel_size=3, stride=1)
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
