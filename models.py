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


class ResNet(nn.Module):
    def __init__(self, input, num_classes):
        super(ResNet, self).__init__()
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
