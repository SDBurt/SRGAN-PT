from collections import namedtuple

import torch
from torchvision import models

class FeatureExtractor(torch.nn.Module):
    def __init__(self, cnn, feature_layer=19):
        super(FeatureExtractor, self).__init__()
        self.features = torch.nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)
