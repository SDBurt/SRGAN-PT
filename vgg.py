import torch
from torchvision import models

# Feature Extractor from https://github.com/aitorzip/PyTorch-SRGAN/blob/master/train
class FeatureExtractor(torch.nn.Module):
    def __init__(self, cnn, feature_layer=5):
        super(FeatureExtractor, self).__init__()
        self.features = torch.nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)
