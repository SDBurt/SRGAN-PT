import torch as tr
from torchvision.models.vgg import vgg19

# Feature Extractor from https://github.com/aitorzip/PyTorch-SRGAN/blob/master/train
class FeatureExtractor(tr.nn.Module):
    def __init__(self, i=5, include_max_pool=False):
        super(FeatureExtractor, self).__init__()
        
        model = vgg19(pretrained=True)

        # Break layers and normalization by Tak-Wai Hui and Wai-Ho Kwok https://github.com/twhui/SRGAN-PyTorch
        children = list(model.features.children())
        max_pool_indices = [index for index, m in enumerate(children) if isinstance(m, tr.nn.MaxPool2d)]
        target_features = children[:max_pool_indices[i-1] + 1] if include_max_pool else children[:max_pool_indices[i-1]]    

        self.features =  tr.nn.Sequential(*target_features)
        
        mean = tr.autograd.Variable(tr.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)) # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
        std = tr.autograd.Variable(tr.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)) # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)



    def forward(self, x):
        x = (x - self.mean) / self.std
        y = self.features(x)
        return y
