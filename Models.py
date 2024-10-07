import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 6)

    def forward(self, x):
        return self.model(x)


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        self.model.fc = nn.Linear(2048, 6)

    def forward(self, x):
        return self.model(x)


class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.model = models.vit_base_patch16_224_in21k(pretrained=True)
        self.model.fc = nn.Linear(768, 6)

    def forward(self, x):
        return self.model(x)
