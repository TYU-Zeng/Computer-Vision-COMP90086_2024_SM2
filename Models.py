import torch
import torch.nn as nn
from torchvision import models

import timm

class CoCaModel(nn.Module):
    def __init__(self):
        super(CoCaModel, self).__init__()
        # 使用 timm 加载 CoCa 模型，选择预训练权重
        self.model = timm.create_model('coca_Large', pretrained=True)

        # 修改最后的分类层，适应任务需要的输出类别
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.model(x)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = models.resnet50(pretrained=True)

        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.model(x)


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        self.model.fc = nn.Linear(2048, 6)

    def forward(self, x):

        if self.training:
            main_logits, aux_logits = self.model(x)
            return main_logits, aux_logits
        else:
            main_logits = self.model(x)
            return main_logits


class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.model = models.vit_base_patch16_224_in21k(pretrained=True)
        self.model.fc = nn.Linear(768, 6)

    def forward(self, x):
        return self.model(x)
