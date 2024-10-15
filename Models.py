import torch
import torch.nn as nn
from torchvision import models


import open_clip
import torch.nn as nn


class CoCaModel(nn.Module):
    def __init__(self, num_classes=6):
        super(CoCaModel, self).__init__()
        # Load the CoCa model (contrastive + captioning architecture)
        self.model, _, preprocess = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14",
            pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )

        # Disable the gradient computation for the CoCa model (use it as a feature extractor)
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the classifier head with custom layers for classification
        self.fc = nn.Sequential(
            nn.Linear(self.model.visual.width, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Extract features using CoCa
        features = self.model.encode_image(x)

        # Pass features through the custom classifier
        return self.fc(features)


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

        self.model.Conv2d_1a_3x3.conv = nn.Conv2d(
            in_channels=2,  # Modify this to match the input channels from preprocessing
            out_channels=32,
            kernel_size=(3, 3),
            stride=(2, 2),
            bias=False
        )


        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(128, 6)
        )

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
