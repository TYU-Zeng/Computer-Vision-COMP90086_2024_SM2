import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class TrainingsDataset(Dataset):
    def __init__(self, data, directory, transform=None):
        self.data = data
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.data.iloc[idx, 0])
        image = Image.open(img_name + '.jpg')
        label = self.data.iloc[idx, -1]

        if self.transform:
            image = self.transform(image)

        return image, label


class TestDataset(Dataset):
    def __init__(self, data, directory, transform=None):
        self.data = data
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.data.iloc[idx, 0])
        image = Image.open(img_name + '.jpg')

        if self.transform:
            image = self.transform(image)

        return image, self.data.iloc[idx, 0]