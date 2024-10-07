import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from ReadingDataset import TrainingsDataset


class Trainer:
    def __init__(self, csv_file, directory, model, batch_size=64, num_epochs=20,
                 learning_rate=0.001, stratify_feature='stable_height', validation_size=0.2):
        self.csv_file = csv_file
        self.directory = directory
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.stratify_feature = stratify_feature
        self.validation_size = validation_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.data = pd.read_csv(self.csv_file)
        self.trainings_data, self.validation_data = self.split_data()

        self.trainings_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.validation_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.trainings_dataset = TrainingsDataset(self.trainings_data, self.directory, self.trainings_transform)
        self.validation_dataset = TrainingsDataset(self.validation_data, self.directory, self.validation_transform)

        self.trainings_loader = DataLoader(self.trainings_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False)

    def split_data(self):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.validation_size, random_state=42)
        for train_index, validation_index in sss.split(self.data, self.data[self.stratify_feature]):
            trainings_data = self.data.iloc[train_index]
            validation_data = self.data.iloc[validation_index]

        print(f'Trainings data: {len(trainings_data)}', f'Validation data: {len(validation_data)}')
        return trainings_data, validation_data

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            running_acc = 0.0

            with tqdm(self.trainings_loader, unit='batch') as tepoch:
                for images, labels in tepoch:
                    images, labels = images.to(self.device), labels.to(self.device).long() - 1
                    self.optimizer.zero_grad()
                    outputs = self.model(images)

                    predictions = torch.argmax(outputs, 1)

                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                    running_acc += (predictions == labels).sum().item()
                    tepoch.set_postfix(loss=running_loss / len(self.trainings_loader), acc=running_acc / len(self.trainings_loader))

            # validation
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in self.validation_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    predictions = torch.argmax(outputs, 1)
                    total += labels.size(0)
                    correct += (predictions == labels).sum().item()

            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(self.trainings_loader)}, '
                  f'Accuracy: {100 * correct / total}')

            # self.model.train()
            # running_loss = 0.0
            # for images, labels in tqdm(self.trainings_loader):
            #     images, labels = images.to(self.device), labels.to(self.device).long() - 1
            #     self.optimizer.zero_grad()
            #     outputs = self.model(images)
            #
            #     predictions = torch.argmax(outputs, 1)
            #
            #     loss = self.criterion(outputs, labels)
            #     loss.backward()
            #     self.optimizer.step()
            #     running_loss += loss.item()
            # print(f'Epoch {epoch + 1}, Loss: {running_loss / len(self.trainings_loader)}')
            #
            # self.model.eval()
            # correct = 0
            # total = 0
            # with torch.no_grad():
            #     for images, labels in self.validation_loader:
            #         images, labels = images.to(self.device), labels.to(self.device)
            #         outputs = self.model(images)
            #         predictions = torch.argmax(outputs, 1)
            #         total += labels.size(0)
            #         correct += (predictions == labels).sum().item()
            # print(f'Epoch {epoch + 1}, Accuracy: {100 * correct / total}')