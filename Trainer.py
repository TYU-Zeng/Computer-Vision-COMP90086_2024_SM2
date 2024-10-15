import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

import cv2


from ReadingDataset import TrainingsDataset


class Trainer:
    def __init__(self, csv_file, directory, model, batch_size=128, num_epochs=50,
                 learning_rate=0.001, stratify_feature='stable_height', validation_size=0.2):

        # 128比64更好
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        self.data = pd.read_csv(self.csv_file)
        self.trainings_data, self.validation_data = self.split_data()

        # acc list
        self.acc_list_train = []
        self.acc_list_val = []

        # loss list
        self.loss_list_train = []
        self.loss_list_val = []

        # data augmentation
        self.trainings_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=(0, 0), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.validation_transform = transforms.Compose([
            transforms.Resize((299, 299)),
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

    def accuracy(self, pred, true):
        correct = (pred == true).sum().item()
        total = len(true)
        return correct / total



    def preprocess_image(self, batch_images):
        """
        Preprocess a batch of 3D rendered block stack images to 2D representations using depth map and edge detection.

        Args:
        - batch_images (tensor): A batch of 3D images.

        Returns:
        - processed_batch (tensor): The preprocessed batch of images with depth and edge features.
        """
        processed_batch = []
        batch_images = batch_images.cpu().numpy()  # Convert tensor to numpy array for OpenCV processing

        for img in batch_images:
            # Convert from tensor (C, H, W) to (H, W, C) for OpenCV
            img = np.transpose(img, (1, 2, 0))
            img = (img * 255).astype(np.uint8)  # Convert back to image format

            # Convert the image to grayscale for simplicity
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Normalize the grayscale image to create a pseudo-depth map
            depth_map = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)

            # Apply edge detection (Canny Edge Detector)
            edges = cv2.Canny(depth_map, threshold1=30, threshold2=100)

            # Stack the depth map and edges as channels
            processed_image = np.stack((depth_map, edges), axis=-1)

            # Convert back to tensor and normalize for the model
            preprocess_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])  # Adjust normalization as needed
            ])

            processed_image = preprocess_transform(processed_image)
            processed_batch.append(processed_image)

        return torch.stack(processed_batch).to(self.device)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            running_acc = 0.0

            with tqdm(self.trainings_loader, unit='batch') as tepoch:
                for images, labels in tepoch:

                    images, labels = images.to(self.device), labels.to(self.device).long() - 1
                    self.optimizer.zero_grad()

                    preprocessed_img = self.preprocess_image(images)

                    # InceptionV3 在训练时返回两个输出，main_logits 和 aux_logits
                    outputs, aux_outputs = self.model(preprocessed_img)

                    # 计算主损失和辅助损失
                    main_loss = self.criterion(outputs, labels)
                    aux_loss = self.criterion(aux_outputs, labels)

                    loss = main_loss + 0.4 * aux_loss
                    loss.backward()

                    training_loss = loss.item()
                    self.optimizer.step()
                    running_loss += loss.item()

                    predictions = torch.argmax(outputs, 1)
                    acc = self.accuracy(predictions, labels)
                    running_acc += acc

                    tepoch.set_postfix(loss=loss.item(), acc=acc)

            # validation
            self.model.eval()
            correct = 0
            # total is the length of the validation data
            total = len(self.validation_data)

            all_preds = []
            all_labels = []

            with (torch.no_grad()):
                for images, labels in self.validation_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device).long() - 1
                    outputs = self.model(images)

                    loss = self.criterion(outputs, labels)

                    preds = torch.argmax(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    correct += (preds == labels).sum().item()

            self.acc_list_train.append(running_acc / len(self.trainings_loader))
            self.acc_list_val.append(correct / total)

            self.loss_list_train.append(running_loss / len(self.trainings_loader))
            self.loss_list_val.append(loss.item())

            print(f'Epoch {epoch + 1}, Training Loss: {running_loss / len(self.trainings_loader)}, '
                  f'Valid Accuracy: {100 * correct / total}')

            self.create_classification_report(torch.tensor(all_preds), torch.tensor(all_labels))
            self.create_confusion_matrix(predictions, labels)

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

    def create_classification_report(self, predictions, targets):
        pred = predictions.cpu().numpy()
        labels = targets.cpu().numpy()
        print(classification_report(labels, pred))

    def create_confusion_matrix(self, predictions, targets):
        pred = predictions.cpu().numpy()
        labels = targets.cpu().numpy()
        print(confusion_matrix(labels, pred))

    def plt_curve(self):
        # 绘制准确率曲线
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.acc_list_train, label='Training Accuracy', color='blue', marker='o')
        plt.plot(self.acc_list_val, label='Validation Accuracy', color='green', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()

        # 绘制损失曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.loss_list_train, label='Training Loss', color='blue', marker='o')
        plt.plot(self.loss_list_val, label='Validation Loss', color='green', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

        # 显示绘图
        plt.tight_layout()
        plt.show()


