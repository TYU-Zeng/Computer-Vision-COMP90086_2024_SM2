import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ReadingDataset import TestDataset


class Prediction:
    def __init__(self, model, csv_file, image_dir, model_path, batch_size=64):
        self.model = model
        self.model_path = model_path
        self.csv_file = csv_file
        self.image_dir = image_dir
        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.load_model()

        self.data_frame = pd.read_csv(self.csv_file)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_dataset = self.create_test_data()

    def load_model(self):
        # Load the model's saved state_dict
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()  # Set the model to evaluation mode for inference

    def create_test_data(self):
        test_data = TestDataset(self.data_frame, self.image_dir, self.transform)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        return test_loader


    def predict(self):
        all_predictions = []
        all_image_ids = []

        with torch.no_grad():
            for images, image_ids in tqdm(self.test_dataset):
                images = images.to(self.device)
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim = 1)
                all_predictions.extend(preds.cpu().numpy())
                all_image_ids.extend(image_ids.cpu().numpy())

        predictions_df = pd.DataFrame({'id': all_image_ids, 'stable_height': all_predictions})
        predictions_df['stable_height'] += 1

        return predictions_df

    def save_predictions(self, prediction, output_path):
        prediction.to_csv(output_path, index=False)