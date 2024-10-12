import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import csv
from dataset import TestDataset  # Assuming you have your TestDataset class defined in a separate file

def predict(model, test_csv, test_directory, batch_size=64):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Define test transformations (same as validation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the test dataset
    test_data = pd.read_csv(test_csv)
    test_dataset = TestDataset(test_data, test_directory, transform=test_transform)

    # Create a DataLoader for test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # List to store all predictions
    predictions = []

    # Disable gradient calculations for inference
    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            # Forward pass through the model
            outputs = model(images)

            # Get predicted class labels
            preds = torch.argmax(outputs, dim=1)

            # Store the predictions and corresponding image IDs
            predictions.extend(zip(image_ids, preds.cpu().numpy()))

    # Save predictions to a CSV file
    with open('predictions.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ImageID', 'Prediction'])
        writer.writerows(predictions)

    print("Predictions saved to predictions.csv")
