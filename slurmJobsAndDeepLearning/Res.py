import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import os
import shutil
import zipfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from PIL import Image  
from torchvision.datasets import DatasetFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Define transformations
transformations = ToTensor()

# Define the dataset class
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # List all folders (black, blue, green, TTR) in the root directory
        self.classes = sorted(os.listdir(root))

        # Create a list of image file paths and corresponding labels
        self.samples = []
        for class_id, class_name in enumerate(self.classes):
            class_dir = os.path.join(root, class_name)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                self.samples.append((file_path, class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load image and apply transformations
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')  # Use PIL's Image
        if self.transform:
            image = self.transform(image)
        return image, label


"""## Model Training"""

# Split data into test, validation, test datasets
# Model Training Code:
# Training accuracy, precision, F1 score

# Ensuring repoducibility of results

random_seed = 42
torch.manual_seed(random_seed)

# Define the root directory
data_dir = './Rice'

# Load datasets
train_data = CustomImageDataset(root=os.path.join(data_dir, 'train'), transform=transformations)
val_data = CustomImageDataset(root=os.path.join(data_dir, 'val'), transform=transformations)
test_data = CustomImageDataset(root=os.path.join(data_dir, 'test'), transform=transformations)

batch_size = 32
train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 2, pin_memory = True)
test_dl = DataLoader(val_data, batch_size, num_workers = 2, shuffle=False)
val_dl = DataLoader(test_data , batch_size*2, num_workers = 2, pin_memory = True)

print("done loading")

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow = 16).permute(1, 2, 0))
        break
show_batch(train_dl)

model = models.resnet18(pretrained=True)
num_classes = len(train_data.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store the loss values for plotting
train_losses = []
val_losses = []
test_losses = []
all_predicted_labels = []
all_true_labels = []

# Training and Validation loop
import numpy as np

num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_correct = 0
    total_samples = 0
    epoch_loss = 0.0
    num_batches = 0

    for inputs, labels in train_dl:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    train_loss = epoch_loss / num_batches
    train_accuracy = total_correct / total_samples
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")

    # Validation loop
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_accuracy = correct / total
    print(f"Validation Accuracy after epoch {epoch + 1}: {validation_accuracy * 100:.2f}%")

torch.save(model.state_dict(), 'model.pth')
print("Model saved to model.pth")

#Testing Loop
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_dl:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_predicted_labels.extend(predicted.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())

accuracy = correct / total
print(f"Testing Accuracy: {accuracy * 100:.2f}%")