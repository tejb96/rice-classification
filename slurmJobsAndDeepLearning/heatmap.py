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



random_seed = 42
torch.manual_seed(random_seed)

# Define the root directory
data_dir = './Rice'

# Load datasets
#train_data = CustomImageDataset(root=os.path.join(data_dir, 'train'), transform=transformations)
val_data = CustomImageDataset(root=os.path.join(data_dir, 'val'), transform=transformations)
#test_data = CustomImageDataset(root=os.path.join(data_dir, 'test'), transform=transformations)

batch_size = 32
#train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 2, pin_memory = True)
test_dl = DataLoader(val_data, batch_size, num_workers = 2, shuffle=False)
#val_dl = DataLoader(test_data , batch_size*2, num_workers = 2, pin_memory = True)

print("done loading")

#load model
model = models.resnet18(pretrained=True)
num_classes = len(val_data.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load("model.pth"))

#test model to generate test results and heatmaps
all_predicted_labels = []
all_true_labels = []
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

# Get unique classes from true and predicted labels
unique_classes = np.unique(np.concatenate((all_true_labels, all_predicted_labels)))

# Create confusion matrix
conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels, labels=unique_classes)

# Plot confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig('heatmap_res.png')


