import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Path to your Brain MRI dataset
dataset_path = "brain_tumor_dataset"  

# Transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),         # Resize to 64x64
    transforms.ToTensor(),               # Convert to tensor
    transforms.Normalize([0.5], [0.5])   # Normalize to [-1, 1]
])

# Load dataset using ImageFolder
dataset = ImageFolder(root=dataset_path, transform=transform)

# Split into train and validation sets (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Check class mapping
print("Class-to-Index mapping:", dataset.class_to_idx)

# Check shape of one batch
for images, labels in train_loader:
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    break

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)    # input: 3x64x64 → output: 6x60x60
        self.pool = nn.MaxPool2d(2, 2)                 # output after pool: 6x30x30
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)   # output: 16x26x26 → pool: 16x13x13
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # Output layer for binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pool
        x = x.view(-1, 16 * 13 * 13)           # Flatten
        x = F.relu(self.fc1(x))                # Fully connected layers
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                        # No softmax here (use CrossEntropyLoss)
        return x

# Initialize model, loss, optimizer
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, val_loader, epochs=5):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
        evaluate(model, val_loader)

def evaluate(model, val_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    
    return acc, f1

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

train(model, train_loader, val_loader, epochs=5)


from torchvision import models

# --- Load ResNet18 Pretrained on ImageNet ---
resnet_model = models.resnet18(pretrained=True)

# Freeze all layers
for param in resnet_model.parameters():
    param.requires_grad = False

# Replace the final layer for 2 classes (binary classification)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 2)

# Move to device
resnet_model = resnet_model.to(device)

# Define loss and optimizer (only train the final layer)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet_model.fc.parameters(), lr=0.001)

print("\n=== Training ResNet18 (Transfer Learning) ===")
train(resnet_model, train_loader, val_loader, epochs=5)
