# train_teacher_student.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

from models.teacher_resnet import get_teacher_model
from models.student_mobilenet import get_student_model
from distillation import distillation_loss
from data.fewshot_loader import create_few_shot_dataset

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 1. Prepare the Few-shot Dataset
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset_path = "brain_tumor_dataset"
full_dataset = ImageFolder(root=dataset_path, transform=transform)

# Choose shots (5 or 10)
shots_per_class = 5
fewshot_dataset = create_few_shot_dataset(full_dataset, shots_per_class)
train_loader = DataLoader(fewshot_dataset, batch_size=4, shuffle=True)

# -----------------------------
# 2. Load Teacher & Student Models
# -----------------------------
teacher = get_teacher_model(device)
student = get_student_model(device)

# -----------------------------
# 3. Optimizer for Student
# -----------------------------
optimizer = optim.Adam(student.parameters(), lr=0.001)

# -----------------------------
# 4. Training Loop
# -----------------------------
def train_student(epochs=10, alpha=0.5, temperature=3.0):
    student.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)
            loss = distillation_loss(student_logits, teacher_logits, labels, alpha, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(student_logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Loss: {total_loss / len(train_loader):.4f}")
        print(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
        print("Confusion Matrix:\n", cm)
        torch.save(student.state_dict(), "mobilenet_student.pth")
        print("Student model saved as mobilenet_student.pth")

# -----------------------------
# 5. Train Student
# -----------------------------
train_student(epochs=10)
