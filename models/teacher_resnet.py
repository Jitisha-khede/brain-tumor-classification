# models/teacher_resnet.py
import torch.nn as nn
from torchvision import models

def get_teacher_model(device):
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)

    # Replace the final layer for 2-class classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    model = model.to(device)
    model.eval()  # Set to eval mode, teacher won't be trained

    return model
