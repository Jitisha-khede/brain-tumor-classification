# models/student_mobilenet.py
import torch.nn as nn
from torchvision import models

def get_student_model(device):
    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(pretrained=True)

    # Replace the classifier for 2-class classification
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)

    model = model.to(device)
    model.train()  # Student will be trained

    return model
