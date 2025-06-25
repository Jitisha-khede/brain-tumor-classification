# proto_baseline.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# ✅ Feature Extractor (Backbone CNN)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(model.children())[:-1])  # Remove classifier
        self.out_features = model.fc.in_features  # 512

    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1)  # Flatten to (batch_size, feature_dim)


# ✅ Create N-shot support and query sets
def create_episode(dataset, n_shot=5, n_query=5):
    label_to_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    support_indices = []
    query_indices = []

    for label, indices in label_to_indices.items():
        selected = random.sample(indices, n_shot + n_query)
        support_indices.extend(selected[:n_shot])
        query_indices.extend(selected[n_shot:])

    return Subset(dataset, support_indices), Subset(dataset, query_indices)


# ✅ Euclidean Distance
def euclidean_dist(x, y):
    n, d = x.shape
    m, _ = y.shape
    return torch.cdist(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0)  # (n, m)


# ✅ Prototypical Network Inference
def run_proto_episode(dataset, device, n_shot=5, n_query=5):
    support_set, query_set = create_episode(dataset, n_shot, n_query)

    support_loader = DataLoader(support_set, batch_size=n_shot*2, shuffle=False)
    query_loader = DataLoader(query_set, batch_size=n_query*2, shuffle=False)

    model = FeatureExtractor().to(device)
    model.eval()

    # Get support embeddings
    support_features, support_labels = [], []
    for images, labels in support_loader:
        with torch.no_grad():
            images = images.to(device)
            features = model(images).cpu()
            support_features.append(features)
            support_labels.extend(labels)

    support_features = torch.cat(support_features)
    support_labels = torch.tensor(support_labels)

    # Compute class prototypes
    prototypes = []
    for cls in torch.unique(support_labels):
        cls_features = support_features[support_labels == cls]
        proto = cls_features.mean(dim=0)
        prototypes.append(proto)

    prototypes = torch.stack(prototypes)

    # Classify queries
    all_preds, all_labels = [], []
    for images, labels in query_loader:
        with torch.no_grad():
            images = images.to(device)
            features = model(images).cpu()
            dists = euclidean_dist(features, prototypes)
            preds = dists.argmin(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Prototypical Network Results ({n_shot}-shot):")
    print(f"Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
