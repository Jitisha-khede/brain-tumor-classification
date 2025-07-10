import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import os

def get_few_shot_loaders(n_shot=5, root='brain_tumor_dataset'):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=root, transform=transform)

    class_indices = {label: [] for label in range(len(dataset.classes))}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    support_indices, query_indices = [], []

    for label in class_indices:
        indices = class_indices[label]
        random.shuffle(indices)
        support_indices.extend(indices[:n_shot])
        query_indices.extend(indices[n_shot:n_shot + 5])  # 5 query images per class

    support_set = Subset(dataset, support_indices)
    query_set = Subset(dataset, query_indices)

    support_loader = DataLoader(support_set, batch_size=len(support_set), shuffle=False)
    query_loader = DataLoader(query_set, batch_size=len(query_set), shuffle=False)

    return support_loader, query_loader
