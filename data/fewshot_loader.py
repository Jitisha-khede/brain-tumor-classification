# data/fewshot_loader.py
import random
from torch.utils.data import Subset
from collections import defaultdict

def create_few_shot_dataset(dataset, shots_per_class):
    """
    Selects `shots_per_class` samples per class from the dataset.

    Args:
        dataset: ImageFolder dataset
        shots_per_class: int (e.g., 5 or 10)

    Returns:
        fewshot_subset: Subset of the dataset
    """
    class_to_indices = defaultdict(list)

    # Collect indices for each class
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    fewshot_indices = []
    for label, indices in class_to_indices.items():
        selected = random.sample(indices, shots_per_class)
        fewshot_indices.extend(selected)

    fewshot_subset = Subset(dataset, fewshot_indices)
    return fewshot_subset
