# run_proto.py

from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
from proto_baseline import run_proto_episode

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms (same as main)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dataset
dataset = ImageFolder("brain_tumor_dataset", transform=transform)

# Run Prototypical Network - 5-shot
run_proto_episode(dataset, device, n_shot=5, n_query=5)

# Try also:
# run_proto_episode(dataset, device, n_shot=10, n_query=5)
