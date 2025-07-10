import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class FewShotDataset(Dataset):
    def __init__(self, data):
        self.data = data  # list of (img_tensor, label)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def get_few_shot_data(base_dir, k_shot=1, q_query=5):
    classes = os.listdir(base_dir)
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    support, query = [], []

    for cls in classes:
        path = os.path.join(base_dir, cls)
        images = [os.path.join(path, img) for img in os.listdir(path)]
        random.shuffle(images)
        support_imgs = images[:k_shot]
        query_imgs = images[k_shot:k_shot+q_query]

        for img_path in support_imgs:
            img = Image.open(img_path).convert("RGB")
            support.append((transform(img), label_map[cls]))

        for img_path in query_imgs:
            img = Image.open(img_path).convert("RGB")
            query.append((transform(img), label_map[cls]))

    return support, query
