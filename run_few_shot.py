import torch
import torch.nn.functional as F
from proto_net import PrototypicalNetwork
from utils import get_few_shot_loaders
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "mobilenet_student.pth"
N_SHOT = 10   # change to 1, 5, or 10
N_TRIALS = 5  # how many few-shot tasks to run for averaging

# Load model
model = PrototypicalNetwork(model_path=model_path, device=device).to(device)
model.eval()

accuracies, f1_scores = [], []
conf_matrices = []

for trial in range(N_TRIALS):
    support_loader, query_loader = get_few_shot_loaders(n_shot=N_SHOT)

    # Build prototypes
    class_embeddings = {}
    with torch.no_grad():
        for images, labels in support_loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            for i in range(len(images)):
                label = labels[i].item()
                if label not in class_embeddings:
                    class_embeddings[label] = []
                class_embeddings[label].append(embeddings[i])

        prototypes = {k: torch.stack(v).mean(0) for k, v in class_embeddings.items()}

        # Evaluate on query set
        all_preds, all_labels = [], []
        for images, labels in query_loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)

            preds = []
            for emb in embeddings:
                distances = [F.pairwise_distance(emb.unsqueeze(0), proto.unsqueeze(0)) for proto in prototypes.values()]
                pred = list(prototypes.keys())[torch.argmin(torch.tensor(distances))]
                preds.append(pred)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    accuracies.append(acc)
    f1_scores.append(f1)
    conf_matrices.append(cm)

    print(f"\nTrial {trial+1}/{N_TRIALS}")
    print(f"Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

# Summary
print("\n=== Few-Shot Classification Summary ===")
print(f"{N_SHOT}-Shot over {N_TRIALS} Trials")
print(f"Avg Accuracy: {np.mean(accuracies):.4f}")
print(f"Avg F1-Score: {np.mean(f1_scores):.4f}")
