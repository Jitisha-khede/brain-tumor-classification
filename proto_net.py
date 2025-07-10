# proto_net.py
import torch
import torch.nn as nn
from models.student_mobilenet import get_student_model  # load your MobileNetV2

class PrototypicalNetwork(nn.Module):
    def __init__(self, model_path=None, device='cpu'):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = get_student_model(device)
        if model_path:
            self.backbone.load_state_dict(torch.load(model_path, map_location=device))
        
    def forward(self, x):
        """
        Extract embeddings from input images using the backbone.
        """
        embeddings = self.backbone(x)
        # Normalize embeddings
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        return embeddings

    def compute_prototypes(self, support_embeddings, support_labels):
        """
        Compute class prototypes (mean embeddings per class).
        """
        classes = torch.unique(support_labels)
        prototypes = []
        for cls in classes:
            cls_embeddings = support_embeddings[support_labels == cls]
            prototype = cls_embeddings.mean(dim=0)
            prototype = prototype / prototype.norm()  # normalize prototype
            prototypes.append(prototype)
        return torch.stack(prototypes), classes

    def predict(self, query_embeddings, prototypes):
        """
        Predict class by computing cosine similarity to prototypes.
        """
        # Compute cosine similarity
        sims = torch.mm(query_embeddings, prototypes.t())  # [N_query, N_proto]
        dists = 1 - sims  # Convert similarity to distance
        preds = torch.argmin(dists, dim=1)
        return preds
