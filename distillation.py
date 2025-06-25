# losses/distillation_loss.py
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=3.0):
    # Cross Entropy loss with true labels
    ce_loss = F.cross_entropy(student_logits, labels)

    # KL Divergence between teacher and student predictions
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)

    # Combined loss
    total_loss = alpha * ce_loss + (1 - alpha) * kd_loss
    return total_loss
