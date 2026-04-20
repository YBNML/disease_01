"""KD loss: alpha * CE(student, hard_label) + (1-alpha) * T^2 * KL(soft_student || soft_teacher)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, temperature: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        hard = self.ce(student_logits, labels)
        soft_s = F.log_softmax(student_logits / self.T, dim=1)
        soft_t = F.softmax(teacher_logits / self.T, dim=1)
        soft = F.kl_div(soft_s, soft_t, reduction="batchmean") * (self.T ** 2)
        return self.alpha * hard + (1.0 - self.alpha) * soft
