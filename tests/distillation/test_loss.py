import torch
from distillation.loss import DistillationLoss


def test_kd_loss_perfect_teacher_zero_hard_loss_near_zero_soft():
    """When teacher equals student and matches labels perfectly, loss should be near zero."""
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    labels = torch.tensor([0, 1])
    loss_fn = DistillationLoss(alpha=0.5, temperature=4.0)
    loss = loss_fn(logits, logits, labels)
    assert loss.item() < 0.1


def test_kd_loss_returns_scalar():
    logits = torch.randn(4, 2)
    labels = torch.tensor([0, 1, 0, 1])
    loss_fn = DistillationLoss(alpha=0.5, temperature=4.0)
    loss = loss_fn(logits, logits, labels)
    assert loss.dim() == 0


def test_kd_loss_alpha_1_equals_ce():
    """alpha=1 → pure CE on student, independent of teacher."""
    student = torch.tensor([[2.0, 1.0]])
    labels = torch.tensor([0])
    teacher_a = torch.tensor([[10.0, -10.0]])
    teacher_b = torch.tensor([[-10.0, 10.0]])
    loss_fn = DistillationLoss(alpha=1.0, temperature=4.0)
    la = loss_fn(student, teacher_a, labels).item()
    lb = loss_fn(student, teacher_b, labels).item()
    assert abs(la - lb) < 1e-6
