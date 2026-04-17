import numpy as np
import torch
from classification.metrics import ClassificationMetrics


def test_metrics_perfect_prediction():
    m = ClassificationMetrics(num_classes=2, positive_class=1)
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0], [10.0, -10.0]])
    labels = torch.tensor([0, 1, 0])
    m.update(logits, labels)
    result = m.compute()
    assert result["accuracy"] == 1.0
    assert result["f1_positive"] == 1.0
    # 2 true negatives, 1 true positive
    cm = result["confusion_matrix"]
    assert cm[0, 0] == 2 and cm[1, 1] == 1
    assert cm[0, 1] == 0 and cm[1, 0] == 0


def test_metrics_all_wrong():
    m = ClassificationMetrics(num_classes=2, positive_class=1)
    logits = torch.tensor([[-10.0, 10.0], [10.0, -10.0]])
    labels = torch.tensor([0, 1])
    m.update(logits, labels)
    result = m.compute()
    assert result["accuracy"] == 0.0
    assert result["f1_positive"] == 0.0


def test_metrics_mixed_batch_f1():
    """2 TP, 1 FP, 1 FN → precision=2/3, recall=2/3, F1=2/3 on positive class."""
    m = ClassificationMetrics(num_classes=2, positive_class=1)
    # pred positive for samples 0,1,2; pred negative for sample 3
    # true positive for samples 0,1,3; true negative for sample 2
    # so: 0=TP, 1=TP, 2=FP, 3=FN
    logits = torch.tensor([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])
    labels = torch.tensor([1, 1, 0, 1])
    m.update(logits, labels)
    result = m.compute()
    assert abs(result["f1_positive"] - 2 / 3) < 1e-6


def test_metrics_reset():
    m = ClassificationMetrics(num_classes=2, positive_class=1)
    logits = torch.tensor([[10.0, -10.0]])
    labels = torch.tensor([0])
    m.update(logits, labels)
    m.reset()
    # After reset, computing with no data should not raise but may return NaN / 0
    # We at least require no exception
    result = m.compute()
    assert "accuracy" in result


def test_metrics_auc_binary():
    m = ClassificationMetrics(num_classes=2, positive_class=1)
    # rank-correct scores: positives get higher score → AUC = 1
    logits = torch.tensor([[2.0, 1.0], [1.0, 2.0], [2.0, 1.0], [1.0, 3.0]])
    labels = torch.tensor([0, 1, 0, 1])
    m.update(logits, labels)
    result = m.compute()
    assert abs(result["auc"] - 1.0) < 1e-6
