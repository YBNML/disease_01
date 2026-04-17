"""Integration test: eval.py loads a checkpoint and produces a metrics.json."""
import json
from pathlib import Path
import pytest


@pytest.fixture
def trained_output(tmp_path, synthetic_dataset_root):
    """Train one epoch on synthetic data to produce a checkpoint to evaluate."""
    import yaml
    from classification.train import main as train_main

    out_root = tmp_path / "outputs"
    cfg = {
        "data": {
            "database_root": str(synthetic_dataset_root),
            "num_workers": 0,
            "batch_size": 2,
            "image_size": 32,
        },
        "model": {"num_classes": 2, "pretrained": False},
        "train": {
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "use_weighted_sampler": True,
        },
        "eval": {"save_misclassified": False, "save_qualitative_every_n_epochs": 100},
        "output": {"root": str(out_root)},
        "seed": 0,
        "device": "cpu",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out_dir = train_main(str(cfg_path))
    return out_dir, cfg_path


def test_eval_produces_metrics_json(trained_output):
    out_dir, cfg_path = trained_output
    from classification.eval import main as eval_main
    result = eval_main(str(cfg_path), str(Path(out_dir) / "ckpt" / "best.pt"))
    assert "accuracy" in result
    assert "f1_positive" in result
    assert "confusion_matrix" in result


def test_eval_writes_confusion_matrix_image(trained_output, tmp_path):
    out_dir, cfg_path = trained_output
    from classification.eval import main as eval_main
    eval_main(str(cfg_path), str(Path(out_dir) / "ckpt" / "best.pt"))
    cm_path = Path(out_dir) / "confusion_matrix.png"
    assert cm_path.exists()
