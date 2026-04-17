"""Integration test: train.py runs one epoch on synthetic data without errors."""
import shutil
from pathlib import Path
import yaml
import pytest


@pytest.fixture
def mini_config(tmp_path, synthetic_dataset_root):
    out_root = tmp_path / "outputs"
    cfg = {
        "data": {
            "database_root": str(synthetic_dataset_root),
            "num_workers": 0,
            "batch_size": 2,
            "image_size": 32,  # tiny for speed
        },
        "model": {"num_classes": 2, "pretrained": False},  # no download in test
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
        "device": "cpu",  # tests must not require MPS
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p, out_root


def test_train_runs_one_epoch(mini_config, monkeypatch):
    """Smoke: train.main(cfg_path) runs one epoch on synthetic data, produces ckpt."""
    cfg_path, out_root = mini_config
    from classification.train import main
    exit_dir = main(str(cfg_path))
    # returns the output dir it created
    assert Path(exit_dir).exists()
    # ckpt/last.pt must exist
    assert (Path(exit_dir) / "ckpt" / "last.pt").exists()
