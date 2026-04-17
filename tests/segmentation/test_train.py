"""Integration test: segmentation train runs one epoch on synthetic data."""
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
            "image_size": 64,  # tiny for speed
        },
        "model": {
            "num_classes": 3,
            "encoder_name": "resnet34",
            "encoder_weights": None,  # no download in tests
        },
        "train": {
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "ce_weight": 0.5,
            "dice_weight": 0.5,
        },
        "eval": {"save_qualitative_every_n_epochs": 100, "num_qualitative_samples": 0},
        "output": {"root": str(out_root)},
        "seed": 0,
        "device": "cpu",
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p, out_root


def test_seg_train_runs_one_epoch(mini_config):
    cfg_path, out_root = mini_config
    from segmentation.train import main
    out_dir = main(str(cfg_path))
    assert Path(out_dir).exists()
    assert (Path(out_dir) / "ckpt" / "last.pt").exists()
    assert (Path(out_dir) / "ckpt" / "best.pt").exists()
