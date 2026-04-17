"""Integration test: eval loads a segmentation checkpoint and writes metrics."""
import json
from pathlib import Path
import pytest


@pytest.fixture
def trained_output(tmp_path, synthetic_dataset_root):
    import yaml
    from segmentation.train import main as train_main

    out_root = tmp_path / "outputs"
    cfg = {
        "data": {
            "database_root": str(synthetic_dataset_root),
            "num_workers": 0,
            "batch_size": 2,
            "image_size": 64,
        },
        "model": {"num_classes": 3, "encoder_name": "resnet34", "encoder_weights": None},
        "train": {"epochs": 1, "lr": 0.001, "weight_decay": 0.0,
                  "optimizer": "adamw", "scheduler": "cosine",
                  "ce_weight": 0.5, "dice_weight": 0.5},
        "eval": {"save_qualitative_every_n_epochs": 100, "num_qualitative_samples": 0},
        "output": {"root": str(out_root)},
        "seed": 0,
        "device": "cpu",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out_dir = train_main(str(cfg_path))
    return out_dir, cfg_path


def test_seg_eval_produces_metrics(trained_output):
    out_dir, cfg_path = trained_output
    from segmentation.eval import main as eval_main
    result = eval_main(str(cfg_path), str(Path(out_dir) / "ckpt" / "best.pt"))
    assert "miou" in result
    assert "pixel_accuracy" in result
    assert "iou_per_class" in result
    assert len(result["iou_per_class"]) == 3


def test_seg_eval_writes_metrics_json_and_qualitative(trained_output):
    out_dir, cfg_path = trained_output
    from segmentation.eval import main as eval_main
    eval_main(str(cfg_path), str(Path(out_dir) / "ckpt" / "best.pt"),
              num_qualitative_samples=2)
    metrics_path = Path(out_dir) / "metrics.json"
    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text())
    assert "miou" in data

    # qualitative/: original | GT | pred overlay PNGs
    qdir = Path(out_dir) / "qualitative"
    assert qdir.exists()
    pngs = list(qdir.glob("*.png"))
    assert len(pngs) >= 2
