"""Integration test: compare.py runs training+benchmark on 2 synthetic models."""
import json
from pathlib import Path
import yaml
import pytest


@pytest.fixture
def mini_compare_config(tmp_path, synthetic_dataset_root):
    out_root = tmp_path / "outputs"
    cfg = {
        "common": {
            "data": {
                "database_root": str(synthetic_dataset_root),
                "num_workers": 0,
                "batch_size": 2,
                "image_size": 32,  # tiny to keep it fast
            },
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
        },
        "benchmark": {"warmup": 1, "iters": 3, "batch_sizes": [1, 4]},
        # Two tiny timm models — downloads avoided by pretrained: False
        "models": [
            {"name": "resnet50", "pretrained": False},
            {"name": "mobilenetv3_large_100", "pretrained": False},
        ],
    }
    p = tmp_path / "compare.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p, out_root


def test_compare_runs_and_writes_report(mini_compare_config):
    cfg_path, out_root = mini_compare_config
    from classification.compare import main
    out_dir = main(str(cfg_path))
    out_dir = Path(out_dir)

    # report files
    csv = out_dir / "comparison.csv"
    md = out_dir / "comparison.md"
    assert csv.exists()
    assert md.exists()

    # each model got its own sub-run
    runs = [d for d in out_dir.iterdir() if d.is_dir()]
    assert len(runs) >= 2


def test_compare_csv_has_expected_columns(mini_compare_config):
    cfg_path, _ = mini_compare_config
    from classification.compare import main
    out_dir = Path(main(str(cfg_path)))
    csv = (out_dir / "comparison.csv").read_text().splitlines()
    header = csv[0].split(",")
    required = {"model", "params", "accuracy", "f1_positive", "auc",
                "latency_bs1_ms", "throughput_bs_fps"}
    assert required.issubset(set(header))
