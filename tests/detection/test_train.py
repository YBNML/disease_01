import pytest
import yaml


def test_train_module_imports():
    from detection import train
    assert hasattr(train, "main")


def test_train_build_ultralytics_kwargs_from_config(tmp_path):
    from detection.train import build_ultralytics_kwargs
    cfg = {
        "data": {"data_yaml": "detection/data/data.yaml"},
        "model": {"name": "yolov8s.pt", "imgsz": 640},
        "train": {"epochs": 5, "batch": 16, "lr0": 0.01, "workers": 4,
                  "patience": 30, "device": "mps"},
        "output": {"project": "outputs/detection", "name": "run"},
        "seed": 42,
    }
    kwargs = build_ultralytics_kwargs(cfg)
    assert kwargs["data"] == "detection/data/data.yaml"
    assert kwargs["epochs"] == 5
    assert kwargs["batch"] == 16
    assert kwargs["imgsz"] == 640
    assert kwargs["device"] == "mps"
    assert kwargs["project"] == "outputs/detection"
    assert kwargs["name"] == "run"
    assert kwargs["seed"] == 42


def test_train_device_auto_resolves_to_mps_or_cpu(tmp_path):
    from detection.train import build_ultralytics_kwargs
    cfg = {
        "data": {"data_yaml": "x"},
        "model": {"name": "yolov8s.pt", "imgsz": 640},
        "train": {"epochs": 1, "batch": 1, "lr0": 0.01, "workers": 0,
                  "patience": 10, "device": "auto"},
        "output": {"project": "o", "name": "r"},
        "seed": 0,
    }
    kwargs = build_ultralytics_kwargs(cfg)
    assert kwargs["device"] in ("mps", "cpu")
