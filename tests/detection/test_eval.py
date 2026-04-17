def test_eval_module_imports():
    from detection import eval as deval
    assert hasattr(deval, "main")


def test_eval_build_val_kwargs():
    from detection.eval import build_val_kwargs
    cfg = {
        "data": {"data_yaml": "detection/data/data.yaml"},
        "model": {"name": "yolov8s.pt", "imgsz": 640},
        "train": {"batch": 16, "workers": 4, "device": "mps"},
        "output": {"project": "outputs/detection", "name": "val"},
    }
    kwargs = build_val_kwargs(cfg)
    assert kwargs["data"] == "detection/data/data.yaml"
    assert kwargs["imgsz"] == 640
    assert kwargs["device"] == "mps"
    assert kwargs["project"] == "outputs/detection"
