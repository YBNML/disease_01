from pathlib import Path
import yaml
import pytest
from detection.prepare_yolo import prepare_split, prepare_all, CLASS_NAMES


def test_prepare_split_train_creates_structure(synthetic_dataset_root, tmp_path):
    dest = tmp_path / "yolo_data"
    n = prepare_split(synthetic_dataset_root, dest, split="train")
    # synthetic fixture has 1 normal + 1 canker polygon = 2 samples
    assert n == 2

    images = list((dest / "train" / "images").iterdir())
    labels = list((dest / "train" / "labels").iterdir())
    assert len(images) == 2
    assert len(labels) == 2

    # each label is one line: "<cls> <x> <y> <w> <h>"
    for lf in labels:
        lines = lf.read_text().strip().splitlines()
        assert len(lines) == 1
        parts = lines[0].split()
        assert len(parts) == 5
        cls = int(parts[0])
        coords = [float(p) for p in parts[1:]]
        assert cls in (0, 1)
        assert all(0.0 <= c <= 1.0 for c in coords)


def test_prepare_split_labels_match_classes(synthetic_dataset_root, tmp_path):
    dest = tmp_path / "yolo_data"
    prepare_split(synthetic_dataset_root, dest, split="train")

    # 열매_정상 = class 0, 열매_궤양병 = class 1
    # (as defined by CLASS_NAMES order)
    for lf in (dest / "train" / "labels").iterdir():
        cls = int(lf.read_text().split()[0])
        if "00FT" in lf.stem:  # normal fixture stem
            assert cls == CLASS_NAMES.index("normal")
        elif "01FT" in lf.stem:  # canker fixture stem
            assert cls == CLASS_NAMES.index("canker")


def test_prepare_all_creates_data_yaml(synthetic_dataset_root, tmp_path):
    dest = tmp_path / "yolo_data"
    summary = prepare_all(synthetic_dataset_root, dest)
    assert summary["train"] == 2
    assert summary["val"] == 2

    yaml_path = dest / "data.yaml"
    assert yaml_path.exists()
    doc = yaml.safe_load(yaml_path.read_text())
    assert doc["names"] == CLASS_NAMES
    assert doc["train"] == "train/images"
    assert doc["val"] == "val/images"
    assert Path(doc["path"]).resolve() == dest.resolve()


def test_prepare_split_skips_images_without_polygon(synthetic_dataset_root, tmp_path):
    """Synthetic fixture has 4 images per split total; only 2 have polygons."""
    dest = tmp_path / "yolo_data"
    n = prepare_split(synthetic_dataset_root, dest, split="val")
    assert n == 2
