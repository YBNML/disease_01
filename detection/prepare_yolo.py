"""Convert AI Hub citrus polygon labels into YOLO training format.

Usage (as a script):
    python detection/prepare_yolo.py --source database --dest detection/data

Produces:
    <dest>/train/images/*.jpg
    <dest>/train/labels/*.txt
    <dest>/val/images/*.jpg
    <dest>/val/labels/*.txt
    <dest>/data.yaml

Only images that have polygon annotations are copied (~787 out of 3834).
Each YOLO label file contains one line (one box per image):
    <class_id> <x_center> <y_center> <w> <h>   # all normalized to [0, 1]
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import yaml

from common.dataset import SPLIT_DIRS, _iter_label_files
from common.label_parser import load_sample
from detection.yolo_format import polygon_to_yolo_bbox


# Order defines class_id: index 0 → 정상, index 1 → 궤양병
CLASS_NAMES = ["normal", "canker"]
CLASS_CODE_TO_ID = {"감귤_정상": 0, "감귤_궤양병": 1}


def prepare_split(database_root, dest_root, split: str) -> int:
    """Build <dest_root>/<split>/{images,labels}/ from polygon-labeled samples.
    Returns the number of samples written."""
    database_root = Path(database_root)
    dest_root = Path(dest_root)
    img_dir = dest_root / split / "images"
    lbl_dir = dest_root / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for img_path, json_path in _iter_label_files(database_root, split):
        info = load_sample(json_path)
        if not info["has_polygon"]:
            continue
        img_w, img_h = info["image_size"]
        if img_w <= 0 or img_h <= 0:
            continue
        try:
            xc, yc, w, h = polygon_to_yolo_bbox(info["polygon"], img_w, img_h)
        except ValueError:
            continue  # skip degenerate polygons

        cls_id = CLASS_CODE_TO_ID[info["class_code"]]
        stem = json_path.stem
        # copy image using the original extension
        shutil.copy2(img_path, img_dir / img_path.name)
        (lbl_dir / f"{stem}.txt").write_text(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        n += 1
    return n


def prepare_all(database_root, dest_root) -> dict:
    """Build both train and val splits, plus data.yaml."""
    dest_root = Path(dest_root)
    summary = {}
    for split in ("train", "val"):
        summary[split] = prepare_split(database_root, dest_root, split)

    yaml_path = dest_root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(dest_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": CLASS_NAMES,
    }, sort_keys=False))
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="AI Hub database root")
    p.add_argument("--dest", required=True, help="Output YOLO data directory")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = prepare_all(args.source, args.dest)
    print(f"train: {summary['train']}  val: {summary['val']}")
    print(f"wrote {Path(args.dest) / 'data.yaml'}")
