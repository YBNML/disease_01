from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset

import numpy as np
from .label_parser import load_sample, polygon_to_mask


CLASS_NAME_TO_LABEL = {"감귤_정상": 0, "감귤_궤양병": 1}

SPLIT_DIRS = {
    "train": {
        "split": "1.Training",
        "img": "원천데이터/TS1.감귤",
        "lbl": "라벨링데이터/TL1.감귤",
    },
    "val": {
        "split": "2.Validation",
        "img": "원천데이터/VS1.감귤",
        "lbl": "라벨링데이터/VL1.감귤",
    },
}

CLASS_DIRS = ["열매_정상", "열매_궤양병"]


def _iter_label_files(database_root, split: str):
    """Yield (image_path, json_path) pairs for a given split across both classes."""
    root = Path(database_root)
    cfg = SPLIT_DIRS[split]
    split_dir = root / cfg["split"]
    for cls_dir in CLASS_DIRS:
        img_dir = split_dir / cfg["img"] / cls_dir
        lbl_dir = split_dir / cfg["lbl"] / cls_dir
        if not lbl_dir.exists():
            continue
        for jp in sorted(lbl_dir.iterdir()):
            if jp.suffix.lower() != ".json":
                continue
            stem = jp.stem
            # image filename may have any extension; take first match
            cands = list(img_dir.glob(f"{stem}.*"))
            if not cands:
                continue
            yield cands[0], jp


class ClassificationDataset(Dataset):
    """All images in the split, labeled 0 (정상) / 1 (궤양병) from folder names.
    Loads images as BGR numpy arrays (H, W, 3) uint8. Transform (if given) is
    called with the numpy image and may return anything (tensor/array)."""

    def __init__(self, database_root, split: str = "train", transform=None):
        if split not in SPLIT_DIRS:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.database_root = Path(database_root)
        self.split = split
        self.transform = transform
        self.items = list(_iter_label_files(self.database_root, split))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        img_path, json_path = self.items[idx]
        info = load_sample(json_path)
        label = CLASS_NAME_TO_LABEL[info["class_code"]]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {img_path}")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "metadata": info["metadata"],
            "image_path": str(img_path),
        }


SEG_CLASS_FROM_CODE = {"감귤_정상": 1, "감귤_궤양병": 2}


class SegmentationDataset(Dataset):
    """Only images that have polygon labels. Produces a 3-class semantic mask:
    0 = background, 1 = 정상 감귤, 2 = 궤양병 감귤.
    Transform (if given) is called as transform(image, mask) and should return a
    dict {'image': ..., 'mask': ...} (albumentations style)."""

    def __init__(self, database_root, split: str = "train", transform=None):
        if split not in SPLIT_DIRS:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.database_root = Path(database_root)
        self.split = split
        self.transform = transform
        self.items = [
            (ip, jp) for ip, jp in _iter_label_files(self.database_root, split)
            if load_sample(jp)["has_polygon"]
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        img_path, json_path = self.items[idx]
        info = load_sample(json_path)
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {img_path}")
        h, w = image.shape[:2]

        cls_value = SEG_CLASS_FROM_CODE[info["class_code"]]
        binary = polygon_to_mask(info["polygon"], h=h, w=w)  # 0/1
        mask = (binary * cls_value).astype(np.uint8)         # 0 or cls_value

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        return {
            "image": image,
            "mask": mask,
            "metadata": info["metadata"],
            "image_path": str(img_path),
        }
