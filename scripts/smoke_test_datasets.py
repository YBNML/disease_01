"""Smoke test: verify ClassificationDataset and SegmentationDataset work
against the real AI Hub data. Prints sizes and one sample from each."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.dataset import ClassificationDataset, SegmentationDataset

DB = ROOT / "database"

print("== ClassificationDataset ==")
for split in ("train", "val"):
    ds = ClassificationDataset(DB, split=split)
    print(f"  {split}: n={len(ds)}")
    s = ds[0]
    print(f"    sample image={s['image'].shape}, label={s['label']}, "
          f"camera={s['metadata']['camera']}")

print("\n== SegmentationDataset ==")
for split in ("train", "val"):
    ds = SegmentationDataset(DB, split=split)
    print(f"  {split}: n={len(ds)}")
    s = ds[0]
    import numpy as np
    print(f"    sample image={s['image'].shape}, mask={s['mask'].shape}, "
          f"unique={np.unique(s['mask']).tolist()}")

print("\nSmoke test OK.")
