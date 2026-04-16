import torch
from common.dataset import ClassificationDataset


def test_classification_dataset_train_size(synthetic_dataset_root):
    ds = ClassificationDataset(synthetic_dataset_root, split="train", transform=None)
    # 2 normal + 2 canker in Training
    assert len(ds) == 4


def test_classification_dataset_val_size(synthetic_dataset_root):
    ds = ClassificationDataset(synthetic_dataset_root, split="val", transform=None)
    assert len(ds) == 4  # same count in synthetic fixture


def test_classification_dataset_item_shape(synthetic_dataset_root):
    ds = ClassificationDataset(synthetic_dataset_root, split="train", transform=None)
    sample = ds[0]
    assert "image" in sample
    assert "label" in sample
    assert "metadata" in sample
    # label is 0 (normal) or 1 (canker)
    assert sample["label"] in (0, 1)
    # image from fixture is 64x64x3
    assert sample["image"].shape == (64, 64, 3)


def test_classification_dataset_labels_balanced(synthetic_dataset_root):
    ds = ClassificationDataset(synthetic_dataset_root, split="train", transform=None)
    labels = [ds[i]["label"] for i in range(len(ds))]
    assert labels.count(0) == 2
    assert labels.count(1) == 2


def test_classification_dataset_metadata_populated(synthetic_dataset_root):
    ds = ClassificationDataset(synthetic_dataset_root, split="train", transform=None)
    meta = ds[0]["metadata"]
    assert meta["camera"] == "samsung"
    assert "env" in meta
    assert meta["env"]["temp"] == 25.0


import numpy as np
from common.dataset import SegmentationDataset


def test_segmentation_dataset_filters_polygon_only(synthetic_dataset_root):
    ds = SegmentationDataset(synthetic_dataset_root, split="train", transform=None)
    # fixture has 1 normal + 1 canker with polygons (others null)
    assert len(ds) == 2


def test_segmentation_dataset_mask_shape_and_values(synthetic_dataset_root):
    ds = SegmentationDataset(synthetic_dataset_root, split="train", transform=None)
    sample = ds[0]
    assert sample["image"].shape == (64, 64, 3)
    assert sample["mask"].shape == (64, 64)
    assert sample["mask"].dtype == np.uint8
    # mask values must be in {0, 1, 2}: bg, normal fruit, canker fruit
    assert set(np.unique(sample["mask"]).tolist()).issubset({0, 1, 2})
    # something non-zero should exist (polygon was rasterized)
    assert sample["mask"].max() in (1, 2)


def test_segmentation_dataset_mask_class_matches_folder(synthetic_dataset_root):
    ds = SegmentationDataset(synthetic_dataset_root, split="train", transform=None)
    for i in range(len(ds)):
        s = ds[i]
        non_bg = np.unique(s["mask"][s["mask"] != 0]).tolist()
        # a single-class mask: either {1} for normal or {2} for canker
        assert non_bg in ([1], [2])
