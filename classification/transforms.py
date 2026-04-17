"""Image transforms for classification training and validation.

Input is a BGR numpy array (H, W, 3) uint8 from cv2.imread. The pipeline
converts BGR→RGB, then applies torchvision transforms to produce a normalized
float tensor of shape (3, image_size, image_size).
"""
from typing import Callable
import cv2
import numpy as np
import torchvision.transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_transforms(image_size: int = 224, train: bool = True) -> Callable:
    """Return a callable that converts a BGR uint8 numpy image into a
    normalized float tensor of shape (3, image_size, image_size)."""
    resize_short = int(round(image_size * 256 / 224))  # keep 256/224 ratio
    if train:
        pipeline = T.Compose([
            T.Lambda(_bgr_to_rgb),
            T.ToPILImage(),
            T.Resize(resize_short),
            T.CenterCrop(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.ColorJitter(0.1, 0.1, 0.1),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        pipeline = T.Compose([
            T.Lambda(_bgr_to_rgb),
            T.ToPILImage(),
            T.Resize(resize_short),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return pipeline
