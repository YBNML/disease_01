"""Image + mask transforms for segmentation training and validation.

Input is a BGR numpy image (H, W, 3) uint8 from cv2.imread and a (H, W) uint8
mask with values in {0: bg, 1: normal fruit, 2: canker fruit}. The pipeline
converts BGR→RGB (via albumentations), then applies geometric/photometric
transforms, and finally normalizes with ImageNet stats and stacks to tensor.
Mask is resized with nearest-neighbor interpolation to preserve class ids.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _bgr_to_rgb(image, **kw):
    """Convert BGR image to RGB (needed for pickle-compatibility with multiprocessing)."""
    return image[..., ::-1].copy()


def build_transforms(image_size: int = 512, train: bool = True):
    """Return an albumentations Compose accepting image=<uint8 HxWx3> and
    mask=<uint8 HxW>, producing image=<float32 3xHxW> and mask=<int64 HxW>."""
    base = [
        # AI Hub images are BGR from cv2; albumentations' geometry ops are
        # channel-agnostic, but Normalize expects values in the convention the
        # user supplies. We convert BGR→RGB first so ImageNet stats apply
        # correctly after normalization.
        A.Lambda(image=_bgr_to_rgb, mask=None),
        A.Resize(image_size, image_size, interpolation=1, mask_interpolation=0),
    ]
    if train:
        base += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ColorJitter(0.1, 0.1, 0.1, p=0.5),
        ]
    base += [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return A.Compose(base)
