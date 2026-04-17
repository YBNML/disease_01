import numpy as np
import pytest
from detection.yolo_format import polygon_to_yolo_bbox


def test_polygon_to_yolo_bbox_basic():
    # polygon covers a 40×40 square from (100,100) to (140,140)
    # in a 1000×1000 image
    poly = np.array([[100, 100], [140, 100], [140, 140], [100, 140]], dtype=np.int32)
    x, y, w, h = polygon_to_yolo_bbox(poly, img_w=1000, img_h=1000)
    # x_center = 120/1000 = 0.12
    assert abs(x - 0.12) < 1e-6
    # y_center = 120/1000 = 0.12
    assert abs(y - 0.12) < 1e-6
    # w = 40/1000 = 0.04
    assert abs(w - 0.04) < 1e-6
    # h = 40/1000 = 0.04
    assert abs(h - 0.04) < 1e-6


def test_polygon_to_yolo_bbox_non_square_image():
    poly = np.array([[0, 0], [960, 0], [960, 540], [0, 540]], dtype=np.int32)
    x, y, w, h = polygon_to_yolo_bbox(poly, img_w=1920, img_h=1080)
    # Full half of the image starting from top-left
    # bbox (0,0)-(960,540); x_center = 480/1920 = 0.25; y_center = 270/1080 = 0.25
    # w = 960/1920 = 0.5; h = 540/1080 = 0.5
    assert abs(x - 0.25) < 1e-6
    assert abs(y - 0.25) < 1e-6
    assert abs(w - 0.5) < 1e-6
    assert abs(h - 0.5) < 1e-6


def test_polygon_to_yolo_bbox_clamped_to_unit_interval():
    """Edge coords should stay within [0, 1]."""
    poly = np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], dtype=np.int32)
    x, y, w, h = polygon_to_yolo_bbox(poly, img_w=1920, img_h=1080)
    for v in (x, y, w, h):
        assert 0.0 <= v <= 1.0


def test_polygon_to_yolo_bbox_zero_dimension_raises():
    """A degenerate polygon should raise a clear error."""
    poly = np.array([[100, 100], [100, 100]], dtype=np.int32)  # single point
    with pytest.raises(ValueError):
        polygon_to_yolo_bbox(poly, img_w=1000, img_h=1000)
