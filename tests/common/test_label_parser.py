import numpy as np
import pytest
from common.label_parser import parse_antn_pt


def test_parse_antn_pt_basic():
    s = "[10|20|30],[100|200|300]"
    pts = parse_antn_pt(s)
    assert pts.shape == (3, 2)
    assert pts.dtype == np.int32
    np.testing.assert_array_equal(pts, [[10, 100], [20, 200], [30, 300]])


def test_parse_antn_pt_single_point():
    s = "[5],[9]"
    pts = parse_antn_pt(s)
    assert pts.shape == (1, 2)
    np.testing.assert_array_equal(pts, [[5, 9]])


def test_parse_antn_pt_mismatched_lengths_raises():
    s = "[1|2|3],[10|20]"
    with pytest.raises(ValueError):
        parse_antn_pt(s)


from common.label_parser import polygon_to_bbox, polygon_to_mask


def test_polygon_to_bbox():
    poly = np.array([[10, 20], [30, 15], [25, 40]], dtype=np.int32)
    bbox = polygon_to_bbox(poly)
    assert bbox == (10, 15, 30, 40)


def test_polygon_to_mask_square():
    poly = np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=np.int32)
    mask = polygon_to_mask(poly, h=10, w=10)
    assert mask.shape == (10, 10)
    assert mask.dtype == np.uint8
    # interior pixel
    assert mask[4, 4] == 1
    # outside pixel
    assert mask[0, 0] == 0


def test_polygon_to_mask_empty_outside():
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int32)
    mask = polygon_to_mask(poly, h=5, w=5)
    assert mask.sum() >= 1
    assert mask.max() == 1
