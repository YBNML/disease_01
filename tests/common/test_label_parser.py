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
