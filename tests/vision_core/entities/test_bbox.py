import math
import pytest
from vision_core.entities.bbox import BBox


def test_iou_partial_overlap():
    a = BBox(x_min=0, y_min=0, x_max=10, y_max=10)
    b = BBox(x_min=5, y_min=5, x_max=15, y_max=15)

    # intersection = 5*5 = 25, area each = 100, union = 175
    assert math.isclose(a.iou(b), 25.0 / 175.0, rel_tol=1e-9)


def test_iou_disjoint():
    a = BBox(x_min=0, y_min=0, x_max=10, y_max=10)
    b = BBox(x_min=20, y_min=20, x_max=30, y_max=30)
    assert a.iou(b) == 0.0


def test_iou_touching_edges():
    a = BBox(x_min=0, y_min=0, x_max=10, y_max=10)
    b = BBox(x_min=10, y_min=0, x_max=20, y_max=10)  # touch on vertical edge
    assert a.iou(b) == 0.0


def test_iou_identical_boxes():
    a = BBox(x_min=1.5, y_min=2.5, x_max=5.5, y_max=6.5)
    b = BBox(x_min=1.5, y_min=2.5, x_max=5.5, y_max=6.5)
    assert a.iou(b) == 1.0


def test_iou_degenerate_identical():
    # zero-area boxes at same point should be IoU == 1.0
    a = BBox(x_min=0.0, y_min=0.0, x_max=0.0, y_max=0.0)
    b = BBox(x_min=0.0, y_min=0.0, x_max=0.0, y_max=0.0)
    assert a.area == 0.0 and b.area == 0.0
    assert a.iou(b) == 1.0


def test_iou_degenerate_different():
    # zero-area boxes at different points -> IoU == 0.0
    a = BBox(x_min=0.0, y_min=0.0, x_max=0.0, y_max=0.0)
    b = BBox(x_min=1.0, y_min=1.0, x_max=1.0, y_max=1.0)
    assert a.iou(b) == 0.0


def test_inside_small_box():
    big = BBox(x_min=0, y_min=0, x_max=100, y_max=100)
    small = BBox(x_min=10, y_min=10, x_max=20, y_max=20)
    assert big.iou(small) == pytest.approx(100 / 10000)  # 0.01
    assert big.intersection_over_min(small) == pytest.approx(1.0)
    assert big.contains(small) is True
    assert big.contains_center(small) is True


def test_no_intersection():
    a = BBox(x_min=0, y_min=0, x_max=10, y_max=10)
    b = BBox(x_min=20, y_min=20, x_max=30, y_max=30)
    assert a.iou(b) == 0.0
    assert a.intersection_over_min(b) == 0.0
    assert a.contains(b) is False
    assert a.contains_center(b) is False


def test_degenerate_boxes():
    a = BBox(x_min=1, y_min=1, x_max=1, y_max=1)
    b = BBox(x_min=1, y_min=1, x_max=1, y_max=1)
    assert a.iou(b) == 1.0
    assert a.intersection_over_min(b) == 1.0


def test_from_tuple_valid_coords():
    coords = (10, 20, 100, 200)
    bbox = BBox.from_tuple(coords)
    assert bbox.x_min == 10
    assert bbox.y_min == 20
    assert bbox.x_max == 100
    assert bbox.y_max == 200


def test_from_tuple_zero_coords():
    coords = (0, 0, 0, 0)
    bbox = BBox.from_tuple(coords)
    assert bbox.x_min == 0
    assert bbox.y_min == 0
    assert bbox.x_max == 0
    assert bbox.y_max == 0


def test_from_tuple_negative_coords():
    coords = (-10, -20, 50, 60)
    bbox = BBox.from_tuple(coords)
    assert bbox.x_min == -10
    assert bbox.y_min == -20
    assert bbox.x_max == 50
    assert bbox.y_max == 60


def test_from_tuple_inverted_coords():
    # x_max < x_min, y_max < y_min
    coords = (100, 200, 10, 20)
    bbox = BBox.from_tuple(coords)
    assert bbox.x_min == 100
    assert bbox.y_min == 200
    assert bbox.x_max == 10
    assert bbox.y_max == 20
    assert bbox.width == 0.0  # protected by max(0.0, ...)
    assert bbox.height == 0.0
