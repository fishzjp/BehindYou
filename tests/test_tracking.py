"""Tests for tracking.py pure functions."""

import numpy as np

from behindyou.tracking import box_center, is_reasonable_shift, point_in_box, update_ema


def test_box_center():
    box = np.array([0, 0, 10, 20])
    cx, cy = box_center(box)
    assert cx == 5.0
    assert cy == 10.0


def test_box_center_float():
    box = np.array([1.0, 3.0, 5.0, 7.0])
    cx, cy = box_center(box)
    assert cx == 3.0
    assert cy == 5.0


def test_point_in_box_inside():
    box = np.array([0, 0, 100, 100])
    assert point_in_box((50, 50), box)


def test_point_in_box_outside():
    box = np.array([0, 0, 100, 100])
    assert not point_in_box((200, 200), box)


def test_point_in_box_margin():
    box = np.array([0, 0, 100, 100])
    # Box center is (50, 50), half_w = 100*1.3/2 = 65, so range is [-15, 115]
    assert point_in_box((110, 110), box, margin=0.3)
    # Point far outside even with margin
    assert not point_in_box((200, 200), box, margin=0.3)


def test_update_ema():
    old = np.array([0.0, 0.0, 100.0, 100.0])
    new = np.array([10.0, 10.0, 110.0, 110.0])
    result = update_ema(old, new, alpha=0.5)
    np.testing.assert_allclose(result, [5.0, 5.0, 105.0, 105.0])


def test_update_ema_alpha_one():
    old = np.array([0.0, 0.0, 100.0, 100.0])
    new = np.array([10.0, 10.0, 110.0, 110.0])
    result = update_ema(old, new, alpha=1.0)
    np.testing.assert_allclose(result, new)


def test_update_ema_alpha_zero():
    old = np.array([0.0, 0.0, 100.0, 100.0])
    new = np.array([10.0, 10.0, 110.0, 110.0])
    result = update_ema(old, new, alpha=0.0)
    np.testing.assert_allclose(result, old)


def test_is_reasonable_shift_small():
    old = np.array([0, 0, 100, 100])
    new = np.array([5, 5, 105, 105])
    assert is_reasonable_shift(old, new)


def test_is_reasonable_shift_large():
    old = np.array([0, 0, 100, 100])
    new = np.array([500, 500, 600, 600])
    assert not is_reasonable_shift(old, new)


def test_is_reasonable_shift_boundary():
    old = np.array([0, 0, 100, 100])
    # Shift of exactly 30% of width/height
    new = np.array([30, 30, 130, 130])
    assert is_reasonable_shift(old, new, max_shift=0.3)
