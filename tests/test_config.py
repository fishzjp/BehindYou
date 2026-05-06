"""Tests for Config validation."""

import pytest

from behindyou.config import Config


def test_default_config_valid():
    Config()  # should not raise


def test_invalid_confidence_zero():
    with pytest.raises(ValueError, match="confidence"):
        Config(confidence=0.0)


def test_invalid_confidence_negative():
    with pytest.raises(ValueError, match="confidence"):
        Config(confidence=-0.5)


def test_invalid_confidence_above_one():
    with pytest.raises(ValueError, match="confidence"):
        Config(confidence=1.5)


def test_valid_confidence_boundary():
    Config(confidence=1.0)  # should not raise


def test_invalid_cooldown_negative():
    with pytest.raises(ValueError, match="cooldown"):
        Config(cooldown=-1)


def test_valid_cooldown_zero():
    Config(cooldown=0)  # should not raise


def test_invalid_persistence_zero():
    with pytest.raises(ValueError, match="persistence"):
        Config(persistence=0)


def test_invalid_min_area_zero():
    with pytest.raises(ValueError, match="min_area"):
        Config(min_area=0.0)


def test_invalid_min_area_one():
    with pytest.raises(ValueError, match="min_area"):
        Config(min_area=1.0)


def test_invalid_ema_alpha_zero():
    with pytest.raises(ValueError, match="ema_alpha"):
        Config(ema_alpha=0.0)


def test_invalid_ema_alpha_above_one():
    with pytest.raises(ValueError, match="ema_alpha"):
        Config(ema_alpha=1.5)


def test_invalid_self_iou_threshold():
    with pytest.raises(ValueError, match="self_iou_threshold"):
        Config(self_iou_threshold=0.0)


def test_invalid_face_match_threshold():
    with pytest.raises(ValueError, match="face_match_threshold"):
        Config(face_match_threshold=0.0)


def test_face_max_yaw_default():
    cfg = Config()
    assert cfg.face_max_yaw == 45.0


def test_face_max_yaw_validation():
    with pytest.raises(ValueError, match="face_max_yaw"):
        Config(face_max_yaw=0.0)
    with pytest.raises(ValueError, match="face_max_yaw"):
        Config(face_max_yaw=91.0)
    Config(face_max_yaw=90.0)  # boundary OK
