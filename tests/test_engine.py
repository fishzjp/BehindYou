"""Tests for engine.py process_frame and helper functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import supervision as sv

from behindyou.config import Config
from behindyou.engine import (
    _LoopState,
    _scalar_iou,
    _update_self_track,
    process_frame,
)


def _make_state(**overrides) -> _LoopState:
    defaults = dict(
        config=Config(),
        self_id=1,
        ema_box=np.array([100.0, 100.0, 300.0, 300.0]),
        face_detector=None,
        face_recognizer=None,
        min_box_area=1000.0,
    )
    defaults.update(overrides)
    return _LoopState(**defaults)


def _make_detections(xyxy_list: list[list[float]], tracker_ids: list[int]) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array(xyxy_list, dtype=np.float32),
        tracker_id=np.array(tracker_ids, dtype=int),
        confidence=np.ones(len(tracker_ids), dtype=np.float32),
        class_id=np.zeros(len(tracker_ids), dtype=int),
    )


def _bbox(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([x1, y1, x2, y2])


# --- _scalar_iou tests ---


def test_scalar_iou_identical():
    box = np.array([0.0, 0.0, 100.0, 100.0])
    assert _scalar_iou(box, box) > 0.99


def test_scalar_iou_no_overlap():
    a = np.array([0.0, 0.0, 50.0, 50.0])
    b = np.array([100.0, 100.0, 150.0, 150.0])
    assert _scalar_iou(a, b) == 0.0


def test_scalar_iou_partial():
    a = np.array([0.0, 0.0, 100.0, 100.0])
    b = np.array([50.0, 50.0, 150.0, 150.0])
    result = _scalar_iou(a, b)
    assert 0.0 < result < 1.0


# --- _update_self_track tests ---


def test_update_self_track_reasonable_shift():
    state = _make_state()
    new_box = _bbox(110, 110, 310, 310)
    _update_self_track(state, 1, new_box)
    assert state.ema_skip_count == 0


def test_update_self_track_unreasonable_shift_increments_skip():
    state = _make_state()
    new_box = _bbox(500, 500, 700, 700)
    _update_self_track(state, 1, new_box)
    assert state.ema_skip_count == 1


def test_update_self_track_max_skips_adopt():
    state = _make_state(ema_skip_count=9)
    new_box = _bbox(500, 500, 700, 700)
    _update_self_track(state, 1, new_box)
    assert state.self_id == 1
    assert state.ema_skip_count == 0


def test_update_self_track_new_tid():
    state = _make_state()
    new_box = _bbox(110, 110, 310, 310)
    _update_self_track(state, 5, new_box)
    assert state.self_id == 5


# --- process_frame tests ---


def test_process_frame_no_tracker_ids():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        tracker_id=None,
        confidence=np.empty(0, dtype=np.float32),
    )
    state = _make_state()
    intruders, faces = process_frame(frame, detections, state)
    assert intruders == []
    assert faces == []


def test_process_frame_self_track_updates_ema():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = _make_detections([[100, 100, 300, 300]], [1])
    state = _make_state()
    old_ema = state.ema_box.copy()
    process_frame(frame, detections, state)
    assert not np.array_equal(state.ema_box, old_ema)


def test_process_frame_small_box_filtered():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = _make_detections([[0, 0, 5, 5]], [99])
    state = _make_state(min_box_area=10000.0)
    intruders, _ = process_frame(frame, detections, state)
    assert intruders == []


def test_process_frame_stranger_detected_no_face_check():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = _make_detections([[100, 100, 300, 300], [400, 400, 600, 600]], [1, 2])
    config = Config(no_face_check=True, persistence=1)
    state = _make_state(config=config)
    intruders, _ = process_frame(frame, detections, state)
    assert len(intruders) == 1


def test_process_frame_stranger_filtered_by_persistence():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = _make_detections([[400, 400, 600, 600]], [2])
    config = Config(no_face_check=True, persistence=3)
    state = _make_state(config=config)
    intruders, _ = process_frame(frame, detections, state)
    assert len(intruders) == 0


def test_process_frame_stranger_after_persistence_reached():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = _make_detections([[400, 400, 600, 600]], [2])
    config = Config(no_face_check=True, persistence=2)
    state = _make_state(config=config)
    process_frame(frame, detections, state)
    process_frame(frame, detections, state)
    intruders, _ = process_frame(frame, detections, state)
    assert len(intruders) == 1


def test_process_frame_iou_self_adoption():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    near_self_box = [105, 105, 305, 305]
    detections = _make_detections([near_self_box], [2])
    config = Config(no_face_check=True, persistence=1, self_iou_threshold=0.3)
    state = _make_state(config=config)
    intruders, _ = process_frame(frame, detections, state)
    assert len(intruders) == 0
    assert state.self_id == 2


def test_process_frame_face_recognizer_reidentifies_owner():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = _make_detections([[400, 400, 600, 600]], [2])

    recognizer = MagicMock()
    embedding = np.random.randn(512).astype(np.float32)
    recognizer.has_owner = True
    recognizer.get_cached_embedding.return_value = embedding
    recognizer.is_owner.return_value = True

    config = Config(no_face_check=True, persistence=1)
    state = _make_state(config=config, face_recognizer=recognizer)

    process_frame(frame, detections, state)
    assert state.self_id == 2


def test_process_frame_tracks_cleaned_up():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections1 = _make_detections([[400, 400, 600, 600]], [2])
    config = Config(no_face_check=True, persistence=1)
    state = _make_state(config=config)
    process_frame(frame, detections1, state)
    assert 2 in state.track_persistence

    detections2 = _make_detections([[400, 400, 600, 600]], [3])
    process_frame(frame, detections2, state)
    assert 2 not in state.track_persistence
    assert 3 in state.track_persistence


# --- Config persistence tests ---


def test_config_roundtrip():
    from behindyou.config import Config as _C

    cfg = _C(camera=2, confidence=0.8, cooldown=5.0)
    d = cfg.to_dict()
    assert d["camera"] == 2
    assert d["confidence"] == 0.8
    restored = _C.from_dict(d)
    assert restored == cfg


def test_config_from_dict_ignores_unknown_keys():
    from behindyou.config import Config as _C

    d = {"camera": 1, "bogus": 999}
    cfg = _C.from_dict(d)
    assert cfg.camera == 1


def test_config_save_load(tmp_path):
    from behindyou.config import Config, save_config, load_config
    import behindyou.config as cfg_mod

    original = cfg_mod._CONFIG_FILE
    cfg_mod._CONFIG_FILE = tmp_path / "config.json"
    cfg_mod.DATA_DIR = tmp_path
    try:
        save_config(Config(camera=3, confidence=0.75))
        loaded = load_config()
        assert loaded is not None
        assert loaded.camera == 3
        assert loaded.confidence == 0.75
    finally:
        cfg_mod._CONFIG_FILE = original
        cfg_mod.DATA_DIR = original.parent


def test_load_config_missing_file(tmp_path):
    from behindyou.config import load_config
    import behindyou.config as cfg_mod

    original = cfg_mod._CONFIG_FILE
    cfg_mod._CONFIG_FILE = tmp_path / "nonexistent.json"
    try:
        assert load_config() is None
    finally:
        cfg_mod._CONFIG_FILE = original
