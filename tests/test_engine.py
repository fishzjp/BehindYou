"""Tests for engine.py process_frame and helper functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv

from behindyou.config import Config
from behindyou.engine import (
    _LoopState,
    _classify_stranger,
    _scalar_iou,
    _update_self_track,
    annotate_frame,
    process_frame,
)
from behindyou.face import FaceCacheEntry, FaceInfo


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


def _make_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


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


def test_scalar_iou_contained_box():
    outer = np.array([0.0, 0.0, 100.0, 100.0])
    inner = np.array([25.0, 25.0, 75.0, 75.0])
    expected = 50.0 * 50.0 / (100.0 * 100.0)
    assert abs(_scalar_iou(outer, inner) - expected) < 1e-6


def test_scalar_iou_zero_area_box():
    a = np.array([50.0, 50.0, 50.0, 50.0])
    b = np.array([0.0, 0.0, 100.0, 100.0])
    assert _scalar_iou(a, b) == 0.0


def test_scalar_iou_overflow_clamped_to_one():
    a = np.array([0.0, 0.0, 100.0, 100.0])
    b = np.array([0.0, 0.0, 100.0, 100.0])
    assert _scalar_iou(a, b) <= 1.0


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


# --- _classify_stranger tests ---


def test_classify_stranger_no_face_check_returns_true():
    state = _make_state(config=Config(no_face_check=True))
    face_boxes: list[FaceInfo] = []
    assert _classify_stranger(_make_frame(), _bbox(0, 0, 100, 100), 2, state, face_boxes)


def test_classify_stranger_cached_embedding_returns_true():
    recognizer = MagicMock()
    recognizer.has_owner = True
    state = _make_state(
        config=Config(),
        face_recognizer=recognizer,
        face_cache={2: FaceCacheEntry(embedding=np.ones(512, dtype=np.float32))},
    )
    face_boxes: list[FaceInfo] = []
    assert _classify_stranger(_make_frame(), _bbox(0, 0, 100, 100), 2, state, face_boxes)


def test_classify_stranger_recognizer_finds_face():
    recognizer = MagicMock()
    recognizer.has_owner = False
    face_info = FaceInfo(bbox=_bbox(0, 0, 50, 50), score=0.9)
    recognizer.check_frontal_and_get_embedding.return_value = face_info
    state = _make_state(config=Config(), face_recognizer=recognizer, face_cache={2: FaceCacheEntry()})
    face_boxes: list[FaceInfo] = []
    assert _classify_stranger(_make_frame(), _bbox(0, 0, 100, 100), 2, state, face_boxes)
    assert face_boxes[0] is face_info


def test_classify_stranger_recognizer_caches_embedding():
    emb = np.random.randn(512).astype(np.float32)
    recognizer = MagicMock()
    recognizer.has_owner = True
    face_info = FaceInfo(bbox=_bbox(0, 0, 50, 50), score=0.9, embedding=emb)
    recognizer.check_frontal_and_get_embedding.return_value = face_info
    state = _make_state(config=Config(), face_recognizer=recognizer, face_cache={2: FaceCacheEntry()})
    face_boxes: list[FaceInfo] = []
    _classify_stranger(_make_frame(), _bbox(0, 0, 100, 100), 2, state, face_boxes)
    assert state.face_cache[2].embedding is not None
    assert np.array_equal(state.face_cache[2].embedding, emb)


def test_classify_stranger_recognizer_no_face_returns_false():
    recognizer = MagicMock()
    recognizer.has_owner = True
    recognizer.check_frontal_and_get_embedding.return_value = None
    state = _make_state(config=Config(), face_recognizer=recognizer, face_cache={2: FaceCacheEntry()})
    face_boxes: list[FaceInfo] = []
    assert not _classify_stranger(_make_frame(), _bbox(0, 0, 100, 100), 2, state, face_boxes)


def test_classify_stranger_haar_fallback_found():
    detector = MagicMock()
    detector.has_frontal_face.return_value = True
    state = _make_state(config=Config(), face_recognizer=None, face_detector=detector)
    face_boxes: list[FaceInfo] = []
    assert _classify_stranger(_make_frame(), _bbox(0, 0, 100, 100), 2, state, face_boxes)


def test_classify_stranger_haar_fallback_not_found():
    detector = MagicMock()
    detector.has_frontal_face.return_value = False
    state = _make_state(config=Config(), face_recognizer=None, face_detector=detector)
    face_boxes: list[FaceInfo] = []
    assert not _classify_stranger(_make_frame(), _bbox(0, 0, 100, 100), 2, state, face_boxes)


def test_classify_stranger_no_detector_no_recognizer_returns_false():
    state = _make_state(config=Config(), face_recognizer=None, face_detector=None)
    face_boxes: list[FaceInfo] = []
    assert not _classify_stranger(_make_frame(), _bbox(0, 0, 100, 100), 2, state, face_boxes)


# --- process_frame tests ---


def test_process_frame_no_tracker_ids():
    frame = _make_frame()
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
    frame = _make_frame()
    detections = _make_detections([[100, 100, 300, 300]], [1])
    state = _make_state()
    old_ema = state.ema_box.copy()
    process_frame(frame, detections, state)
    assert not np.array_equal(state.ema_box, old_ema)


def test_process_frame_small_box_filtered():
    frame = _make_frame()
    detections = _make_detections([[0, 0, 5, 5]], [99])
    state = _make_state(min_box_area=10000.0)
    intruders, _ = process_frame(frame, detections, state)
    assert intruders == []


def test_process_frame_stranger_detected_no_face_check():
    frame = _make_frame()
    detections = _make_detections([[100, 100, 300, 300], [400, 400, 600, 600]], [1, 2])
    config = Config(no_face_check=True, persistence=1)
    state = _make_state(config=config)
    intruders, _ = process_frame(frame, detections, state)
    assert len(intruders) == 1


def test_process_frame_stranger_filtered_by_persistence():
    frame = _make_frame()
    detections = _make_detections([[400, 400, 600, 600]], [2])
    config = Config(no_face_check=True, persistence=3)
    state = _make_state(config=config)
    intruders, _ = process_frame(frame, detections, state)
    assert len(intruders) == 0


def test_process_frame_stranger_after_persistence_reached():
    frame = _make_frame()
    detections = _make_detections([[400, 400, 600, 600]], [2])
    config = Config(no_face_check=True, persistence=2)
    state = _make_state(config=config)
    process_frame(frame, detections, state)
    process_frame(frame, detections, state)
    intruders, _ = process_frame(frame, detections, state)
    assert len(intruders) == 1


def test_process_frame_iou_self_adoption():
    frame = _make_frame()
    near_self_box = [105, 105, 305, 305]
    detections = _make_detections([near_self_box], [2])
    config = Config(no_face_check=True, persistence=1, self_iou_threshold=0.3)
    state = _make_state(config=config)
    intruders, _ = process_frame(frame, detections, state)
    assert len(intruders) == 0
    assert state.self_id == 2


def test_process_frame_face_recognizer_reidentifies_owner():
    frame = _make_frame()
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
    frame = _make_frame()
    detections1 = _make_detections([[400, 400, 600, 600]], [2])
    config = Config(no_face_check=True, persistence=1)
    state = _make_state(config=config)
    process_frame(frame, detections1, state)
    assert 2 in state.track_persistence

    detections2 = _make_detections([[400, 400, 600, 600]], [3])
    process_frame(frame, detections2, state)
    assert 2 not in state.track_persistence
    assert 3 in state.track_persistence


# --- process_frame edge cases ---


def test_process_frame_tracker_swap_ema_catches_self():
    state = _make_state(config=Config(no_face_check=True, persistence=1))
    # New track ID 5 at same EMA position
    detections = _make_detections([[105, 105, 305, 305]], [5])
    intruders, _ = process_frame(_make_frame(), detections, state)
    assert len(intruders) == 0
    assert state.self_id == 5


def test_process_frame_tracker_swap_face_reid():
    recognizer = MagicMock()
    recognizer.has_owner = True
    recognizer.get_cached_embedding.return_value = np.ones(512, dtype=np.float32)
    recognizer.is_owner.return_value = True
    state = _make_state(config=Config(persistence=1), face_recognizer=recognizer)
    # New track ID 5 far from EMA but face-matches owner
    detections = _make_detections([[400, 400, 600, 600]], [5])
    intruders, _ = process_frame(_make_frame(), detections, state)
    assert len(intruders) == 0
    assert state.self_id == 5


def test_process_frame_tracker_swap_no_face_false_alarm():
    recognizer = MagicMock()
    recognizer.has_owner = True
    recognizer.get_cached_embedding.return_value = None
    recognizer.check_frontal_and_get_embedding.return_value = None
    state = _make_state(config=Config(no_face_check=True, persistence=1), face_recognizer=recognizer)
    detections = _make_detections([[400, 400, 600, 600]], [5])
    intruders, _ = process_frame(_make_frame(), detections, state)
    assert len(intruders) == 1  # no_face_check=True, so triggers alarm


def test_process_frame_face_similarity_below_threshold():
    recognizer = MagicMock()
    recognizer.has_owner = True
    recognizer.get_cached_embedding.return_value = np.random.randn(512).astype(np.float32)
    recognizer.is_owner.return_value = False
    recognizer.check_frontal_and_get_embedding.return_value = FaceInfo(
        bbox=_bbox(400, 400, 500, 500), score=0.9, embedding=np.random.randn(512).astype(np.float32)
    )
    state = _make_state(config=Config(no_face_check=False, persistence=1), face_recognizer=recognizer)
    detections = _make_detections([[400, 400, 600, 600]], [2])
    intruders, _ = process_frame(_make_frame(), detections, state)
    assert len(intruders) == 1  # stranger correctly classified


@pytest.mark.parametrize("frames_seen,expected_intruders", [
    (0, 0),  # persistence=3, 0+1=1 < 3
    (1, 0),  # 1+1=2 < 3
    (2, 1),  # 2+1=3 >= 3
])
def test_process_frame_persistence_boundary(frames_seen, expected_intruders):
    config = Config(no_face_check=True, persistence=3)
    state = _make_state(config=config)
    frame = _make_frame()
    detections = _make_detections([[400, 400, 600, 600]], [2])
    for _ in range(frames_seen):
        process_frame(frame, detections, state)
    intruders, _ = process_frame(frame, detections, state)
    assert len(intruders) == expected_intruders


def test_process_frame_min_area_boundary():
    frame = _make_frame()
    # 640*480 = 307200, min_area=0.02 → min_box_area=6144.0
    # box 78*78 = 6084 < 6144, box 79*79 = 6241 > 6144
    state = _make_state(config=Config(no_face_check=True, persistence=1), min_box_area=6144.0)
    intruders_small, _ = process_frame(
        frame, _make_detections([[0, 0, 78, 78]], [2]), state
    )
    assert len(intruders_small) == 0

    state2 = _make_state(config=Config(no_face_check=True, persistence=1), min_box_area=6144.0)
    intruders_big, _ = process_frame(
        frame, _make_detections([[0, 0, 79, 79]], [2]), state2
    )
    assert len(intruders_big) == 1


def test_process_frame_iou_adoption_boundary():
    frame = _make_frame()
    config = Config(no_face_check=True, persistence=1, self_iou_threshold=0.3)
    # ema_box is [100,100,300,300], area=40000
    # Box with ~0.25 IoU: [0, 0, 200, 200] → inter=100*100=10000, union=40000+40000-10000=70000 → ~0.14
    # Box with ~0.33 IoU: [50, 50, 300, 300] → inter=250*250=62500... wait, let me recalculate
    # Better: [80, 80, 300, 300] → inter=220*220=48400, union=40000+48400-48400=40000 → 1.21... that's wrong
    # Let me use simpler math: ema=[100,100,300,300], box=[100,100,300,300] → IoU=1.0
    # box=[150,150,350,350] → inter=150*150=22500, union=40000+40000-22500=57500 → IoU=0.39
    state = _make_state(config=config)
    intruders, _ = process_frame(frame, _make_detections([[150, 150, 350, 350]], [2]), state)
    assert len(intruders) == 0  # IoU ~0.39 > 0.3, adopted as self
    assert state.self_id == 2


def test_process_frame_ema_max_skips_boundary():
    config = Config(no_face_check=True, persistence=1, ema_max_skips=3)
    state = _make_state(config=config, ema_skip_count=2)
    far_box = _bbox(500, 500, 700, 700)
    # 3rd unreasonable shift (skip_count becomes 3 >= max_skips=3)
    process_frame(_make_frame(), _make_detections([[far_box[0], far_box[1], far_box[2], far_box[3]]], [1]), state)
    assert state.ema_skip_count == 0  # force-adopted


# --- annotate_frame tests ---


def test_annotate_frame_returns_frame():
    frame = _make_frame()
    detections = _make_detections([[100, 100, 300, 300]], [1])
    result = annotate_frame(frame, detections, [])
    assert result is not None
    assert result.shape == frame.shape


def test_annotate_frame_with_intruders():
    frame = _make_frame()
    detections = _make_detections([[100, 100, 300, 300]], [1])
    intruder_boxes = [np.array([400, 400, 600, 600])]
    result = annotate_frame(frame, detections, intruder_boxes)
    assert result is not None


def test_annotate_frame_with_face_info():
    frame = _make_frame()
    detections = _make_detections([[100, 100, 300, 300]], [1])
    face_info = [FaceInfo(bbox=_bbox(120, 120, 180, 180), score=0.95)]
    result = annotate_frame(frame, detections, [], face_info)
    assert result is not None


def test_annotate_frame_empty_detections():
    frame = _make_frame()
    detections = sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        tracker_id=None,
        confidence=np.empty(0, dtype=np.float32),
    )
    result = annotate_frame(frame, detections, [])
    assert result is not None


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
        loaded, err = load_config()
        assert err is None
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
        loaded, err = load_config()
        assert loaded is None
        assert err is None
    finally:
        cfg_mod._CONFIG_FILE = original
