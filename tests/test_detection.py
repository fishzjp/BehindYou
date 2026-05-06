from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from behindyou.detection import detect_people, reset_tracker


def _make_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_detect_people_returns_empty_on_exception():
    model = MagicMock()
    model.track.side_effect = RuntimeError("YOLO crash")
    result = detect_people(model, _make_frame(), 0.6)
    assert len(result) == 0


def test_detect_people_returns_empty_when_no_results():
    model = MagicMock()
    model.track.return_value = []
    result = detect_people(model, _make_frame(), 0.6)
    assert len(result) == 0


def test_reset_tracker_handles_missing_predictor():
    model = MagicMock(spec=[])  # no predictor attribute
    reset_tracker(model)  # should not raise
