from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import supervision as sv
from ultralytics import YOLO

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def detect_people(model: YOLO, frame: np.ndarray, confidence: float) -> sv.Detections:
    try:
        results = model.track(frame, conf=confidence, classes=[0], verbose=False, persist=True)
    except Exception:
        logger.warning("YOLO track 异常", exc_info=True)
        reset_tracker(model)
        return sv.Detections.empty()
    if not results:
        return sv.Detections.empty()
    return sv.Detections.from_ultralytics(results[0])


def reset_tracker(model: YOLO) -> None:
    # ultralytics 内部 API：重置 ByteTrack 状态
    try:
        model.predictor = None
    except AttributeError:
        pass
