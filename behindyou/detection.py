from __future__ import annotations

from typing import TYPE_CHECKING

import supervision as sv
from ultralytics import YOLO

if TYPE_CHECKING:
    import numpy as np


def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def detect_people(model: YOLO, frame: np.ndarray, confidence: float) -> sv.Detections:
    results = model.track(frame, conf=confidence, classes=[0], verbose=False, persist=True)
    if not results:
        return sv.Detections.empty()
    return sv.Detections.from_ultralytics(results[0])


def reset_tracker(model: YOLO) -> None:
    model.predictor = None
