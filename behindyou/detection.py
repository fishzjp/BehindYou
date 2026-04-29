from __future__ import annotations

from typing import TYPE_CHECKING

from ultralytics import YOLO

if TYPE_CHECKING:
    import numpy as np
    from ultralytics.engine.results import Results


def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def detect_people(model: YOLO, frame: np.ndarray, confidence: float) -> Results:
    return model.track(frame, conf=confidence, classes=[0], verbose=False, persist=True)
