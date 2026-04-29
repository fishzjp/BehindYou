from __future__ import annotations

import numpy as np


def box_center(box: np.ndarray) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def point_in_box(center: tuple[float, float], box: np.ndarray, margin: float = 0.3) -> bool:
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx, cy = box_center(box)
    half_w = w * (1 + margin) / 2
    half_h = h * (1 + margin) / 2
    return abs(center[0] - cx) <= half_w and abs(center[1] - cy) <= half_h


def update_ema(ema: np.ndarray, new_val: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    return alpha * new_val + (1 - alpha) * ema


def is_reasonable_shift(old_box: np.ndarray, new_box: np.ndarray, max_shift: float = 0.3) -> bool:
    old_cx, old_cy = box_center(old_box)
    new_cx, new_cy = box_center(new_box)
    w = old_box[2] - old_box[0]
    h = old_box[3] - old_box[1]
    dx = abs(new_cx - old_cx) / max(w, 1)
    dy = abs(new_cy - old_cy) / max(h, 1)
    return dx <= max_shift and dy <= max_shift


def crop_upper_body(frame: np.ndarray, bbox: np.ndarray, crop_ratio: float = 0.55) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop_bottom = min(y1 + int((y2 - y1) * crop_ratio), h)
    return frame[max(0, y1):crop_bottom, max(0, x1):min(x2, w)]
