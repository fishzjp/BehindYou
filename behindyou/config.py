from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    camera: int = 0
    confidence: float = 0.6
    cooldown: float = 10.0
    persistence: int = 3
    min_area: float = 0.02
    face_min_size: float = 0.15
    no_face_check: bool = False
    recalibrate: bool = False
    ema_alpha: float = 0.15
    ema_max_shift: float = 0.3
    ema_max_skips: int = 10
    face_crop_ratio: float = 0.55
    face_match_threshold: float = 0.55
    face_retry_interval: int = 15
    self_iou_threshold: float = 0.3
    face_det_score: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError(f"confidence must be in (0, 1], got {self.confidence}")
        if self.cooldown < 0:
            raise ValueError(f"cooldown must be >= 0, got {self.cooldown}")
        if self.persistence < 1:
            raise ValueError(f"persistence must be >= 1, got {self.persistence}")
        if not 0.0 < self.min_area < 1.0:
            raise ValueError(f"min_area must be in (0, 1), got {self.min_area}")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {self.ema_alpha}")
        if not 0.0 < self.ema_max_shift <= 1.0:
            raise ValueError(f"ema_max_shift must be in (0, 1], got {self.ema_max_shift}")
        if not 0.0 < self.face_crop_ratio <= 1.0:
            raise ValueError(f"face_crop_ratio must be in (0, 1], got {self.face_crop_ratio}")
        if not 0.0 < self.face_match_threshold <= 1.0:
            raise ValueError(f"face_match_threshold must be in (0, 1], got {self.face_match_threshold}")
        if not 0.0 < self.self_iou_threshold <= 1.0:
            raise ValueError(f"self_iou_threshold must be in (0, 1], got {self.self_iou_threshold}")
        if not 0.0 < self.face_det_score <= 1.0:
            raise ValueError(f"face_det_score must be in (0, 1], got {self.face_det_score}")
