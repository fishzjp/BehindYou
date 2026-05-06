from __future__ import annotations

import dataclasses
import json
import logging

from behindyou.paths import DATA_DIR

logger = logging.getLogger(__name__)

_CONFIG_FILE = DATA_DIR / "config.json"


@dataclasses.dataclass(frozen=True)
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
    face_max_yaw: float = 45.0

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
            raise ValueError(
                f"face_match_threshold must be in (0, 1], got {self.face_match_threshold}"
            )
        if not 0.0 < self.self_iou_threshold <= 1.0:
            raise ValueError(f"self_iou_threshold must be in (0, 1], got {self.self_iou_threshold}")
        if not 0.0 < self.face_det_score <= 1.0:
            raise ValueError(f"face_det_score must be in (0, 1], got {self.face_det_score}")
        if not 0.0 < self.face_min_size <= 1.0:
            raise ValueError(f"face_min_size must be in (0, 1], got {self.face_min_size}")
        if self.face_retry_interval < 1:
            raise ValueError(f"face_retry_interval must be >= 1, got {self.face_retry_interval}")
        if self.ema_max_skips < 1:
            raise ValueError(f"ema_max_skips must be >= 1, got {self.ema_max_skips}")
        if self.camera < 0:
            raise ValueError(f"camera must be >= 0, got {self.camera}")
        if not 0.0 < self.face_max_yaw <= 90.0:
            raise ValueError(f"face_max_yaw must be in (0, 90], got {self.face_max_yaw}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def save_config(cfg: Config) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(json.dumps(cfg.to_dict(), indent=2))


def load_config() -> Config | None:
    try:
        d = json.loads(_CONFIG_FILE.read_text())
        return Config.from_dict(d)
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
        return None
