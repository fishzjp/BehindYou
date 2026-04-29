from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    camera: int = 0
    confidence: float = 0.6
    cooldown: float = 10.0
    persistence: int = 3
    min_area: float = 0.02
    face_min_size: float = 0.15
    no_preview: bool = False
    no_face_check: bool = False
    recalibrate: bool = False
    ema_alpha: float = 0.15
    ema_max_shift: float = 0.3
    ema_max_skips: int = 10
    face_crop_ratio: float = 0.55
    face_match_threshold: float = 0.5
    face_retry_interval: int = 15
    self_iou_threshold: float = 0.3


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="BehindYou - 身后人员检测系统")
    parser.add_argument("--cooldown", type=float, default=10, help="通知冷却时间（秒），默认 10")
    parser.add_argument("--confidence", type=float, default=0.6, help="检测置信度阈值，默认 0.6")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引，默认 0")
    parser.add_argument("--no-preview", action="store_true", help="隐藏预览窗口")
    parser.add_argument("--persistence", type=int, default=3, help="帧持久性阈值，连续 N 帧检测到才报警，默认 3")
    parser.add_argument("--min-area", type=float, default=0.02, help="最小检测框面积占比（相对画面），默认 0.02")
    parser.add_argument("--face-min-size", type=float, default=0.15, help="人脸最小尺寸比例（相对裁剪区域），默认 0.15")
    parser.add_argument("--face-match-threshold", type=float, default=0.5, help="人脸匹配相似度阈值，默认 0.5")
    parser.add_argument("--ema-alpha", type=float, default=0.15, help="EMA 平滑系数，默认 0.15")
    parser.add_argument("--self-iou-threshold", type=float, default=0.3, help="自身 IoU 匹配阈值，默认 0.3")
    parser.add_argument("--ema-max-shift", type=float, default=0.3, help="EMA 最大偏移比例，默认 0.3")
    parser.add_argument("--ema-max-skips", type=int, default=10, help="EMA 最大跳过帧数，默认 10")
    parser.add_argument("--face-crop-ratio", type=float, default=0.55, help="人脸裁剪区域比例，默认 0.55")
    parser.add_argument("--face-retry-interval", type=int, default=15, help="人脸重试间隔帧数，默认 15")
    parser.add_argument("--no-face-check", action="store_true", help="禁用面部检测，任何人靠近都告警")
    parser.add_argument("--recalibrate", action="store_true", help="强制重新采集人脸数据")
    args = parser.parse_args()

    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if args.cooldown < 0:
        parser.error("--cooldown must be non-negative")
    if args.persistence < 1:
        parser.error("--persistence must be at least 1")
    if args.min_area <= 0:
        parser.error("--min-area must be positive")
    if not 0 < args.face_crop_ratio <= 1:
        parser.error("--face-crop-ratio must be between 0 and 1")
    if args.face_retry_interval < 1:
        parser.error("--face-retry-interval must be at least 1")

    return Config(
        camera=args.camera,
        confidence=args.confidence,
        cooldown=args.cooldown,
        persistence=args.persistence,
        min_area=args.min_area,
        face_min_size=args.face_min_size,
        face_match_threshold=args.face_match_threshold,
        ema_alpha=args.ema_alpha,
        ema_max_shift=args.ema_max_shift,
        ema_max_skips=args.ema_max_skips,
        face_crop_ratio=args.face_crop_ratio,
        face_retry_interval=args.face_retry_interval,
        self_iou_threshold=args.self_iou_threshold,
        no_preview=args.no_preview,
        no_face_check=args.no_face_check,
        recalibrate=args.recalibrate,
    )
