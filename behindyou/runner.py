from __future__ import annotations

import dataclasses
import logging
import os
import time

import cv2
import numpy as np

from behindyou.calibration import calibrate
from behindyou.config import Config
from behindyou.detection import detect_people, load_model
from behindyou.face import FaceDetector, FaceRecognizer
from behindyou.notification import send_notification
from behindyou.tracking import (
    box_center,
    get_track_id,
    is_reasonable_shift,
    point_in_box,
    update_ema,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(config: Config) -> None:
    model_path = os.path.join(PROJECT_ROOT, "yolo26n.pt")
    if not os.path.exists(model_path):
        logger.error("模型文件不存在：%s", model_path)
        logger.error("请从 https://github.com/ultralytics/assets/releases 下载 yolo26n.pt")
        return

    try:
        model = load_model(model_path)
    except Exception as e:
        logger.error("无法加载模型文件：%s", e)
        return

    face_detector: FaceDetector | None = None
    if not config.no_face_check:
        try:
            face_detector = FaceDetector()
        except RuntimeError as e:
            logger.warning("人脸检测器初始化失败（%s），将禁用人脸过滤", e)
            config = dataclasses.replace(config, no_face_check=True)

    cap = cv2.VideoCapture(config.camera)
    if not cap.isOpened():
        logger.error("无法打开摄像头 (index=%d)", config.camera)
        return

    face_recognizer: FaceRecognizer | None = None
    try:
        face_recognizer = FaceRecognizer()
        logger.info("人脸识别模型已加载")
        if not config.recalibrate:
            face_recognizer.load_embedding()
    except Exception as e:
        logger.warning("人脸识别初始化失败（%s），主人重入识别将不可用", e)

    has_saved_embedding = face_recognizer is not None and face_recognizer.owner_embedding is not None
    self_id, ema_box = calibrate(model, cap, config, face_recognizer, quick=has_saved_embedding)
    if self_id is None or ema_box is None:
        cap.release()
        return

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    min_box_area = frame_h * frame_w * config.min_area

    state = _LoopState(
        config=config,
        self_id=self_id,
        ema_box=ema_box,
        face_detector=face_detector,
        face_recognizer=face_recognizer,
        min_box_area=min_box_area,
    )

    logger.info("监控已启动，按 q 退出...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("读取摄像头画面失败")
                break

            results = detect_people(model, frame, config.confidence)
            state.frame_count += 1

            intruder_boxes = _process_frame(frame, results, state)

            if not config.no_preview:
                if _render_preview(results, intruder_boxes):
                    break

            _notify_if_needed(intruder_boxes, state)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("检测已停止")


@dataclasses.dataclass
class _LoopState:
    config: Config
    self_id: int
    ema_box: np.ndarray
    face_detector: FaceDetector | None
    face_recognizer: FaceRecognizer | None
    min_box_area: float
    frame_count: int = 0
    ema_skip_count: int = 0
    last_notify_time: float = 0.0
    track_persistence: dict[int, int] = dataclasses.field(default_factory=dict)
    face_cache: dict[int, np.ndarray | None] = dataclasses.field(default_factory=dict)
    face_retry_frame: dict[int, int] = dataclasses.field(default_factory=dict)


def _process_frame(
    frame: np.ndarray,
    results: object,
    state: _LoopState,
) -> list[np.ndarray]:
    config = state.config
    intruder_boxes: list[np.ndarray] = []
    current_tracks: set[int] = set()

    for box in results[0].boxes:
        xyxy_f = box.xyxy[0].cpu().numpy()
        xyxy = xyxy_f.astype(int)
        box_area = (xyxy_f[2] - xyxy_f[0]) * (xyxy_f[3] - xyxy_f[1])
        if box_area < state.min_box_area:
            continue

        tid = get_track_id(box)
        if tid is None:
            continue
        current_tracks.add(tid)

        is_self = tid == state.self_id or point_in_box(box_center(xyxy_f), state.ema_box)
        if not is_self and state.face_recognizer is not None and state.face_recognizer.owner_embedding is not None:
            emb = state.face_recognizer.get_cached_embedding(
                frame, xyxy_f, tid, state.face_cache, state.face_retry_frame,
                state.frame_count, config.face_retry_interval,
            )
            if emb is not None and state.face_recognizer.is_owner(emb, config.face_match_threshold):
                is_self = True
                state.self_id = tid
                state.ema_box = xyxy_f.copy()
                state.ema_skip_count = 0
                state.track_persistence.pop(tid, None)
                logger.info("人脸匹配成功：主人重入，新 ID=%d", tid)

        if is_self:
            if is_reasonable_shift(state.ema_box, xyxy_f, config.ema_max_shift):
                state.ema_box = update_ema(state.ema_box, xyxy_f, config.ema_alpha)
                state.ema_skip_count = 0
                if tid != state.self_id:
                    state.self_id = tid
            else:
                state.ema_skip_count += 1
                if state.ema_skip_count >= config.ema_max_skips:
                    state.ema_box = xyxy_f
                    state.self_id = tid
                    state.ema_skip_count = 0
            continue

        state.track_persistence[tid] = state.track_persistence.get(tid, 0) + 1
        if state.track_persistence[tid] >= config.persistence:
            if state.face_detector is None or state.face_detector.has_frontal_face(frame, xyxy_f, config.face_min_size):
                intruder_boxes.append(xyxy)

    for tid in list(state.track_persistence):
        if tid not in current_tracks:
            del state.track_persistence[tid]
            state.face_cache.pop(tid, None)
            state.face_retry_frame.pop(tid, None)

    return intruder_boxes


def _render_preview(results: object, intruder_boxes: list[np.ndarray]) -> bool:
    annotated = results[0].plot()
    for x1, y1, x2, y2 in intruder_boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(annotated, "INTRUDER", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("BehindYou", annotated)
    return cv2.waitKey(1) & 0xFF == ord("q")


def _notify_if_needed(intruder_boxes: list[np.ndarray], state: _LoopState) -> None:
    if not intruder_boxes:
        return
    now = time.monotonic()
    if now - state.last_notify_time >= state.config.cooldown:
        send_notification(len(intruder_boxes))
        state.last_notify_time = now
