from __future__ import annotations

import dataclasses
import logging
import os
import time

import supervision as sv

import cv2
import numpy as np

from behindyou.calibration import calibrate
from behindyou.config import Config
from behindyou.detection import detect_people, load_model
from behindyou.face import FaceDetector, FaceRecognizer
from behindyou.notification import save_screenshot, send_notification
from behindyou.paths import PROJECT_ROOT
from behindyou.tracking import (
    box_center,
    is_reasonable_shift,
    point_in_box,
    update_ema,
)

logger = logging.getLogger(__name__)


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

    target_interval = 1.0 / 30.0

    try:
        while True:
            frame_start = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                logger.error("读取摄像头画面失败")
                break

            results = detect_people(model, frame, config.confidence)
            state.frame_count += 1

            intruder_boxes = _process_frame(frame, results, state)

            annotated: np.ndarray | None = None
            if not config.no_preview or intruder_boxes:
                annotated = _annotate_frame(frame, results, intruder_boxes)

            if not config.no_preview:
                if _render_preview(annotated):
                    break

            if annotated is not None:
                _notify_if_needed(intruder_boxes, state, annotated)

            elapsed = time.monotonic() - frame_start
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

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

    def adopt_as_self(self, tid: int, xyxy: np.ndarray) -> None:
        old_self_id = self.self_id
        self.self_id = tid
        self.ema_box = xyxy.copy()
        self.ema_skip_count = 0
        for d in (self.track_persistence, self.face_cache, self.face_retry_frame):
            d.pop(old_self_id, None)
        self.track_persistence.pop(tid, None)
        self.face_cache.pop(tid, None)
        self.face_retry_frame.pop(tid, None)


def _process_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    state: _LoopState,
) -> list[np.ndarray]:
    config = state.config
    intruder_boxes: list[np.ndarray] = []
    current_tracks: set[int] = set()

    tracker_ids = detections.tracker_id
    if tracker_ids is None:
        return intruder_boxes

    for i in range(len(detections.xyxy)):
        xyxy_f = detections.xyxy[i]
        box_area = (xyxy_f[2] - xyxy_f[0]) * (xyxy_f[3] - xyxy_f[1])
        if box_area < state.min_box_area:
            continue

        tid = int(tracker_ids[i])
        current_tracks.add(tid)

        is_self = tid == state.self_id or point_in_box(box_center(xyxy_f), state.ema_box)
        if not is_self and state.face_recognizer is not None and state.face_recognizer.has_owner:
            emb = state.face_recognizer.get_cached_embedding(
                frame, xyxy_f, tid, state.face_cache, state.face_retry_frame,
                state.frame_count, config.face_retry_interval,
            )
            if emb is not None and state.face_recognizer.is_owner(emb, config.face_match_threshold):
                is_self = True
                state.adopt_as_self(tid, xyxy_f)
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
                    state.adopt_as_self(tid, xyxy_f)
            continue

        state.track_persistence[tid] = state.track_persistence.get(tid, 0) + 1
        if state.track_persistence[tid] >= config.persistence:
            if _scalar_iou(xyxy_f, state.ema_box) > config.self_iou_threshold:
                state.adopt_as_self(tid, xyxy_f)
                continue
            if state.face_recognizer is not None and state.face_recognizer.has_owner:
                cached_emb = state.face_cache.get(tid)
                if cached_emb is None:
                    if state.face_detector is not None:
                        if not state.face_detector.has_frontal_face(
                            frame, xyxy_f, config.face_min_size
                        ):
                            continue
                    else:
                        continue
            elif state.face_detector is not None:
                if not state.face_detector.has_frontal_face(frame, xyxy_f, config.face_min_size):
                    continue
            intruder_boxes.append(xyxy_f.astype(int))

    for tid in list(state.track_persistence):
        if tid not in current_tracks:
            del state.track_persistence[tid]
            state.face_cache.pop(tid, None)
            state.face_retry_frame.pop(tid, None)

    return intruder_boxes


_box_annotator = sv.BoxAnnotator()
_label_annotator = sv.LabelAnnotator(
    text_scale=0.5, text_thickness=1, color=sv.Color.BLACK, text_color=sv.Color.WHITE, text_padding=2,
)
_red_box_annotator = sv.BoxAnnotator(color=sv.Color.RED)
_red_label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, color=sv.Color.RED)


def _scalar_iou(a: np.ndarray, b: np.ndarray) -> float:
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (union + 1e-6)


def _annotate_frame(frame: np.ndarray, detections: sv.Detections, intruder_boxes: list[np.ndarray]) -> np.ndarray:
    annotated = frame.copy()

    annotated = _box_annotator.annotate(scene=annotated, detections=detections)

    tracker_ids = detections.tracker_id
    confidences = detections.confidence
    if tracker_ids is not None and confidences is not None:
        labels = [f"person #{int(tid)} {conf:.2f}" for tid, conf in zip(tracker_ids, confidences)]
        annotated = _label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    if intruder_boxes:
        intruder_det = sv.Detections(
            xyxy=np.array(intruder_boxes, dtype=np.float32),
            class_id=np.zeros(len(intruder_boxes), dtype=int),
        )
        annotated = _red_box_annotator.annotate(scene=annotated, detections=intruder_det)
        annotated = _red_label_annotator.annotate(
            scene=annotated, detections=intruder_det,
            labels=["INTRUDER"] * len(intruder_boxes),
        )

    return annotated


def _render_preview(annotated: np.ndarray) -> bool:
    cv2.imshow("BehindYou", annotated)
    return cv2.waitKey(1) & 0xFF == ord("q")


def _notify_if_needed(
    intruder_boxes: list[np.ndarray],
    state: _LoopState,
    annotated: np.ndarray,
) -> None:
    if not intruder_boxes:
        return
    now = time.monotonic()
    if now - state.last_notify_time < state.config.cooldown:
        return
    path = save_screenshot(annotated)
    send_notification(len(intruder_boxes), path)
    state.last_notify_time = now
