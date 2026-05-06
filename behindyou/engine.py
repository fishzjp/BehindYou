from __future__ import annotations

import dataclasses
import logging
import time
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING

import cv2
import numpy as np
import supervision as sv

from behindyou.config import Config
from behindyou.detection import detect_people, load_model
from behindyou.face import FaceDetector, FaceCacheEntry, FaceInfo, FaceRecognizer
from behindyou.notification import save_screenshot, send_notification_async
from behindyou.paths import MODEL_FILE, PROJECT_ROOT
from behindyou.tracking import (
    box_center,
    is_reasonable_shift,
    point_in_box,
    update_ema,
)

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = logging.getLogger(__name__)

_LABEL_INTRUDER = "INTRUDER"

# --- Annotators (module-level, shared) ---

_box_annotator = sv.BoxAnnotator(thickness=2)
_label_annotator = sv.LabelAnnotator(
    text_scale=1.0,
    text_thickness=2,
    color=sv.Color.BLACK,
    text_color=sv.Color.WHITE,
    text_padding=4,
)
_red_box_annotator = sv.BoxAnnotator(color=sv.Color.RED, thickness=2)
_red_label_annotator = sv.LabelAnnotator(text_scale=1.0, text_thickness=2, color=sv.Color.RED)
_face_box_annotator = sv.BoxAnnotator(
    color=sv.Color.GREEN,
    thickness=2,
    color_lookup=sv.ColorLookup.INDEX,
)
_face_label_annotator = sv.LabelAnnotator(
    text_scale=1.0,
    text_thickness=2,
    color=sv.Color.GREEN,
    text_color=sv.Color.BLACK,
    text_padding=4,
    color_lookup=sv.ColorLookup.INDEX,
)


# --- Data classes ---


@dataclasses.dataclass(frozen=True)
class FrameResult:
    annotated_frame: np.ndarray
    detections: sv.Detections
    intruder_boxes: list[np.ndarray]
    should_notify: bool
    screenshot_path: str | None
    frame_number: int


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
    face_cache: dict[int, FaceCacheEntry] = dataclasses.field(default_factory=dict)

    def adopt_as_self(self, tid: int, xyxy: np.ndarray) -> None:
        old_self_id = self.self_id
        self.self_id = tid
        self.ema_box = xyxy.copy()
        self.ema_skip_count = 0
        self.track_persistence.pop(old_self_id, None)
        self.face_cache.pop(old_self_id, None)
        self.track_persistence.pop(tid, None)
        self.face_cache.pop(tid, None)

    def recompute_min_box_area(self, frame_h: int, frame_w: int) -> None:
        self.min_box_area = frame_h * frame_w * self.config.min_area


# --- Pure processing functions ---


def _scalar_iou(a: np.ndarray, b: np.ndarray) -> float:
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return min(1.0, inter / (max(0.0, union) + 1e-6))


def process_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    state: _LoopState,
) -> tuple[list[np.ndarray], list[FaceInfo]]:
    config = state.config
    intruder_boxes: list[np.ndarray] = []
    face_boxes: list[FaceInfo] = []
    current_tracks: set[int] = set()

    tracker_ids = detections.tracker_id
    if tracker_ids is None:
        return intruder_boxes, face_boxes

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
                frame, xyxy_f, tid, state.face_cache, state.frame_count,
                config.face_retry_interval, config.face_crop_ratio,
            )
            if emb is not None and state.face_recognizer.is_owner(emb, config.face_match_threshold):
                is_self = True
                if tid != state.self_id:
                    state.adopt_as_self(tid, xyxy_f)
                    logger.info("人脸匹配成功：主人重入，新 ID=%d", tid)

        if is_self:
            _update_self_track(state, tid, xyxy_f)
            continue

        state.track_persistence[tid] = state.track_persistence.get(tid, 0) + 1
        if state.track_persistence[tid] < config.persistence:
            continue

        if _scalar_iou(xyxy_f, state.ema_box) > config.self_iou_threshold:
            state.adopt_as_self(tid, xyxy_f)
            continue

        if _classify_stranger(frame, xyxy_f, tid, state, face_boxes):
            intruder_boxes.append(xyxy_f.astype(int))

    for tid in list(state.track_persistence):
        if tid not in current_tracks:
            del state.track_persistence[tid]
            state.face_cache.pop(tid, None)

    return intruder_boxes, face_boxes


def _update_self_track(state: _LoopState, tid: int, xyxy: np.ndarray) -> None:
    config = state.config
    if is_reasonable_shift(state.ema_box, xyxy, config.ema_max_shift):
        state.ema_box = update_ema(state.ema_box, xyxy, config.ema_alpha)
        state.ema_skip_count = 0
        if tid != state.self_id:
            state.self_id = tid
    else:
        state.ema_skip_count += 1
        if state.ema_skip_count >= config.ema_max_skips:
            state.adopt_as_self(tid, xyxy)


def _classify_stranger(
    frame: np.ndarray,
    xyxy: np.ndarray,
    tid: int,
    state: _LoopState,
    face_boxes: list[FaceInfo],
) -> bool:
    config = state.config
    if config.no_face_check:
        return True

    recognizer = state.face_recognizer
    if recognizer is not None and recognizer.has_owner:
        entry = state.face_cache.get(tid)
        if entry is not None and entry.embedding is not None:
            return True

    if recognizer is not None:
        face_info = recognizer.check_frontal_and_get_embedding(
            frame, xyxy, config.face_crop_ratio, config.face_det_score,
        )
        if face_info is None:
            return False
        face_boxes.append(face_info)
        if face_info.embedding is not None and recognizer.has_owner:
            entry = state.face_cache.get(tid)
            if entry is not None:
                entry.embedding = face_info.embedding
                entry.last_retry = state.frame_count
            else:
                state.face_cache[tid] = FaceCacheEntry(
                    embedding=face_info.embedding,
                    first_seen=state.frame_count,
                    last_retry=state.frame_count,
                )
        return True
    elif state.face_detector is not None:
        return state.face_detector.has_frontal_face(
            frame, xyxy, config.face_min_size, config.face_crop_ratio,
        )
    return False


def annotate_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    intruder_boxes: list[np.ndarray],
    face_info_list: list[FaceInfo] | None = None,
) -> np.ndarray:
    annotated = _box_annotator.annotate(scene=frame, detections=detections)

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
            scene=annotated,
            detections=intruder_det,
            labels=[_LABEL_INTRUDER] * len(intruder_boxes),
        )

    if face_info_list:
        fb_xyxy = np.array([fi.bbox for fi in face_info_list], dtype=np.float32)
        face_det = sv.Detections(xyxy=fb_xyxy)
        annotated = _face_box_annotator.annotate(scene=annotated, detections=face_det)
        face_labels = [f"face {fi.score:.2f}" for fi in face_info_list]
        annotated = _face_label_annotator.annotate(
            scene=annotated, detections=face_det, labels=face_labels
        )

    return annotated


# --- DetectionEngine ---


class DetectionEngine:
    """Stateful detection engine. Call step() once per frame."""

    def __init__(self, config: Config) -> None:
        self._config = config

        if MODEL_FILE.exists():
            model_path = str(MODEL_FILE)
        else:
            dev_path = PROJECT_ROOT / "yolo26n.pt"
            if dev_path.exists():
                model_path = str(dev_path)
            else:
                raise FileNotFoundError(
                    f"模型文件不存在：{MODEL_FILE}\n"
                    "请将 yolo26n.pt 放到 ~/.behindyou/ 目录下\n"
                    "下载地址: https://github.com/ultralytics/assets/releases"
                )
        self._model: YOLO = load_model(model_path)

        self._face_detector: FaceDetector | None = None
        if not config.no_face_check:
            try:
                self._face_detector = FaceDetector()
            except (RuntimeError, OSError, cv2.error) as e:
                logger.warning("人脸检测器初始化失败（%s），将禁用人脸过滤", e)
                self._config = dataclasses.replace(self._config, no_face_check=True)

        self._face_recognizer: FaceRecognizer | None = None
        try:
            self._face_recognizer = FaceRecognizer()
            logger.info("人脸识别模型已加载")
            if not config.recalibrate:
                self._face_recognizer.load_embedding()
        except (RuntimeError, OSError, ImportError) as e:
            logger.warning("人脸识别初始化失败（%s），主人重入识别将不可用", e)

        self._state: _LoopState | None = None
        self._cap: cv2.VideoCapture | None = None

    @property
    def face_recognizer(self) -> FaceRecognizer | None:
        return self._face_recognizer

    @property
    def has_saved_embedding(self) -> bool:
        return (
            self._face_recognizer is not None and self._face_recognizer.owner_embedding is not None
        )

    def open_camera(self, camera_index: int = 0) -> bool:
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            logger.error("无法打开摄像头 (index=%d)", camera_index)
            self._cap = None
            return False
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def calibrate(
        self,
        quick: bool = False,
        progress_cb: Callable[[int, int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[int | None, np.ndarray | None]:
        if self._cap is None:
            return None, None

        config = self._config
        face_recognizer = self._face_recognizer

        if quick:
            msg = "快速校准：检测画面中的人物..."
        else:
            msg = "校准中：请确保只有你一个人在画面中，3 秒后开始采样..."
        logger.info(msg)
        if progress_cb:
            progress_cb(0, 0, msg)
        if not quick:
            for _ in range(30):
                if cancel_check and cancel_check():
                    return None, None
                time.sleep(0.1)

        sample_frames = 10 if quick else 30
        track_ids: list[int] = []
        boxes: list[np.ndarray] = []
        embeddings_by_id: dict[int, list[np.ndarray]] = {}
        frame: np.ndarray | None = None
        unique_ids: set[int] = set()

        for frame_idx in range(sample_frames):
            if cancel_check and cancel_check():
                return None, None
            ret, frame = self._cap.read()
            if not ret:
                continue
            detections = detect_people(self._model, frame, config.confidence)
            frame_area = frame.shape[0] * frame.shape[1]
            frame_got_face = False
            tracker_ids = detections.tracker_id
            if tracker_ids is None:
                continue
            for i in range(len(detections.xyxy)):
                tid = int(tracker_ids[i])
                xyxy = detections.xyxy[i]
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                if area < frame_area * config.min_area:
                    continue
                track_ids.append(tid)
                boxes.append(xyxy)
                unique_ids.add(tid)
                if not quick and face_recognizer is not None:
                    if tid not in embeddings_by_id:
                        embeddings_by_id[tid] = []
                    emb = face_recognizer.get_embedding(frame, xyxy)
                    if emb is not None:
                        embeddings_by_id[tid].append(emb)
                        frame_got_face = True

            if progress_cb:
                status = "正在采集人脸..." if frame_got_face else "请面向摄像头"
                progress_cb(frame_idx + 1, sample_frames, status)

        if not track_ids or frame is None:
            logger.warning("校准失败：未检测到人物，请确保你在画面中后重试")
            return None, None

        if len(unique_ids) > 1:
            logger.warning("检测到 %d 个不同的人，校准结果可能不准确", len(unique_ids))

        most_common = Counter(track_ids).most_common(2)
        self_id = most_common[0][0]
        if len(most_common) > 1:
            ratio = most_common[0][1] / len(track_ids)
            if ratio < 0.6:
                logger.warning("主人 ID 置信度偏低 (%.0f%%)，校准可能不准确", ratio * 100)
        self_boxes = [b for b, t in zip(boxes, track_ids) if t == self_id]
        avg_box = np.mean(self_boxes, axis=0)

        if face_recognizer is not None:
            owner_embs = embeddings_by_id.get(self_id, [])
            if owner_embs:
                face_recognizer.set_owner_embedding(np.mean(owner_embs, axis=0))
                face_recognizer.save_embedding()
                logger.info("人脸特征已采集（%d/%d 帧成功）", len(owner_embs), sample_frames)
            else:
                logger.warning("校准期间未能采集到人脸特征，人脸识别回退将不可用")

        logger.info(
            "校准完成：自身 ID=%d，位置 [%.0f,%.0f,%.0f,%.0f]",
            self_id,
            avg_box[0],
            avg_box[1],
            avg_box[2],
            avg_box[3],
        )
        return self_id, avg_box

    def start(self, self_id: int, ema_box: np.ndarray) -> None:
        cap = self._cap
        if cap is None:
            raise RuntimeError("摄像头未打开，请先调用 open_camera()")

        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        min_box_area = frame_h * frame_w * self._config.min_area

        self._state = _LoopState(
            config=self._config,
            self_id=self_id,
            ema_box=ema_box,
            face_detector=self._face_detector,
            face_recognizer=self._face_recognizer,
            min_box_area=min_box_area,
        )

    def read_frame(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def step(self, frame: np.ndarray) -> FrameResult:
        if self._state is None:
            raise RuntimeError("引擎未启动，请先调用 start()")

        state = self._state
        config = state.config

        results = detect_people(self._model, frame, config.confidence)
        state.frame_count += 1

        intruder_boxes, face_info_list = process_frame(frame, results, state)

        annotated = annotate_frame(frame, results, intruder_boxes, face_info_list)

        should_notify = False
        screenshot_path: str | None = None
        if intruder_boxes:
            now = time.monotonic()
            if now - state.last_notify_time >= config.cooldown:
                screenshot_path = save_screenshot(annotated)
                send_notification_async(len(intruder_boxes), screenshot_path)
                state.last_notify_time = now
                should_notify = True

        return FrameResult(
            annotated_frame=annotated,
            detections=results,
            intruder_boxes=intruder_boxes,
            should_notify=should_notify,
            screenshot_path=screenshot_path,
            frame_number=state.frame_count,
        )

    def update_config(self, new_config: Config) -> None:
        self._config = new_config
        if self._state is not None:
            self._state.config = new_config
            if self._cap is not None:
                self._state.recompute_min_box_area(
                    int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                )

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._state = None
        logger.info("检测引擎已停止")
