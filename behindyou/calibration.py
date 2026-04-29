from __future__ import annotations

import logging
import time
from collections import Counter
from typing import TYPE_CHECKING

import supervision as sv

import cv2
import numpy as np

from behindyou.detection import detect_people

if TYPE_CHECKING:
    from ultralytics import YOLO

    from behindyou.config import Config
    from behindyou.face import FaceRecognizer

logger = logging.getLogger(__name__)


def calibrate(
    model: YOLO,
    cap: cv2.VideoCapture,
    config: Config,
    face_recognizer: FaceRecognizer | None = None,
    quick: bool = False,
) -> tuple[int | None, np.ndarray | None]:
    if quick:
        logger.info("快速校准：检测画面中的人物...")
    else:
        logger.info("校准中：请确保只有你一个人在画面中，3 秒后开始采样...")
        time.sleep(3)

    sample_frames = 10 if quick else 30
    track_ids: list[int] = []
    boxes: list[np.ndarray] = []
    embeddings_by_id: dict[int, list[np.ndarray]] = {}
    frame: np.ndarray | None = None
    unique_ids: set[int] = set()

    for _ in range(sample_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        detections = detect_people(model, frame, config.confidence)
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

        if not quick and face_recognizer is not None and not config.no_preview:
            if not frame_got_face:
                display = frame.copy()
                h, w = display.shape[:2]
                text = "请面向摄像头"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1.2
                thickness = 3
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                cv2.putText(display, text, ((w - tw) // 2, h // 2),
                            font, scale, (0, 0, 255), thickness, cv2.LINE_AA)
                cv2.imshow("BehindYou", display)
            else:
                cv2.imshow("BehindYou", frame)
            cv2.waitKey(1)

    if not track_ids or frame is None:
        logger.warning("校准失败：未检测到人物，请确保你在画面中后重试")
        return None, None

    if len(unique_ids) > 1:
        logger.warning("检测到 %d 个不同的人，校准结果可能不准确", len(unique_ids))

    self_id = Counter(track_ids).most_common(1)[0][0]
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

    h, w = frame.shape[:2]
    if not config.no_preview:
        cv2.destroyAllWindows()

    logger.info(
        "校准完成：自身 ID=%d，位置 [%.0f,%.0f,%.0f,%.0f] (画面 %dx%d)",
        self_id, avg_box[0], avg_box[1], avg_box[2], avg_box[3], w, h,
    )
    return self_id, avg_box
