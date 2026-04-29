from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from behindyou.paths import PROJECT_ROOT
from behindyou.tracking import crop_upper_body

logger = logging.getLogger(__name__)

FACE_DATA_FILE = os.path.join(PROJECT_ROOT, "owner_face.npy")


class FaceDetector:
    def __init__(self) -> None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"无法加载 Haar Cascade: {path}")

    def has_frontal_face(self, frame: np.ndarray, box: np.ndarray, min_face_ratio: float = 0.15) -> bool:
        roi = crop_upper_body(frame, box)
        if roi.size == 0:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        min_size = (int(roi.shape[1] * min_face_ratio), int(roi.shape[0] * min_face_ratio))
        faces = self.cascade.detectMultiScale(gray, 1.1, 3, minSize=min_size)
        return len(faces) > 0


class FaceRecognizer:
    def __init__(self) -> None:
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(320, 320))
        self.owner_embedding: np.ndarray | None = None
        self._owner_norm: float = 0.0

    def get_embedding(self, frame: np.ndarray, person_bbox: np.ndarray) -> np.ndarray | None:
        roi = crop_upper_body(frame, person_bbox)
        if roi.size == 0:
            return None
        faces = self.app.get(roi)
        if not faces:
            return None
        return faces[0].embedding

    def set_owner_embedding(self, embedding: np.ndarray) -> None:
        self.owner_embedding = embedding.copy()
        self._owner_norm = float(np.linalg.norm(self.owner_embedding))

    @property
    def has_owner(self) -> bool:
        return self.owner_embedding is not None

    def is_owner(self, embedding: np.ndarray, threshold: float = 0.5) -> bool:
        if self.owner_embedding is None or embedding is None:
            return False
        emb_norm = float(np.linalg.norm(embedding))
        denom = self._owner_norm * emb_norm
        if denom < 1e-8:
            return False
        similarity = np.dot(self.owner_embedding, embedding) / denom
        return similarity >= threshold

    def save_embedding(self) -> None:
        if self.owner_embedding is not None:
            np.save(FACE_DATA_FILE, self.owner_embedding)
            logger.info("人脸数据已保存到 %s", FACE_DATA_FILE)

    def load_embedding(self) -> bool:
        try:
            self.owner_embedding = np.load(FACE_DATA_FILE)
            self._owner_norm = float(np.linalg.norm(self.owner_embedding))
            logger.info("已加载保存的人脸数据: %s", FACE_DATA_FILE)
            return True
        except (FileNotFoundError, ValueError, EOFError, OSError) as e:
            logger.warning("加载人脸数据失败（%s），将重新采集", e)
            return False

    def get_cached_embedding(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        tid: int,
        cache: dict[int, np.ndarray | None],
        retry_map: dict[int, int],
        frame_count: int,
        retry_interval: int,
    ) -> np.ndarray | None:
        if tid not in cache:
            emb = self.get_embedding(frame, bbox)
            cache[tid] = emb
            if emb is None:
                retry_map[tid] = frame_count
        elif cache[tid] is None and frame_count - retry_map.get(tid, 0) >= retry_interval:
            emb = self.get_embedding(frame, bbox)
            cache[tid] = emb
            if emb is None:
                retry_map[tid] = frame_count
        return cache.get(tid)
