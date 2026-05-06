from __future__ import annotations

import dataclasses
import logging

import cv2
import numpy as np

from behindyou.paths import FACE_DATA_FILE


@dataclasses.dataclass
class FaceCacheEntry:
    embedding: np.ndarray | None = None
    first_seen: int = 0
    last_retry: int = 0


@dataclasses.dataclass(frozen=True)
class FaceInfo:
    bbox: np.ndarray
    score: float
    embedding: np.ndarray | None = None


def _crop_upper_body(frame: np.ndarray, bbox: np.ndarray, crop_ratio: float = 0.55) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop_bottom = min(y1 + int((y2 - y1) * crop_ratio), h)
    return frame[max(0, y1) : crop_bottom, max(0, x1) : min(x2, w)]


logger = logging.getLogger(__name__)


class FaceDetector:
    def __init__(self) -> None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"无法加载 Haar Cascade: {path}")

    def has_frontal_face(
        self,
        frame: np.ndarray,
        box: np.ndarray,
        min_face_ratio: float = 0.15,
        crop_ratio: float = 0.55,
    ) -> bool:
        roi = _crop_upper_body(frame, box, crop_ratio)
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

    def _get_frontal_faces(
        self, frame: np.ndarray, person_bbox: np.ndarray, crop_ratio: float, max_yaw: float
    ) -> list | None:
        """Return detected faces filtered to frontal-only, or None on error/empty."""
        roi = _crop_upper_body(frame, person_bbox, crop_ratio)
        if roi.size == 0:
            return None
        try:
            faces = self.app.get(roi)
        except (RuntimeError, OSError, cv2.error):
            logger.warning("InsightFace detection 异常", exc_info=True)
            return None
        if not faces:
            return None
        frontal = [f for f in faces if f.pose is None or abs(f.pose[0]) <= max_yaw]
        return frontal or None

    def has_frontal_face(
        self,
        frame: np.ndarray,
        person_bbox: np.ndarray,
        crop_ratio: float = 0.55,
        min_det_score: float = 0.5,
        max_yaw: float = 45.0,
    ) -> bool:
        faces = self._get_frontal_faces(frame, person_bbox, crop_ratio, max_yaw)
        if not faces:
            return False
        return any(f.det_score >= min_det_score for f in faces)

    def check_frontal_and_get_embedding(
        self,
        frame: np.ndarray,
        person_bbox: np.ndarray,
        crop_ratio: float = 0.55,
        min_det_score: float = 0.5,
        max_yaw: float = 45.0,
    ) -> FaceInfo | None:
        frontal = self._get_frontal_faces(frame, person_bbox, crop_ratio, max_yaw)
        if not frontal:
            return None

        best = max(frontal, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        if best.det_score < min_det_score:
            return None

        crop_y1 = max(0, int(person_bbox[1]))
        crop_x1 = max(0, int(person_bbox[0]))
        face_abs = best.bbox + np.array([crop_x1, crop_y1, crop_x1, crop_y1])

        return FaceInfo(
            bbox=face_abs.astype(int),
            score=float(best.det_score),
            embedding=best.embedding,
        )

    def get_embedding(
        self,
        frame: np.ndarray,
        person_bbox: np.ndarray,
        crop_ratio: float = 0.55,
        max_yaw: float = 45.0,
    ) -> np.ndarray | None:
        frontal = self._get_frontal_faces(frame, person_bbox, crop_ratio, max_yaw)
        if not frontal:
            return None
        return frontal[0].embedding

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
            FACE_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(FACE_DATA_FILE), self.owner_embedding)
            logger.info("人脸数据已保存到 %s", FACE_DATA_FILE)

    def load_embedding(self) -> bool:
        try:
            self.owner_embedding = np.load(str(FACE_DATA_FILE))
            if self.owner_embedding.ndim != 1:
                logger.warning(
                    "人脸数据格式异常（shape=%s），将重新采集", self.owner_embedding.shape
                )
                self.owner_embedding = None
                return False
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
        cache: dict[int, FaceCacheEntry],
        frame_count: int,
        retry_interval: int,
        crop_ratio: float = 0.55,
        initial_delay: int = 3,
    ) -> np.ndarray | None:
        if tid not in cache:
            cache[tid] = FaceCacheEntry(first_seen=frame_count)
            return None

        entry = cache[tid]
        if entry.embedding is not None:
            return entry.embedding

        if frame_count - entry.first_seen <= initial_delay:
            return None

        if entry.last_retry == entry.first_seen or frame_count - entry.last_retry >= retry_interval:
            emb = self.get_embedding(frame, bbox, crop_ratio)
            entry.embedding = emb
            entry.last_retry = frame_count

        return entry.embedding
