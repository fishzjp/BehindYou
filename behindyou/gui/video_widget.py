from __future__ import annotations

import logging

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel

logger = logging.getLogger(__name__)


class VideoDisplay(QLabel):
    """Displays BGR numpy frames as QPixmap."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #1e1e1e;")
        self.setText("等待摄像头画面...")

    def update_frame(self, bgr_frame: np.ndarray) -> None:
        try:
            if bgr_frame is None or bgr_frame.size == 0:
                return
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qimg)
            scaled = pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
        except (ValueError, cv2.error, RuntimeError):
            logger.warning("视频帧渲染异常", exc_info=True)
