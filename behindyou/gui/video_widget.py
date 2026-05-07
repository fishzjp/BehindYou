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
        self.setMinimumSize(480, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("等待摄像头画面\n点击侧栏「启动」按钮开始检测")
        self._rendering = False
        self._frame_data: np.ndarray | None = None
        self._drop_count = 0

    def update_frame(self, bgr_frame: np.ndarray) -> None:
        if self._rendering:
            self._drop_count += 1
            if self._drop_count % 100 == 0:
                logger.warning("已丢弃 %d 帧", self._drop_count)
            return
        self._rendering = True
        try:
            if bgr_frame is None or bgr_frame.size == 0:
                return
            self._frame_data = np.ascontiguousarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
            h, w, ch = self._frame_data.shape
            qimg = QImage(self._frame_data.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            scaled = pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
            if self._drop_count > 0:
                logger.debug("恢复渲染前丢帧: %d", self._drop_count)
                self._drop_count = 0
        except (ValueError, cv2.error, RuntimeError):
            logger.warning("视频帧渲染异常", exc_info=True)
        finally:
            self._rendering = False
