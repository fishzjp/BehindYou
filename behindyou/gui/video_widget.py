from __future__ import annotations

import logging
import math

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPaintEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel

from behindyou.gui.styles import current_colors

logger = logging.getLogger(__name__)


class VideoDisplay(QLabel):
    """Displays BGR numpy frames as QPixmap."""

    fps_updated = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder_text = "◉  等待摄像头画面\n点击侧栏「启动」或按 F5 开始检测"
        self.setText(self._placeholder_text)
        self._rendering = False
        self._frame_data: np.ndarray | None = None
        self._drop_count = 0

        # Monitoring glow
        self._monitoring = False
        self._glow_alpha = 0.0
        self._pulse_phase = 0.0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(50)
        self._pulse_timer.timeout.connect(self._tick_pulse)

        # FPS counter
        self._frame_count = 0
        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(1000)
        self._fps_timer.timeout.connect(self._tick_fps)

    def set_monitoring(self, running: bool) -> None:
        self._monitoring = running
        if running:
            self._pulse_timer.start()
            self._fps_timer.start()
            self._frame_count = 0
        else:
            self._pulse_timer.stop()
            self._fps_timer.stop()
            self._glow_alpha = 0.0
            self.update()

    def set_placeholder_mode(self, on: bool) -> None:
        if on:
            self.setPixmap(QPixmap())
            self.setText(self._placeholder_text)

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
            self._frame_count += 1
            if self._drop_count > 0:
                logger.debug("恢复渲染前丢帧: %d", self._drop_count)
                self._drop_count = 0
        except (ValueError, cv2.error, RuntimeError):
            logger.warning("视频帧渲染异常", exc_info=True)
        finally:
            self._rendering = False

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if not self._monitoring or self._glow_alpha <= 0:
            return
        color = QColor(current_colors()["success"])
        color.setAlpha(int(self._glow_alpha * 255))
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(color, 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 7, 7)
        painter.end()

    def _tick_pulse(self) -> None:
        self._pulse_phase += 0.1
        self._glow_alpha = 0.4 + 0.6 * abs(math.sin(self._pulse_phase))
        self.update()

    def _tick_fps(self) -> None:
        self.fps_updated.emit(float(self._frame_count))
        self._frame_count = 0
