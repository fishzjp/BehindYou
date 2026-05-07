from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal, Slot

from behindyou.config import Config
from behindyou.engine import DetectionEngine

logger = logging.getLogger(__name__)


class DetectionWorker(QObject):
    """Runs DetectionEngine in a background QThread."""

    frame_ready = Signal(np.ndarray)
    calibration_progress = Signal(int, int, str)
    calibration_done = Signal(bool, str)
    intrusion_detected = Signal(int, object)  # (count, screenshot_path)
    engine_error = Signal(str)
    status_changed = Signal(str)
    finished = Signal()
    config_requested = Signal(object)

    def __init__(self, config: Config, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._thread: QThread | None = None
        self._engine: DetectionEngine | None = None
        self._stop_event = threading.Event()
        self._config_dirty = threading.Event()
        self.config_requested.connect(self._apply_config)

    def start(self) -> None:
        if self._thread is not None:
            if self._thread.isRunning():
                return
            # Thread finished but cleanup hasn't run yet
            self._thread.wait()
            self._thread = None
        self._stop_event.clear()
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def _is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def _run(self) -> None:
        try:
            self._engine = DetectionEngine(self._config)

            if not self._engine.open_camera(self._config.camera):
                self.engine_error.emit("无法打开摄像头")
                return

            has_saved = self._engine.has_saved_embedding
            self.status_changed.emit("校准中...")

            self_id, ema_box = self._engine.calibrate(
                quick=has_saved,
                progress_cb=lambda cur, tot, msg: self.calibration_progress.emit(cur, tot, msg),
                cancel_check=self._is_stopped,
            )

            if self._stop_event.is_set():
                self.calibration_done.emit(False, "校准已取消")
                return

            if self_id is None or ema_box is None:
                self.calibration_done.emit(False, "校准失败：未检测到人物")
                return

            self.calibration_done.emit(True, "校准完成")
            self._engine.start(self_id, ema_box)
            self.status_changed.emit("运行中")

            target_interval = 1.0 / 30.0

            while not self._stop_event.is_set():
                frame_start = time.monotonic()

                frame = self._engine.read_frame()
                if frame is None:
                    self.engine_error.emit("读取摄像头画面失败")
                    break

                result = self._engine.step(frame)

                self.frame_ready.emit(result.annotated_frame.copy())

                if result.should_notify:
                    self.intrusion_detected.emit(len(result.intruder_boxes), result.screenshot_path)

                elapsed = time.monotonic() - frame_start
                if elapsed < target_interval:
                    remaining = target_interval - elapsed
                    # Only busy-loop with processEvents if there's a pending config update
                    if self._config_dirty.is_set():
                        self._config_dirty.clear()
                        deadline = time.monotonic() + remaining
                        while not self._stop_event.is_set():
                            QCoreApplication.processEvents()
                            time.sleep(min(0.01, max(0, deadline - time.monotonic())))
                            if time.monotonic() >= deadline:
                                break
                    else:
                        self._stop_event.wait(remaining)

        except (RuntimeError, OSError, cv2.error) as e:
            logger.exception("检测工作线程异常")
            self.engine_error.emit(str(e))
        except Exception as e:
            logger.exception("检测工作线程遇到未预期异常")
            self.engine_error.emit(f"内部错误: {e}")
        finally:
            if self._engine is not None:
                self._engine.stop()
            self._engine = None
            self.status_changed.emit("已停止")
            self.finished.emit()
            if self._thread is not None:
                self._thread.quit()

    @Slot(object)
    def _apply_config(self, new_config: Config) -> None:
        self._config = new_config
        if self._engine is not None:
            self._engine.update_config(new_config)
        self._config_dirty.set()

    @Slot()
    def stop(self) -> None:
        self._stop_event.set()

    @Slot()
    def _on_thread_finished(self) -> None:
        if self._thread is not None:
            self._thread.wait()
            self._thread = None

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set() and self._engine is not None
