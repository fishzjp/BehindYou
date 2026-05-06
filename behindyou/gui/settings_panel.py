from __future__ import annotations

import dataclasses

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from behindyou.config import Config, load_config, save_config


class _SliderRow(QWidget):
    """A labeled slider with a value display for float/int parameters."""

    value_changed = Signal()

    def __init__(
        self,
        label: str,
        min_val: float,
        max_val: float,
        default: float,
        step: float = 0.01,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._name = label
        self._step = step
        self._scale = round(1 / step) if step < 1 else 1
        self._is_int = step >= 1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        self._label = QLabel(f"{label}: {int(default) if self._is_int else f'{default:.2f}'}")
        layout.addWidget(self._label)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(int(min_val * self._scale))
        self._slider.setMaximum(int(max_val * self._scale))
        self._slider.setValue(int(default * self._scale))
        self._slider.valueChanged.connect(self._on_changed)
        layout.addWidget(self._slider)

    def _on_changed(self, raw: int) -> None:
        val = raw / self._scale
        if self._is_int:
            self._label.setText(f"{self._name}: {int(val)}")
        else:
            self._label.setText(f"{self._name}: {val:.2f}")
        self.value_changed.emit()

    @property
    def value(self) -> float:
        raw = self._slider.value() / self._scale
        return int(raw) if self._is_int else raw

    def set_value(self, val: float) -> None:
        self._slider.setValue(int(val * self._scale))


class SettingsPanel(QScrollArea):
    """Sidebar with essential config parameters."""

    config_changed = Signal(object)
    start_requested = Signal()
    stop_requested = Signal()
    calibrate_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(240)
        self.setMaximumWidth(300)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(8)

        # --- Buttons ---
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 8)
        btn_layout.setSpacing(6)

        self._btn_start = QPushButton("启动")
        self._btn_start.setObjectName("btn_start")
        self._btn_stop = QPushButton("停止")
        self._btn_stop.setObjectName("btn_stop")
        self._btn_calibrate = QPushButton("校准")
        self._btn_stop.setEnabled(False)

        btn_layout.addWidget(self._btn_start)
        btn_layout.addWidget(self._btn_stop)
        btn_layout.addWidget(self._btn_calibrate)

        self._btn_start.clicked.connect(self.start_requested)
        self._btn_stop.clicked.connect(self.stop_requested)
        self._btn_calibrate.clicked.connect(self.calibrate_requested)

        self._layout.addWidget(btn_row)

        # --- Basic settings ---
        basic_group = QGroupBox("设置")
        basic_layout = QVBoxLayout(basic_group)

        self._camera = _SliderRow("摄像头", 0, 10, 0, step=1.0)
        self._confidence = _SliderRow("灵敏度", 0.1, 1.0, 0.6)
        self._cooldown = _SliderRow("报警间隔(秒)", 0, 60, 10, step=1.0)
        self._no_face_check = QCheckBox("关闭人脸验证（任何人靠近都报警）")
        self._face_det_score = _SliderRow("人脸检测阈值", 0.1, 0.9, 0.5)

        basic_layout.addWidget(self._camera)
        basic_layout.addWidget(self._confidence)
        basic_layout.addWidget(self._cooldown)
        basic_layout.addWidget(self._no_face_check)
        basic_layout.addWidget(self._face_det_score)
        self._layout.addWidget(basic_group)

        self.setWidget(container)

        self._current_config = load_config() or Config()

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(200)
        self._debounce_timer.timeout.connect(self.emit_config)

        for w in (self._camera, self._confidence, self._cooldown, self._face_det_score):
            w.value_changed.connect(self._debounce_timer.start)
        self._no_face_check.stateChanged.connect(self._on_face_check_changed)

    def _on_face_check_changed(self) -> None:
        checked = self._no_face_check.isChecked()
        self._face_det_score.setVisible(not checked)
        self._debounce_timer.start()

    def get_config(self) -> Config:
        return dataclasses.replace(
            self._current_config,
            camera=self._camera.value,
            confidence=self._confidence.value,
            cooldown=self._cooldown.value,
            no_face_check=self._no_face_check.isChecked(),
            face_det_score=self._face_det_score.value,
        )

    def set_running(self, running: bool) -> None:
        self._btn_start.setEnabled(not running)
        self._btn_stop.setEnabled(running)
        self._btn_calibrate.setEnabled(not running)

    def emit_config(self) -> None:
        self._current_config = self.get_config()
        save_config(self._current_config)
        self.config_changed.emit(self._current_config)
