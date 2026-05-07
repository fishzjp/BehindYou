from __future__ import annotations

import dataclasses
import logging

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

logger = logging.getLogger(__name__)


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
        self._label.setWordWrap(True)
        layout.addWidget(self._label)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(int(min_val * self._scale))
        self._slider.setMaximum(int(max_val * self._scale))
        self._slider.setValue(int(default * self._scale))
        self._slider.valueChanged.connect(self._on_changed)
        self._slider.setAccessibleName(label)
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
        self.setMinimumWidth(200)
        self.setMaximumWidth(300)
        self.setAccessibleName("设置面板")

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
        self._btn_start.setAccessibleName("启动检测")
        self._btn_start.setToolTip("开始检测身后人员 (F5)")
        self._btn_stop = QPushButton("停止")
        self._btn_stop.setObjectName("btn_stop")
        self._btn_stop.setAccessibleName("停止检测")
        self._btn_stop.setToolTip("停止检测 (F6)")
        self._btn_calibrate = QPushButton("校准")
        self._btn_calibrate.setAccessibleName("人脸校准")
        self._btn_calibrate.setToolTip("重新采集你的人脸数据用于主人识别")
        self._btn_stop.setEnabled(False)

        btn_layout.addWidget(self._btn_start)
        btn_layout.addWidget(self._btn_stop)
        btn_layout.addWidget(self._btn_calibrate)

        self._btn_start.clicked.connect(self.start_requested)
        self._btn_stop.clicked.connect(self.stop_requested)
        self._btn_calibrate.clicked.connect(self.calibrate_requested)

        self._layout.addWidget(btn_row)

        # --- Detection settings ---
        detect_group = QGroupBox("检测参数")
        detect_layout = QVBoxLayout(detect_group)

        self._camera = _SliderRow("摄像头编号", 0, 10, 0, step=1.0)
        self._camera.setToolTip("摄像头设备编号，0 为默认摄像头")
        self._confidence = _SliderRow("置信度阈值", 0.1, 1.0, 0.6)
        self._confidence.setToolTip(
            "识别置信度阈值 (0.1-1.0)\n值越低越灵敏但误报越多，值越高越严格但可能漏检"
        )

        detect_layout.addWidget(self._camera)
        detect_layout.addWidget(self._confidence)
        self._layout.addWidget(detect_group)

        # --- Alert & face settings ---
        alert_group = QGroupBox("报警与人脸")
        alert_layout = QVBoxLayout(alert_group)

        self._cooldown = _SliderRow("报警间隔(秒)", 1, 60, 10, step=1.0)
        self._cooldown.setToolTip("两次报警之间的最短间隔（秒），避免频繁通知")
        self._no_face_check = QCheckBox("关闭人脸验证（任何人靠近都报警）")
        self._no_face_check.setAccessibleName("关闭人脸验证")
        self._face_det_score = _SliderRow("人脸检测阈值", 0.1, 0.9, 0.8)
        self._face_det_score.setToolTip(
            "人脸检测的置信度阈值 (0.1-0.9)\n用于判断检测到的人是否有可见人脸"
        )

        alert_layout.addWidget(self._cooldown)
        alert_layout.addWidget(self._no_face_check)
        alert_layout.addWidget(self._face_det_score)
        self._layout.addWidget(alert_group)

        self.setWidget(container)

        cfg, err = load_config()
        if err:
            logger.warning(f"加载配置失败: {err}")
        self._current_config = cfg or Config()

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
        overrides: dict = dict(
            camera=self._camera.value,
            confidence=self._confidence.value,
            cooldown=self._cooldown.value,
            no_face_check=self._no_face_check.isChecked(),
        )
        if not self._no_face_check.isChecked():
            overrides["face_det_score"] = self._face_det_score.value
        return dataclasses.replace(self._current_config, **overrides)

    def set_running(self, running: bool) -> None:
        self._btn_start.setEnabled(not running)
        self._btn_stop.setEnabled(running)
        self._btn_calibrate.setEnabled(not running)
        self._camera.setEnabled(not running)

    def emit_config(self) -> None:
        self._current_config = self.get_config()
        save_config(self._current_config)
        self.config_changed.emit(self._current_config)
