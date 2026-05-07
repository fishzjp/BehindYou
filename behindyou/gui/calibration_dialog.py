from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from behindyou.gui.styles import repolish


class CalibrationDialog(QDialog):
    """Guided calibration wizard with visual progress."""

    cancel_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("校准 - BehindYou")
        self.setMinimumSize(420, 220)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        self._instruction = QLabel("正在准备校准...")
        self._instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._instruction.setObjectName("calibration_instruction")
        self._instruction.setAccessibleName("校准状态")
        self._instruction.setWordWrap(True)
        layout.addWidget(self._instruction)

        self._guidance = QLabel("请面向摄像头，保持自然坐姿\n摄像头画面显示在主窗口中")
        self._guidance.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._guidance.setObjectName("calibration_guidance")
        self._guidance.setWordWrap(True)
        layout.addWidget(self._guidance)

        self._progress = QProgressBar()
        self._progress.setMinimum(0)
        self._progress.setMaximum(30)
        self._progress.setValue(0)
        self._progress.setFixedHeight(8)
        self._progress.setAccessibleName("校准进度")
        layout.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setObjectName("calibration_status")
        self._status.setAccessibleName("校准详情")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        layout.addStretch()

        self._btn_action = QPushButton("取消")
        self._btn_action.setFixedHeight(32)
        self._btn_action.setAccessibleName("取消校准")
        self._btn_action.clicked.connect(self._on_action)
        layout.addWidget(self._btn_action)

        self._done = False

    def _on_action(self) -> None:
        if self._done:
            self.accept()
        else:
            self._done = True
            self.cancel_requested.emit()
            self.reject()

    def reject(self) -> None:
        if not self._done:
            self.cancel_requested.emit()
        super().reject()

    @Slot(int, int, str)
    def update_progress(self, current: int, total: int, message: str) -> None:
        if total > 0:
            self._progress.setMaximum(total)
            self._progress.setValue(current)
        self._status.setText(message)
        if current == 0 and total == 0:
            self._instruction.setText(message)

    @Slot(bool, str)
    def on_done(self, success: bool, message: str) -> None:
        self._done = True
        if success:
            self._instruction.setText("校准完成！")
            self._instruction.setProperty("cal_state", "success")
            self._guidance.hide()
            self._status.setText(message)
            self._btn_action.setText("确定")
            self._btn_action.setAccessibleName("关闭校准对话框")
            self._progress.setValue(self._progress.maximum())
        else:
            self._instruction.setText("校准失败")
            self._instruction.setProperty("cal_state", "failure")
            self._guidance.hide()
            self._status.setText(message)
            self._btn_action.setText("关闭")
        repolish(self._instruction)

    @property
    def cancelled(self) -> bool:
        return not self._done
