from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)


class CalibrationDialog(QDialog):
    """Guided calibration wizard with visual progress."""

    cancel_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("校准 - BehindYou")
        self.setMinimumSize(400, 200)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint
        )

        layout = QVBoxLayout(self)

        self._instruction = QLabel("正在准备校准...")
        self._instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._instruction.setStyleSheet("font-size: 14px; margin: 10px;")
        layout.addWidget(self._instruction)

        self._progress = QProgressBar()
        self._progress.setMinimum(0)
        self._progress.setMaximum(30)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        self._btn_action = QPushButton("取消")
        self._btn_action.clicked.connect(self._on_action)
        layout.addWidget(self._btn_action)

        self._done = False

    def _on_action(self) -> None:
        if self._done:
            self.accept()
        else:
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
            self._status.setText(message)
            self._btn_action.setText("确定")
            self._progress.setValue(self._progress.maximum())
        else:
            self._instruction.setText("校准失败")
            self._status.setText(message)
            self._btn_action.setText("关闭")

    @property
    def cancelled(self) -> bool:
        return not self._done
