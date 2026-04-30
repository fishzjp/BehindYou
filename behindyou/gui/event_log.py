from __future__ import annotations

import os
from datetime import datetime

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QListWidget, QListWidgetItem


class EventLog(QListWidget):
    """Displays intrusion events with screenshot browsing."""

    screenshot_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)

    _MAX_EVENTS = 500

    def add_event(self, intruder_count: int, screenshot_path: str | None) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"{timestamp} - {intruder_count} 人入侵"
        if screenshot_path:
            text += " [截图]"
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, screenshot_path)
        self.insertItem(0, item)
        while self.count() > self._MAX_EVENTS:
            self.takeItem(self.count() - 1)

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and os.path.exists(path):
            self.screenshot_requested.emit(path)


class ScreenshotViewer:
    """Opens screenshots using the system default viewer."""

    @staticmethod
    def open(path: str) -> None:
        if os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
