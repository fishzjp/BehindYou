from __future__ import annotations

import os
from datetime import datetime

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from behindyou.gui.styles import COLOR_ACCENT


class EventLog(QWidget):
    """Displays intrusion events with screenshot browsing."""

    screenshot_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(160)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Title bar
        title_bar = QWidget()
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(8, 6, 8, 6)

        title = QLabel("事件记录")
        title.setStyleSheet("font-weight: 600;")
        title_layout.addWidget(title)
        title_layout.addStretch()

        self._btn_clear = QPushButton("清空")
        self._btn_clear.setFlat(True)
        self._btn_clear.setStyleSheet(
            f"QPushButton {{ border: none; color: {COLOR_ACCENT}; font-size: 12px; }}"
            "QPushButton:hover { text-decoration: underline; }"
        )
        self._btn_clear.clicked.connect(self._clear_events)
        title_layout.addWidget(self._btn_clear)

        layout.addWidget(title_bar)

        # List widget
        self._list = QListWidget()
        self._list.setObjectName("event_log_list")
        self._list.setAlternatingRowColors(True)
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._list)

    _MAX_EVENTS = 500

    def add_event(self, intruder_count: int, screenshot_path: str | None) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"{timestamp} — {intruder_count} 人入侵"
        if screenshot_path:
            text += "  [截图]"
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, screenshot_path)
        if screenshot_path:
            item.setForeground(QColor(COLOR_ACCENT))
        self._list.insertItem(0, item)
        while self._list.count() > self._MAX_EVENTS:
            self._list.takeItem(self._list.count() - 1)

    def _clear_events(self) -> None:
        self._list.clear()

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
