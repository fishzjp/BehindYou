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
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from behindyou.gui.styles import current_colors


class EventLog(QWidget):
    """Displays intrusion events with screenshot browsing."""

    screenshot_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setAccessibleName("事件记录")
        self._accent_color = current_colors()["accent"]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Title bar
        title_bar = QWidget()
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(8, 6, 8, 6)

        title = QLabel("事件记录")
        title.setObjectName("event_log_title")
        title_layout.addWidget(title)
        title_layout.addStretch()

        self._btn_clear = QPushButton("清空")
        self._btn_clear.setObjectName("btn_clear")
        self._btn_clear.setFlat(True)
        self._btn_clear.setAccessibleName("清空事件记录")
        self._btn_clear.setToolTip("清空所有事件记录")
        self._btn_clear.clicked.connect(self._clear_events)
        title_layout.addWidget(self._btn_clear)

        layout.addWidget(title_bar)

        # List widget
        self._list = QListWidget()
        self._list.setObjectName("event_log_list")
        self._list.setAlternatingRowColors(True)
        self._list.setAccessibleName("入侵事件列表")
        self._list.setToolTip("双击带有 [截图] 标记的事件可查看截图")
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._list)

        self._set_placeholder()

    _MAX_EVENTS = 500

    _PLACEHOLDER_SENTINEL = False

    def _set_placeholder(self) -> None:
        self._list.clear()
        placeholder = QListWidgetItem("暂无入侵事件 — 启动检测后将在此显示记录")
        placeholder.setFlags(placeholder.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        placeholder.setForeground(QColor(self._accent_color))
        placeholder.setData(Qt.ItemDataRole.UserRole, self._PLACEHOLDER_SENTINEL)
        self._list.addItem(placeholder)

    def add_event(self, intruder_count: int, screenshot_path: str | None) -> None:
        if (
            self._list.count() == 1
            and self._list.item(0).data(Qt.ItemDataRole.UserRole) is self._PLACEHOLDER_SENTINEL
        ):
            self._list.clear()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"{timestamp} — {intruder_count} 人入侵"
        if screenshot_path:
            text += "  [截图]"
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, screenshot_path)
        if screenshot_path:
            item.setForeground(QColor(self._accent_color))
        self._list.insertItem(0, item)
        while self._list.count() > self._MAX_EVENTS:
            self._list.takeItem(self._list.count() - 1)

    def _clear_events(self) -> None:
        if self._list.count() == 0:
            return
        reply = QMessageBox.question(
            self,
            "确认清空",
            "确定要清空所有事件记录吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._set_placeholder()

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
