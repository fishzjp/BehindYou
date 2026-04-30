from __future__ import annotations

from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QAction, QColor, QIcon, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon


def _create_icon() -> QIcon:
    """Create a simple colored square icon for the tray."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(QColor(0, 120, 215))
    return QIcon(pixmap)


class TrayIcon(QSystemTrayIcon):
    """System tray icon with context menu."""

    show_window_requested = Signal()
    start_requested = Signal()
    stop_requested = Signal()
    quit_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setIcon(_create_icon())
        self.setToolTip("BehindYou - 身后人员检测")

        menu = QMenu()

        action_show = QAction("显示窗口", menu)
        action_show.triggered.connect(self.show_window_requested)
        menu.addAction(action_show)

        menu.addSeparator()

        self._action_start = QAction("启动检测", menu)
        self._action_start.triggered.connect(self.start_requested)
        menu.addAction(self._action_start)

        self._action_stop = QAction("停止检测", menu)
        self._action_stop.triggered.connect(self.stop_requested)
        self._action_stop.setEnabled(False)
        menu.addAction(self._action_stop)

        menu.addSeparator()

        action_quit = QAction("退出", menu)
        action_quit.triggered.connect(self.quit_requested)
        menu.addAction(action_quit)

        self.setContextMenu(menu)
        self.activated.connect(self._on_activated)

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.show_window_requested.emit()

    @Slot(int)
    def show_intrusion_alert(self, count: int) -> None:
        self.showMessage(
            "BehindYou",
            f"检测到 {count} 个人出现在你身后",
            QSystemTrayIcon.MessageIcon.Warning,
            5000,
        )

    def set_running(self, running: bool) -> None:
        self._action_start.setEnabled(not running)
        self._action_stop.setEnabled(running)
