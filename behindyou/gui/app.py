from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from behindyou import __version__
from behindyou.gui.main_window import MainWindow
from behindyou.gui.tray import create_app_icon


def gui_main() -> None:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("BehindYou")
    app.setApplicationVersion(__version__)
    app.setWindowIcon(create_app_icon())

    window = MainWindow()
    app.aboutToQuit.connect(window._stop_detection)

    def _on_app_state_changed(state: Qt.ApplicationState) -> None:
        if state == Qt.ApplicationState.ApplicationActive and not window.isVisible():
            window._show_from_tray()

    app.applicationStateChanged.connect(_on_app_state_changed)
    window.show()

    sys.exit(app.exec())
