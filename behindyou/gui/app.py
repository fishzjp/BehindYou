from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from behindyou import __version__
from behindyou.gui.main_window import MainWindow
from behindyou.gui.styles import build_palette, build_stylesheet, set_theme
from behindyou.gui.tray import create_app_icon


def _detect_dark_mode() -> bool:
    try:
        scheme = QGuiApplication.styleHints().colorScheme()
        return scheme == Qt.ColorScheme.Dark
    except AttributeError:
        return False


def gui_main() -> None:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("BehindYou")
    app.setApplicationVersion(__version__)
    app.setWindowIcon(create_app_icon())

    dark = _detect_dark_mode()
    set_theme(dark)
    app.setPalette(build_palette(dark))
    app.setStyleSheet(build_stylesheet(dark=dark))

    window = MainWindow()
    app.aboutToQuit.connect(window._stop_detection)

    def _on_app_state_changed(state: Qt.ApplicationState) -> None:
        if state == Qt.ApplicationState.ApplicationActive and not window.isVisible():
            window._show_from_tray()

    app.applicationStateChanged.connect(_on_app_state_changed)
    window.show()

    sys.exit(app.exec())
