from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from behindyou import __version__
from behindyou.gui.main_window import MainWindow
from behindyou.gui.styles import (
    COLOR_ACCENT,
    COLOR_ALTERNATE_BG,
    COLOR_BORDER,
    COLOR_PANEL_BG,
    COLOR_TEXT,
    COLOR_TEXT_SECONDARY,
    COLOR_WINDOW_BG,
    build_stylesheet,
)
from behindyou.gui.tray import create_app_icon


def _apply_palette(app: QApplication) -> None:
    pal = app.palette()
    pal.setColor(QPalette.ColorRole.Window, QColor(COLOR_WINDOW_BG))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(COLOR_TEXT))
    pal.setColor(QPalette.ColorRole.Base, QColor(COLOR_PANEL_BG))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(COLOR_ALTERNATE_BG))
    pal.setColor(QPalette.ColorRole.Text, QColor(COLOR_TEXT))
    pal.setColor(QPalette.ColorRole.Button, QColor(COLOR_PANEL_BG))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(COLOR_TEXT))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(COLOR_ACCENT))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(COLOR_TEXT_SECONDARY))
    pal.setColor(QPalette.ColorRole.Mid, QColor(COLOR_BORDER))
    app.setPalette(pal)


def gui_main() -> None:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("BehindYou")
    app.setApplicationVersion(__version__)
    app.setWindowIcon(create_app_icon())
    _apply_palette(app)
    app.setStyleSheet(build_stylesheet())

    window = MainWindow()
    app.aboutToQuit.connect(window._stop_detection)

    def _on_app_state_changed(state: Qt.ApplicationState) -> None:
        if state == Qt.ApplicationState.ApplicationActive and not window.isVisible():
            window._show_from_tray()

    app.applicationStateChanged.connect(_on_app_state_changed)
    window.show()

    sys.exit(app.exec())
