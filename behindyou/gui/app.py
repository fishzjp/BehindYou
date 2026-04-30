from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from behindyou import __version__
from behindyou.gui.main_window import MainWindow


def gui_main() -> None:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("BehindYou")
    app.setApplicationVersion(__version__)

    window = MainWindow()
    app.aboutToQuit.connect(window._stop_detection)
    window.show()

    sys.exit(app.exec())
