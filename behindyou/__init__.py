"""BehindYou - 基于 YOLO 的实时身后人员检测系统。"""

__version__ = "0.1.0"

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def gui_main() -> None:
    from behindyou.gui.app import gui_main as _gui_main

    _gui_main()
