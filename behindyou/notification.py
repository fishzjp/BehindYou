from __future__ import annotations

import datetime
import logging
import shutil
import subprocess
import sys

import cv2
import numpy as np

from behindyou.paths import SCREENSHOTS_DIR as _SCREENSHOTS_DIR_PATH

logger = logging.getLogger(__name__)

SCREENSHOTS_DIR = _SCREENSHOTS_DIR_PATH

_HAS_TERMINAL_NOTIFIER = shutil.which("terminal-notifier") is not None
if not _HAS_TERMINAL_NOTIFIER and sys.platform == "darwin":
    logger.info("terminal-notifier 未安装，将使用 osascript 发送通知（brew install terminal-notifier 可获得截图预览）")

_NOTIFICATION_SOUND = "Glass"


def _escape_applescript(s: str) -> str:
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\t", "\\t")
    return s


def _popen_silent(args: list[str]) -> None:
    subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def save_screenshot(annotated_frame: np.ndarray) -> str | None:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"alert_{timestamp}.jpg"
    path = SCREENSHOTS_DIR / filename
    if not cv2.imwrite(str(path), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        logger.warning("截图保存失败：%s", path)
        return None
    logger.info("截图已保存：%s", path)
    return str(path)


def send_notification(person_count: int, screenshot_path: str | None = None) -> None:
    title = "BehindYou"
    message = f"检测到 {person_count} 个人出现在你身后"

    if sys.platform == "darwin":
        if _HAS_TERMINAL_NOTIFIER and screenshot_path:
            _popen_silent([
                "terminal-notifier",
                "-title", title,
                "-message", message,
                "-contentImage", screenshot_path,
                "-sound", _NOTIFICATION_SOUND,
            ])
        else:
            script = (
                f'display notification "{_escape_applescript(message)}" '
                f'with title "{_escape_applescript(title)}" sound name "{_NOTIFICATION_SOUND}"'
            )
            _popen_silent(["osascript", "-e", script])
    elif sys.platform.startswith("linux"):
        cmd = ["notify-send"]
        if screenshot_path:
            cmd += ["-i", screenshot_path]
        cmd += [title, message]
        _popen_silent(cmd)
    else:
        logger.info("[通知] %s %s", title, message)
