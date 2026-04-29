from __future__ import annotations

import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def _escape_applescript(s: str) -> str:
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    return s


def send_notification(person_count: int) -> None:
    title = "BehindYou"
    message = f"检测到 {person_count} 个人出现在你身后"
    if sys.platform == "darwin":
        script = (
            f'display notification "{_escape_applescript(message)}" '
            f'with title "{_escape_applescript(title)}" sound name "Glass"'
        )
        subprocess.Popen(
            ["osascript", "-e", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    elif sys.platform.startswith("linux"):
        subprocess.Popen(
            ["notify-send", title, message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        logger.info("[通知] %s %s", title, message)
