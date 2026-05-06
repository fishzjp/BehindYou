from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path.home() / ".behindyou"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
FACE_DATA_FILE = DATA_DIR / "owner_face.npy"
MODEL_FILE = DATA_DIR / "yolo26n.pt"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
