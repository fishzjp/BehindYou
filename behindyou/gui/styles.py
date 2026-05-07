"""Centralized QSS styles with light/dark theme support."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QWidget

_LIGHT_COLORS: dict[str, str] = {
    "window_bg": "#f5f5f7",
    "panel_bg": "#ffffff",
    "border": "#d2d2d7",
    "text": "#1d1d1f",
    "text_secondary": "#86868b",
    "accent": "#0071e3",
    "accent_hover": "#0077ed",
    "success": "#34c759",
    "danger": "#ff3b30",
    "danger_hover": "#ff453a",
    "slider_groove": "#d2d2d7",
    "slider_handle": "#ffffff",
    "alternate_bg": "#fafafa",
    "disabled_bg": "#f5f5f7",
    "disabled_text": "#a1a1a6",
    "hover_bg": "#e8e8ed",
    "pressed_bg": "#d2d2d7",
    "accent_pressed": "#006edb",
    "accent_disabled": "#a1c9f2",
    "video_bg": "#1e1e1e",
    "row_border": "#f0f0f2",
    "selected_bg": "#e8f0fe",
    "stop_hover_bg": "#fff0f0",
    "stop_pressed_bg": "#ffe0e0",
    "stop_disabled_text": "#d4a0a0",
    "fps_text": "#86868b",
}

_DARK_COLORS: dict[str, str] = {
    "window_bg": "#1e1e1e",
    "panel_bg": "#2c2c2e",
    "border": "#3a3a3c",
    "text": "#ffffff",
    "text_secondary": "#98989d",
    "accent": "#0a84ff",
    "accent_hover": "#409cff",
    "success": "#30d158",
    "danger": "#ff453a",
    "danger_hover": "#ff6961",
    "slider_groove": "#3a3a3c",
    "slider_handle": "#e0e0e0",
    "alternate_bg": "#2c2c2e",
    "disabled_bg": "#2c2c2e",
    "disabled_text": "#636366",
    "hover_bg": "#3a3a3c",
    "pressed_bg": "#48484a",
    "accent_pressed": "#0070c0",
    "accent_disabled": "#3a6b99",
    "video_bg": "#000000",
    "row_border": "#333333",
    "selected_bg": "#2a4a6b",
    "stop_hover_bg": "#3a2020",
    "stop_pressed_bg": "#4a2020",
    "stop_disabled_text": "#6b4040",
    "fps_text": "#98989d",
}


_current_dark = False


def set_theme(dark: bool) -> None:
    global _current_dark
    _current_dark = dark


def current_colors() -> dict[str, str]:
    return _DARK_COLORS if _current_dark else _LIGHT_COLORS


def repolish(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def build_palette(dark: bool = False) -> QPalette:
    c = _DARK_COLORS if dark else _LIGHT_COLORS
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(c["window_bg"]))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(c["text"]))
    pal.setColor(QPalette.ColorRole.Base, QColor(c["panel_bg"]))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(c["alternate_bg"]))
    pal.setColor(QPalette.ColorRole.Text, QColor(c["text"]))
    pal.setColor(QPalette.ColorRole.Button, QColor(c["panel_bg"]))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(c["text"]))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(c["accent"]))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(c["text_secondary"]))
    pal.setColor(QPalette.ColorRole.Mid, QColor(c["border"]))
    return pal


def build_stylesheet(*, dark: bool = False) -> str:
    c = _DARK_COLORS if dark else _LIGHT_COLORS
    return f"""
    /* ── Global ─────────────────────────────────────────────────── */
    QWidget {{
        color: {c["text"]};
        font-size: 13px;
    }}

    /* ── Scrollbar (thin, rounded) ─────────────────────────────── */
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {c["border"]};
        border-radius: 4px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {c["disabled_text"]};
    }}
    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical {{
        height: 0;
        background: none;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 8px;
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: {c["border"]};
        border-radius: 4px;
        min-width: 30px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {c["disabled_text"]};
    }}
    QScrollBar::add-line:horizontal,
    QScrollBar::sub-line:horizontal,
    QScrollBar::add-page:horizontal,
    QScrollBar::sub-page:horizontal {{
        width: 0;
        background: none;
    }}

    /* ── Settings Panel ─────────────────────────────────────────── */
    SettingsPanel {{
        background: {c["panel_bg"]};
        border-right: 1px solid {c["border"]};
    }}
    SettingsPanel > QWidget {{
        background: {c["panel_bg"]};
    }}

    /* ── Buttons ────────────────────────────────────────────────── */
    QPushButton {{
        border: 1px solid {c["border"]};
        border-radius: 6px;
        padding: 6px 14px;
        background: {c["panel_bg"]};
        font-size: 13px;
        min-height: 20px;
    }}
    QPushButton:hover {{
        background: {c["hover_bg"]};
    }}
    QPushButton:pressed {{
        background: {c["pressed_bg"]};
    }}
    QPushButton:disabled {{
        background: {c["disabled_bg"]};
        color: {c["disabled_text"]};
        border-color: {c["hover_bg"]};
    }}

    QPushButton#btn_start {{
        background: {c["accent"]};
        color: white;
        border: none;
        font-weight: 600;
    }}
    QPushButton#btn_start:hover {{
        background: {c["accent_hover"]};
    }}
    QPushButton#btn_start:pressed {{
        background: {c["accent_pressed"]};
    }}
    QPushButton#btn_start:disabled {{
        background: {c["accent_disabled"]};
        color: white;
    }}

    QPushButton#btn_stop {{
        border: 1px solid {c["danger"]};
        color: {c["danger"]};
        background: {c["panel_bg"]};
    }}
    QPushButton#btn_stop:hover {{
        background: {c["stop_hover_bg"]};
    }}
    QPushButton#btn_stop:pressed {{
        background: {c["stop_pressed_bg"]};
    }}
    QPushButton#btn_stop:disabled {{
        background: {c["disabled_bg"]};
        color: {c["stop_disabled_text"]};
        border-color: {c["hover_bg"]};
    }}

    QPushButton#btn_clear {{
        border: none;
        color: {c["accent"]};
        font-size: 12px;
    }}
    QPushButton#btn_clear:hover {{
        text-decoration: underline;
    }}

    /* ── Status dot ─────────────────────────────────────────────── */
    QLabel#status_dot {{
        font-size: 14px;
    }}
    QLabel#status_dot[status="idle"] {{
        color: {c["text_secondary"]};
    }}
    QLabel#status_dot[status="running"] {{
        color: {c["success"]};
    }}

    /* ── Calibration guidance ───────────────────────────────────── */
    QLabel#calibration_guidance {{
        color: {c["text_secondary"]};
        font-size: 12px;
    }}

    /* ── Calibration instruction ────────────────────────────────── */
    QLabel#calibration_instruction {{
        font-size: 16px;
        font-weight: 600;
        margin: 8px 0;
    }}
    QLabel#calibration_instruction[cal_state="success"] {{
        color: {c["success"]};
    }}
    QLabel#calibration_instruction[cal_state="failure"] {{
        color: {c["danger"]};
    }}

    QLabel#calibration_status {{
        color: {c["text_secondary"]};
    }}

    /* ── Event log title ────────────────────────────────────────── */
    QLabel#event_log_title {{
        font-weight: 600;
    }}

    /* ── GroupBox (no border, left accent bar) ──────────────────── */
    QGroupBox {{
        background: {c["panel_bg"]};
        border: none;
        border-left: 3px solid {c["accent"]};
        margin-top: 12px;
        padding: 8px 8px 8px 12px;
        font-weight: 600;
        font-size: 13px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 4px;
        color: {c["text"]};
    }}

    /* ── Slider ─────────────────────────────────────────────────── */
    QSlider::groove:horizontal {{
        height: 4px;
        background: {c["slider_groove"]};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {c["slider_handle"]};
        border: 1px solid {c["border"]};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{
        border-color: {c["accent"]};
    }}
    QSlider::sub-page:horizontal {{
        background: {c["accent"]};
        border-radius: 2px;
    }}

    /* ── CheckBox ───────────────────────────────────────────────── */
    QCheckBox {{
        spacing: 6px;
        padding: 4px 0;
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {c["border"]};
        border-radius: 4px;
        background: {c["panel_bg"]};
    }}
    QCheckBox::indicator:checked {{
        background: {c["accent"]};
        border-color: {c["accent"]};
    }}
    QCheckBox::indicator:hover {{
        border-color: {c["accent"]};
    }}

    /* ── Video Display ──────────────────────────────────────────── */
    VideoDisplay {{
        background-color: {c["video_bg"]};
        border-radius: 8px;
        color: {c["text_secondary"]};
        font-size: 15px;
    }}

    /* ── Event Log ──────────────────────────────────────────────── */
    EventLog {{
        background: {c["panel_bg"]};
        border: 1px solid {c["border"]};
        border-radius: 8px;
    }}
    EventLog QListWidget#event_log_list::item {{
        padding: 6px 8px;
        border-bottom: 1px solid {c["row_border"]};
    }}
    EventLog QListWidget#event_log_list::item:alternate {{
        background: {c["alternate_bg"]};
    }}
    EventLog QListWidget#event_log_list::item:selected {{
        background: {c["selected_bg"]};
        color: {c["text"]};
    }}
    EventLog QListWidget#event_log_list::item:hover {{
        background: {c["row_border"]};
    }}

    /* ── Progress Bar ───────────────────────────────────────────── */
    QProgressBar {{
        border: none;
        border-radius: 4px;
        background: {c["hover_bg"]};
        height: 8px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        background: {c["accent"]};
        border-radius: 4px;
    }}

    /* ── Status Bar ─────────────────────────────────────────────── */
    QStatusBar {{
        background: {c["panel_bg"]};
        border-top: 1px solid {c["border"]};
        color: {c["text_secondary"]};
        font-size: 12px;
    }}
    QStatusBar QLabel {{
        color: {c["text_secondary"]};
        padding: 0 4px;
    }}

    /* ── Menu Bar ───────────────────────────────────────────────── */
    QMenuBar {{
        background: {c["panel_bg"]};
        border-bottom: 1px solid {c["border"]};
    }}
    QMenuBar::item:selected {{
        background: {c["hover_bg"]};
        border-radius: 4px;
    }}
    QMenu {{
        background: {c["panel_bg"]};
        border: 1px solid {c["border"]};
        border-radius: 6px;
        padding: 4px 0;
    }}
    QMenu::item {{
        padding: 6px 24px 6px 12px;
    }}
    QMenu::item:selected {{
        background: {c["accent"]};
        color: white;
    }}

    /* ── Splitter ───────────────────────────────────────────────── */
    QSplitter::handle {{
        background: {c["border"]};
    }}
    QSplitter::handle:horizontal {{
        width: 8px;
    }}
    QSplitter::handle:vertical {{
        height: 8px;
    }}
    QSplitter::handle:hover {{
        background: {c["accent"]};
    }}

    /* ── FPS Label ─────────────────────────────────────────────── */
    QLabel#fps_label {{
        color: {c["fps_text"]};
        font-family: monospace;
        font-size: 11px;
        padding: 0 6px;
    }}
    """
