"""Centralized QSS styles for macOS-native light theme."""

from __future__ import annotations

# ── macOS light palette colors ────────────────────────────────────────
COLOR_WINDOW_BG = "#f5f5f7"
COLOR_PANEL_BG = "#ffffff"
COLOR_BORDER = "#d2d2d7"
COLOR_TEXT = "#1d1d1f"
COLOR_TEXT_SECONDARY = "#86868b"
COLOR_ACCENT = "#0071e3"
COLOR_ACCENT_HOVER = "#0077ed"
COLOR_SUCCESS = "#34c759"
COLOR_DANGER = "#ff3b30"
COLOR_DANGER_HOVER = "#ff453a"
COLOR_SLIDER_GROOVE = "#d2d2d7"
COLOR_SLIDER_HANDLE = "#ffffff"
COLOR_ALTERNATE_BG = "#fafafa"
COLOR_DISABLED_BG = "#f5f5f7"
COLOR_DISABLED_TEXT = "#a1a1a6"
COLOR_HOVER_BG = "#e8e8ed"
COLOR_PRESSED_BG = "#d2d2d7"
COLOR_ACCENT_PRESSED = "#006edb"
COLOR_ACCENT_DISABLED = "#a1c9f2"
COLOR_VIDEO_BG = "#1e1e1e"
COLOR_ROW_BORDER = "#f0f0f2"
COLOR_SELECTED_BG = "#e8f0fe"
COLOR_STOP_HOVER_BG = "#fff0f0"
COLOR_STOP_PRESSED_BG = "#ffe0e0"
COLOR_STOP_DISABLED_TEXT = "#d4a0a0"


def build_stylesheet() -> str:
    """Return the global QSS stylesheet for the application."""
    return f"""
    /* ── Global ─────────────────────────────────────────────────── */
    QWidget {{
        color: {COLOR_TEXT};
        font-size: 13px;
    }}

    /* ── Scrollbar (thin, rounded) ─────────────────────────────── */
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {COLOR_BORDER};
        border-radius: 4px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {COLOR_DISABLED_TEXT};
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
        background: {COLOR_BORDER};
        border-radius: 4px;
        min-width: 30px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {COLOR_DISABLED_TEXT};
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
        background: {COLOR_PANEL_BG};
        border-right: 1px solid {COLOR_BORDER};
    }}
    SettingsPanel > QWidget {{
        background: {COLOR_PANEL_BG};
    }}

    /* ── Buttons ────────────────────────────────────────────────── */
    QPushButton {{
        border: 1px solid {COLOR_BORDER};
        border-radius: 6px;
        padding: 6px 14px;
        background: {COLOR_PANEL_BG};
        font-size: 13px;
        min-height: 20px;
    }}
    QPushButton:hover {{
        background: {COLOR_HOVER_BG};
    }}
    QPushButton:pressed {{
        background: {COLOR_PRESSED_BG};
    }}
    QPushButton:disabled {{
        background: {COLOR_DISABLED_BG};
        color: {COLOR_DISABLED_TEXT};
        border-color: {COLOR_HOVER_BG};
    }}

    QPushButton#btn_start {{
        background: {COLOR_ACCENT};
        color: white;
        border: none;
        font-weight: 600;
    }}
    QPushButton#btn_start:hover {{
        background: {COLOR_ACCENT_HOVER};
    }}
    QPushButton#btn_start:pressed {{
        background: {COLOR_ACCENT_PRESSED};
    }}
    QPushButton#btn_start:disabled {{
        background: {COLOR_ACCENT_DISABLED};
        color: white;
    }}

    QPushButton#btn_stop {{
        border: 1px solid {COLOR_DANGER};
        color: {COLOR_DANGER};
        background: {COLOR_PANEL_BG};
    }}
    QPushButton#btn_stop:hover {{
        background: {COLOR_STOP_HOVER_BG};
    }}
    QPushButton#btn_stop:pressed {{
        background: {COLOR_STOP_PRESSED_BG};
    }}
    QPushButton#btn_stop:disabled {{
        background: {COLOR_DISABLED_BG};
        color: {COLOR_STOP_DISABLED_TEXT};
        border-color: {COLOR_HOVER_BG};
    }}

    /* ── GroupBox (no border, left accent bar) ──────────────────── */
    QGroupBox {{
        background: {COLOR_PANEL_BG};
        border: none;
        border-left: 3px solid {COLOR_ACCENT};
        margin-top: 12px;
        padding: 8px 8px 8px 12px;
        font-weight: 600;
        font-size: 13px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 4px;
        color: {COLOR_TEXT};
    }}

    /* ── Slider ─────────────────────────────────────────────────── */
    QSlider::groove:horizontal {{
        height: 4px;
        background: {COLOR_SLIDER_GROOVE};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {COLOR_SLIDER_HANDLE};
        border: 1px solid {COLOR_BORDER};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{
        border-color: {COLOR_ACCENT};
    }}
    QSlider::sub-page:horizontal {{
        background: {COLOR_ACCENT};
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
        border: 1px solid {COLOR_BORDER};
        border-radius: 4px;
        background: {COLOR_PANEL_BG};
    }}
    QCheckBox::indicator:checked {{
        background: {COLOR_ACCENT};
        border-color: {COLOR_ACCENT};
    }}
    QCheckBox::indicator:hover {{
        border-color: {COLOR_ACCENT};
    }}

    /* ── Video Display ──────────────────────────────────────────── */
    VideoDisplay {{
        background-color: {COLOR_VIDEO_BG};
        border-radius: 8px;
        color: {COLOR_TEXT_SECONDARY};
        font-size: 15px;
    }}

    /* ── Event Log ──────────────────────────────────────────────── */
    EventLog {{
        background: {COLOR_PANEL_BG};
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
    }}
    EventLog QListWidget#event_log_list::item {{
        padding: 6px 8px;
        border-bottom: 1px solid {COLOR_ROW_BORDER};
    }}
    EventLog QListWidget#event_log_list::item:alternate {{
        background: {COLOR_ALTERNATE_BG};
    }}
    EventLog QListWidget#event_log_list::item:selected {{
        background: {COLOR_SELECTED_BG};
        color: {COLOR_TEXT};
    }}
    EventLog QListWidget#event_log_list::item:hover {{
        background: {COLOR_ROW_BORDER};
    }}

    /* ── Progress Bar ───────────────────────────────────────────── */
    QProgressBar {{
        border: none;
        border-radius: 4px;
        background: {COLOR_HOVER_BG};
        height: 8px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        background: {COLOR_ACCENT};
        border-radius: 4px;
    }}

    /* ── Status Bar ─────────────────────────────────────────────── */
    QStatusBar {{
        background: {COLOR_PANEL_BG};
        border-top: 1px solid {COLOR_BORDER};
        color: {COLOR_TEXT_SECONDARY};
        font-size: 12px;
    }}
    QStatusBar QLabel {{
        color: {COLOR_TEXT_SECONDARY};
        padding: 0 4px;
    }}

    /* ── Menu Bar ───────────────────────────────────────────────── */
    QMenuBar {{
        background: {COLOR_PANEL_BG};
        border-bottom: 1px solid {COLOR_BORDER};
    }}
    QMenuBar::item:selected {{
        background: {COLOR_HOVER_BG};
        border-radius: 4px;
    }}
    QMenu {{
        background: {COLOR_PANEL_BG};
        border: 1px solid {COLOR_BORDER};
        border-radius: 6px;
        padding: 4px 0;
    }}
    QMenu::item {{
        padding: 6px 24px 6px 12px;
    }}
    QMenu::item:selected {{
        background: {COLOR_ACCENT};
        color: white;
    }}

    /* ── Splitter ───────────────────────────────────────────────── */
    QSplitter::handle {{
        background: {COLOR_BORDER};
    }}
    QSplitter::handle:horizontal {{
        width: 1px;
    }}
    QSplitter::handle:vertical {{
        height: 1px;
    }}
    """
