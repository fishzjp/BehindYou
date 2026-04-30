from __future__ import annotations

import dataclasses
import logging

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QSystemTrayIcon,
    QWidget,
)

from behindyou.config import Config
from behindyou.gui.calibration_dialog import CalibrationDialog
from behindyou.gui.event_log import EventLog, ScreenshotViewer
from behindyou.gui.settings_panel import SettingsPanel
from behindyou.gui.tray import TrayIcon
from behindyou.gui.video_widget import VideoDisplay
from behindyou.worker import DetectionWorker

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BehindYou - 身后人员检测")
        self.setMinimumSize(1000, 700)

        self._worker: DetectionWorker | None = None
        self._calibration_dialog: CalibrationDialog | None = None

        self._setup_ui()
        self._setup_menu()
        self._setup_status_bar()
        self._setup_tray()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._settings = SettingsPanel()
        self._settings.start_requested.connect(self._start_detection)
        self._settings.stop_requested.connect(self._stop_detection)
        self._settings.calibrate_requested.connect(self._run_calibration)
        self._settings.config_changed.connect(self._on_config_changed)
        splitter.addWidget(self._settings)

        right_splitter = QSplitter(Qt.Orientation.Vertical)

        self._video = VideoDisplay()
        right_splitter.addWidget(self._video)

        self._event_log = EventLog()
        self._event_log.screenshot_requested.connect(ScreenshotViewer.open)
        right_splitter.addWidget(self._event_log)

        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)

        splitter.addWidget(right_splitter)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    def _setup_menu(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("文件")
        action_quit = QAction("退出", self)
        action_quit.setShortcut(QKeySequence("Ctrl+Q"))
        action_quit.triggered.connect(self._quit)
        file_menu.addAction(action_quit)

        det_menu = menu_bar.addMenu("检测")
        self._action_start = QAction("启动", self)
        self._action_start.setShortcut(QKeySequence("Space"))
        self._action_start.triggered.connect(self._start_detection)
        det_menu.addAction(self._action_start)

        self._action_stop = QAction("停止", self)
        self._action_stop.setShortcut(QKeySequence("Escape"))
        self._action_stop.triggered.connect(self._stop_detection)
        self._action_stop.setEnabled(False)
        det_menu.addAction(self._action_stop)

        det_menu.addSeparator()

        action_calibrate = QAction("重新校准", self)
        action_calibrate.triggered.connect(self._run_calibration)
        det_menu.addAction(action_calibrate)

        help_menu = menu_bar.addMenu("帮助")
        action_about = QAction("关于", self)
        action_about.triggered.connect(self._show_about)
        help_menu.addAction(action_about)

    def _setup_status_bar(self) -> None:
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_label = QLabel("就绪")
        self._status_bar.addWidget(self._status_label)

    def _setup_tray(self) -> None:
        self._tray = TrayIcon(self)
        self._tray.show_window_requested.connect(self._show_from_tray)
        self._tray.start_requested.connect(self._start_detection)
        self._tray.stop_requested.connect(self._stop_detection)
        self._tray.quit_requested.connect(self._quit)
        self._tray.show()

    def _connect_worker_signals(self, worker: DetectionWorker) -> None:
        worker.frame_ready.connect(self._video.update_frame)
        worker.intrusion_detected.connect(self._on_intrusion)
        worker.engine_error.connect(self._on_engine_error)
        worker.status_changed.connect(self._on_status_changed)

    def _start_detection(self) -> None:
        if self._worker is not None:
            return

        config = self._settings.get_config()
        self._worker = DetectionWorker(config)

        self._connect_worker_signals(self._worker)
        self._worker.calibration_progress.connect(self._on_calibration_progress)
        self._worker.calibration_done.connect(self._on_calibration_done)
        self._worker.finished.connect(self._on_worker_finished)

        self._set_ui_running(True)
        self._status_label.setText("正在启动...")
        self._worker.start()

    def _stop_detection(self) -> None:
        if self._worker is None:
            return
        self._worker.stop()
        self._set_ui_running(False)
        self._status_label.setText("已停止")

    def _set_ui_running(self, running: bool) -> None:
        self._settings.set_running(running)
        self._action_start.setEnabled(not running)
        self._action_stop.setEnabled(running)
        self._tray.set_running(running)

    @Slot()
    def _on_worker_finished(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    def _run_calibration(self) -> None:
        if self._worker is not None:
            QMessageBox.information(self, "提示", "请先停止检测再进行校准")
            return

        self._calibration_dialog = CalibrationDialog(self)
        self._calibration_dialog.cancel_requested.connect(self._on_calibration_cancel)
        self._calibration_dialog.finished.connect(self._on_calibration_dialog_finished)

        config = dataclasses.replace(self._settings.get_config(), recalibrate=True)
        self._worker = DetectionWorker(config)

        self._connect_worker_signals(self._worker)
        self._worker.calibration_progress.connect(
            self._calibration_dialog.update_progress
        )
        self._worker.calibration_done.connect(self._calibration_dialog.on_done)
        self._worker.calibration_done.connect(self._on_calibration_done)
        self._worker.finished.connect(self._on_worker_finished)

        self._set_ui_running(True)
        self._status_label.setText("校准中...")
        self._calibration_dialog.show()
        self._worker.start()

    @Slot()
    def _on_calibration_cancel(self) -> None:
        if self._worker is not None:
            self._worker.stop()

    @Slot()
    def _on_calibration_dialog_finished(self) -> None:
        if self._calibration_dialog is not None:
            self._calibration_dialog.deleteLater()
            self._calibration_dialog = None

    @Slot(bool, str)
    def _on_calibration_done(self, success: bool, message: str) -> None:
        if success:
            self._status_label.setText("校准完成，检测运行中")
            self._set_ui_running(True)
        else:
            self._stop_detection()
            self._status_label.setText(f"校准失败: {message}")
            QMessageBox.warning(self, "校准失败", message)

    @Slot(int, object)
    def _on_intrusion(self, count: int, screenshot_path: str | None) -> None:
        self._event_log.add_event(count, screenshot_path)
        self._tray.show_intrusion_alert(count)

    @Slot(str)
    def _on_engine_error(self, message: str) -> None:
        self._status_label.setText(f"错误: {message}")
        logger.error("引擎错误: %s", message)
        self._set_ui_running(False)
        if self._worker is not None:
            self._worker.stop()

    @Slot(str)
    def _on_status_changed(self, status: str) -> None:
        self._status_label.setText(status)

    @Slot(int, int, str)
    def _on_calibration_progress(self, current: int, total: int, msg: str) -> None:
        self._status_label.setText(msg)

    @Slot(object)
    def _on_config_changed(self, config: Config) -> None:
        if self._worker is not None:
            self._worker.config_requested.emit(config)

    def _show_from_tray(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def _quit(self) -> None:
        self._stop_detection()
        self._tray.hide()
        QApplication.quit()

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "关于 BehindYou",
            "BehindYou v0.1.0\n\n"
            "基于 YOLO 的实时身后人员检测系统\n"
            "监控摄像头画面，检测陌生人靠近并发送通知",
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._tray.isVisible() and QSystemTrayIcon.isSystemTrayAvailable():
            event.ignore()
            self.hide()
            self._tray.showMessage(
                "BehindYou",
                "程序已最小化到系统托盘",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )
        else:
            event.accept()
            self._quit()
