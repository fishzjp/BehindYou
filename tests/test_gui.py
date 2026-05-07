from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

from behindyou.gui.calibration_dialog import CalibrationDialog
from behindyou.gui.event_log import EventLog
from behindyou.gui.settings_panel import SettingsPanel
from behindyou.gui.styles import build_palette, build_stylesheet, current_colors


class TestStyles:
    def test_light_stylesheet(self):
        ss = build_stylesheet(dark=False)
        assert "QWidget" in ss
        assert ss  # non-empty

    def test_dark_stylesheet(self):
        ss = build_stylesheet(dark=True)
        assert "QWidget" in ss
        assert "#0a84ff" in ss

    def test_build_palette_light(self):
        pal = build_palette(dark=False)
        assert pal is not None

    def test_build_palette_dark(self):
        pal = build_palette(dark=True)
        assert pal is not None

    def test_current_colors_returns_dict(self):
        c = current_colors()
        assert "accent" in c
        assert "text" in c
        assert "window_bg" in c


class TestSettingsPanel:
    def test_defaults(self, qapp):
        panel = SettingsPanel()
        config = panel.get_config()
        assert config.camera == 0
        assert config.confidence == 0.6
        assert config.cooldown == 10.0
        assert config.no_face_check is False

    def test_set_running(self, qapp):
        panel = SettingsPanel()
        panel.set_running(True)
        assert not panel._btn_start.isEnabled()
        assert panel._btn_stop.isEnabled()
        assert not panel._btn_calibrate.isEnabled()
        panel.set_running(False)
        assert panel._btn_start.isEnabled()
        assert not panel._btn_stop.isEnabled()
        assert panel._btn_calibrate.isEnabled()

    def test_slider_value(self, qapp):
        panel = SettingsPanel()
        panel._camera.set_value(2)
        assert panel._camera.value == 2

    def test_no_face_check_hides_slider(self, qapp):
        panel = SettingsPanel()
        panel.show()
        panel._no_face_check.setChecked(True)
        assert not panel._face_det_score.isVisible()
        panel._no_face_check.setChecked(False)
        assert panel._face_det_score.isVisible()


class TestEventLog:
    def test_add_event(self, qapp):
        log = EventLog()
        log.add_event(1, None)
        assert log._list.count() == 1

    def test_add_event_with_screenshot(self, qapp):
        log = EventLog()
        log.add_event(2, "/tmp/test.png")
        assert log._list.count() == 1
        item = log._list.item(0)
        assert "截图" in item.text()

    def test_max_events(self, qapp):
        log = EventLog()
        for i in range(600):
            log.add_event(1, None)
        assert log._list.count() == 500

    def test_clear_events(self, qapp, mocker):
        log = EventLog()
        log.add_event(1, None)
        log.add_event(2, None)
        mocker.patch(
            "behindyou.gui.event_log.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        )
        log._clear_events()
        assert log._list.count() == 1  # placeholder restored


class TestCalibrationDialog:
    def test_initial_state(self, qapp):
        from PySide6.QtWidgets import QWidget

        parent = QWidget()
        dlg = CalibrationDialog(parent)
        assert not dlg._done
        assert dlg._btn_action.text() == "取消"
        parent.deleteLater()

    def test_on_done_success(self, qapp):
        from PySide6.QtWidgets import QWidget

        parent = QWidget()
        dlg = CalibrationDialog(parent)
        dlg.on_done(True, "校准成功")
        assert dlg._done
        assert dlg._btn_action.text() == "确定"
        parent.deleteLater()

    def test_on_done_failure(self, qapp):
        from PySide6.QtWidgets import QWidget

        parent = QWidget()
        dlg = CalibrationDialog(parent)
        dlg.on_done(False, "未检测到人脸")
        assert dlg._done
        assert dlg._btn_action.text() == "关闭"
        parent.deleteLater()

    def test_cancelled_property(self, qapp):
        from PySide6.QtWidgets import QWidget

        parent = QWidget()
        dlg = CalibrationDialog(parent)
        assert dlg.cancelled is True
        dlg.on_done(True, "ok")
        assert dlg.cancelled is False
        parent.deleteLater()
