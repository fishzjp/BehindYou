# BehindYou

基于 YOLO 的实时身后人员检测系统。通过摄像头监控使用者背后区域，当陌生人靠近时自动发送系统通知。

## 功能特性

- **实时人物检测** — 基于 YOLO 目标检测模型，实时识别人物并追踪
- **主人识别** — 通过 InsightFace 人脸特征提取区分主人和陌生人，避免误报
- **智能追踪** — EMA 平滑追踪自身位置，支持快速移动后自动恢复
- **面部验证** — Haar Cascade 前脸检测过滤误检，减少非人物目标干扰
- **系统通知** — 支持 macOS (terminal-notifier / AppleScript) 和 Linux (notify-send) 原生通知，带截图保存
- **图形界面** — PySide6 GUI，支持实时视频预览、参数调节、事件日志、系统托盘
- **灵活配置** — 支持置信度阈值、检测帧持久性、最小面积、冷却时间等参数自定义

## 快速开始

```bash
uv sync
uv run behindyou
```

或使用模块方式：

```bash
uv run python -m behindyou
```

首次运行会自动校准，采集人脸数据。后续启动将快速校准并开始监控。

## GUI 使用

- **启动/停止** — 点击侧边栏按钮或使用快捷键 Space / Escape
- **校准** — 点击"校准"按钮重新采集人脸数据
- **参数调节** — 侧边栏滑块实时调整灵敏度、报警间隔等参数
- **事件日志** — 底部面板显示入侵事件，双击可查看截图
- **系统托盘** — 关闭窗口后程序最小化到托盘，右键菜单控制

## 项目结构

```
behindyou/
├── __init__.py          # 包入口，gui_main()
├── __main__.py          # python -m behindyou
├── config.py            # 配置管理（frozen dataclass）
├── detection.py         # YOLO 检测封装（ultralytics + supervision）
├── engine.py            # 核心检测引擎，process_frame()，annotate_frame()
├── worker.py            # QThread 工作线程
├── face.py              # 人脸检测（Haar Cascade）与识别（InsightFace）
├── tracking.py          # EMA 追踪、几何工具函数
├── notification.py      # 系统通知 + 截图保存
├── paths.py             # 路径常量（~/.behindyou/）
└── gui/
    ├── app.py           # QApplication 启动
    ├── main_window.py   # 主窗口，Worker 生命周期管理
    ├── settings_panel.py # 侧边栏参数配置
    ├── video_widget.py  # 实时视频显示
    ├── event_log.py     # 入侵事件列表 + 截图查看
    ├── tray.py          # 系统托盘图标 + 菜单
    └── calibration_dialog.py # 校准进度对话框
```
