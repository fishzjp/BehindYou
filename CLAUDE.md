# BehindYou - Claude Code 项目约束

## 项目概述

基于 YOLO 的实时身后人员检测系统。通过摄像头监控背后区域，陌生人靠近时自动通知。

**技术栈**: Python 3.10+, PySide6 GUI, YOLO (ultralytics), InsightFace 人脸识别, OpenCV, supervision

## 编码规范

### 风格
- 使用 `from __future__ import annotations` 启用延迟类型注解
- 行宽限制: 100 字符 (ruff 配置)
- 目标 Python 版本: 3.10
- 类型注解: 使用 `TYPE_CHECKING` 守卫导入仅用于类型提示的模块
- 日志: 使用标准库 `logging`，每个模块 `logger = logging.getLogger(__name__)`

### 结构约定
- 配置使用 `@dataclass(frozen=True)` 不可变数据类
- GUI 相关代码放在 `behindyou/gui/` 子包
- 核心逻辑与 GUI 分离 (engine.py vs worker.py)

### 命名
- 模块/函数: snake_case
- 类: PascalCase
- 常量: UPPER_SNAKE_CASE
- 私有: 前缀单下划线 `_name`

## 常用命令

```bash
# 运行应用
uv run behindyou
uv run python -m behindyou

# 安装依赖
uv sync

# 代码检查
uv run ruff check .
uv run ruff format .

# 运行测试
uv run pytest tests/

# 单个测试
uv run pytest tests/test_config.py -v
```

## 项目结构

```
behindyou/
├── __init__.py          # 包入口，gui_main()
├── __main__.py          # python -m behindyou
├── config.py            # 配置管理 (frozen dataclass)
├── detection.py         # YOLO 检测封装
├── engine.py            # 核心检测引擎，process_frame()，annotate_frame()
├── worker.py            # QThread 工作线程
├── face.py              # 人脸识别 (InsightFace)
├── tracking.py          # EMA 追踪、几何工具
├── notification.py      # 系统通知 + 截图
├── paths.py             # 路径常量 (~/.behindyou/)
└── gui/                 # PySide6 GUI 组件
    ├── app.py           # QApplication 启动
    ├── main_window.py   # 主窗口
    ├── settings_panel.py # 参数配置面板
    ├── video_widget.py  # 视频显示
    ├── event_log.py     # 事件列表
    ├── tray.py          # 系统托盘
    └── calibration_dialog.py # 校准对话框
```

## 关键设计决策

1. **主人识别**: InsightFace 人脸特征匹配，阈值可配置 (默认 0.55)
2. **追踪算法**: EMA 平滑自身位置，防止快速移动误报
3. **检测过滤**: 置信度 + 最小面积 + 持续帧数三重过滤
4. **通知冷却**: 防止频繁报警，间隔可配置 (默认 10 秒)

## 已知约束

- macOS 为主要平台，通知依赖 terminal-notifier 或 AppleScript
- 模型文件 `yolo26n.pt` 不入 git (.gitignore)
- 人脸数据存储在 `~/.behindyou/` 用户目录
- PySide6 GUI 线程需与检测线程分离 (Worker QThread)

## 错误记录

**约束**: 每次犯错必须在此处记录，避免后续重复犯同样的错误。

| 日期 | 错误描述 | 原因分析 | 修正方案 |
|------|----------|----------|----------|
| | | | |
