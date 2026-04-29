"""BehindYou - 基于 YOLO 的实时身后人员检测系统。"""

__version__ = "0.1.0"


def cli_main() -> None:
    import logging

    from behindyou.config import parse_args
    from behindyou.runner import run

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = parse_args()
    run(config)
