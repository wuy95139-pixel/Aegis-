"""
日志模块
=======
统一的日志配置，支持控制台输出和文件滚动存储。
支持 text 和 json 两种输出格式。
"""

import json
import logging
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


class JSONFormatter(logging.Formatter):
    """JSON 结构化日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "funcName": record.funcName,
            "file": f"{record.pathname}:{record.lineno}",
            "thread": record.threadName,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False, default=str)


def setup_logger(
    name: str = "aegis",
    level: str = "INFO",
    log_dir: str = "./logs",
    log_file: str = "aegis.log",
    json_format: bool = False,
) -> logging.Logger:
    """
    初始化日志系统

    输出目标:
      - 控制台 (text 或 json 格式)
      - 文件 (自动轮转，单文件最大 10MB，保留 5 个备份)

    Args:
        name: logger 名称
        level: 日志级别
        log_dir: 日志目录
        log_file: 日志文件名
        json_format: 是否使用 JSON 格式 (默认 False)
                     也可通过环境变量 AEGIS_LOG_FORMAT=json 启用
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 环境变量覆盖 JSON 格式
    if os.environ.get("AEGIS_LOG_FORMAT", "").lower() == "json":
        json_format = True

    # --- 控制台 Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    if json_format:
        console_fmt = JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
    else:
        console_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # --- 文件 Handler (带轮转) ---
    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path / log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        if json_format:
            file_fmt = JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        else:
            file_fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    except Exception:
        logger.warning("Failed to setup file logging, continuing with console only.")

    return logger


# 默认 logger 实例
logger = setup_logger()
