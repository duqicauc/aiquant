"""
日志工具
"""
import sys
from loguru import logger
from pathlib import Path

from config.config import config, LOG_DIR


def setup_logger(
    name: str = "aiquant",
    level: str = None,
    log_file: str = None
):
    """
    配置日志系统
    
    Args:
        name: 日志名称
        level: 日志级别
        log_file: 日志文件名
    """
    if level is None:
        level = config.LOG_LEVEL
    
    # 移除默认配置
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sys.stderr,
        format=config.LOG_FORMAT,
        level=level,
        colorize=True
    )
    
    # 添加文件输出
    if log_file is None:
        log_file = f"{name}.log"
    
    log_path = LOG_DIR / log_file
    logger.add(
        log_path,
        format=config.LOG_FORMAT,
        level=level,
        rotation="100 MB",  # 文件大小达到100MB时轮转
        retention="30 days",  # 保留30天
        compression="zip"  # 压缩旧日志
    )
    
    logger.info(f"日志系统已初始化: {log_path}")
    
    return logger


# 全局logger实例
log = setup_logger()

