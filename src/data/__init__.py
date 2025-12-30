"""
数据管理模块

包含：
- DataManager: 统一数据访问接口
- TushareFetcher: Tushare Pro 数据获取器
- CacheManager: SQLite 缓存管理器
"""
from .data_manager import DataManager

__all__ = ['DataManager']
