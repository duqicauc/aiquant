"""
日期工具
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import List


def get_trading_dates(start_date: str, end_date: str) -> List[str]:
    """
    获取交易日列表（简化版，实际应该从数据库或API获取）
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        交易日列表
    """
    # TODO: 从数据库获取实际的交易日历
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # B表示工作日
    return [d.strftime('%Y%m%d') for d in dates]


def format_date(date: str, output_format: str = '%Y%m%d') -> str:
    """
    格式化日期
    
    Args:
        date: 日期字符串
        output_format: 输出格式
        
    Returns:
        格式化后的日期
    """
    # 尝试多种输入格式
    formats = ['%Y%m%d', '%Y-%m-%d', '%Y/%m/%d']
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date, fmt)
            return dt.strftime(output_format)
        except ValueError:
            continue
    
    raise ValueError(f"无法识别的日期格式: {date}")


def get_recent_date(days_ago: int = 0) -> str:
    """
    获取最近的日期
    
    Args:
        days_ago: 几天前（0表示今天）
        
    Returns:
        日期字符串（YYYYMMDD）
    """
    date = datetime.now() - timedelta(days=days_ago)
    return date.strftime('%Y%m%d')


def is_trading_day(date: str) -> bool:
    """
    判断是否为交易日
    
    Args:
        date: 日期字符串
        
    Returns:
        是否为交易日
    """
    # TODO: 从交易日历表查询
    dt = datetime.strptime(format_date(date), '%Y%m%d')
    # 简化判断：周一至周五
    return dt.weekday() < 5

