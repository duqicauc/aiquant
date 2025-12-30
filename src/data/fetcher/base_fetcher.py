"""
数据获取基类
"""
from abc import ABC, abstractmethod
from typing import Optional, List
import pandas as pd
from datetime import datetime


class BaseFetcher(ABC):
    """数据获取基类"""
    
    def __init__(self, source_name: str):
        """
        初始化数据获取器
        
        Args:
            source_name: 数据源名称
        """
        self.source_name = source_name
        self._init_connection()
    
    @abstractmethod
    def _init_connection(self):
        """初始化连接"""
        pass
    
    @abstractmethod
    def get_stock_list(self, **kwargs) -> pd.DataFrame:
        """
        获取股票列表
        
        Returns:
            包含股票代码、名称等信息的DataFrame
        """
        pass
    
    @abstractmethod
    def get_daily_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: Optional[str] = None,
        adjust: str = 'qfq'
    ) -> pd.DataFrame:
        """
        获取日线数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权类型
            
        Returns:
            日线数据DataFrame
        """
        pass
    
    @abstractmethod
    def get_minute_data(
        self,
        stock_code: str,
        freq: str = '5min',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取分钟数据
        
        Args:
            stock_code: 股票代码
            freq: 频率（1min, 5min, 15min, 30min, 60min）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            分钟数据DataFrame
        """
        pass
    
    @abstractmethod
    def get_fundamental_data(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取基本面数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            基本面数据DataFrame
        """
        pass
    
    @abstractmethod
    def get_daily_basic(
        self,
        stock_code: str,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取每日指标（市值、PE、PB等）
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            每日指标DataFrame
        """
        pass
    
    def format_stock_code(self, code: str) -> str:
        """
        格式化股票代码
        
        Args:
            code: 股票代码
            
        Returns:
            格式化后的代码
        """
        code = code.strip().upper()
        
        if '.' in code:
            return code
        
        if code.startswith('6'):
            return f"{code}.SH"
        elif code.startswith(('0', '3')):
            return f"{code}.SZ"
        elif code.startswith(('4', '8', '9')):
            return f"{code}.BJ"
        else:
            raise ValueError(f"无法识别的股票代码: {code}")
    
    def format_date(self, date: str) -> str:
        """
        格式化日期为YYYYMMDD
        
        Args:
            date: 日期字符串
            
        Returns:
            格式化后的日期
        """
        return date.replace('-', '').replace('/', '')

