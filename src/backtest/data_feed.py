"""
Backtrader数据适配器
将Tushare数据转换为Backtrader格式
"""

import backtrader as bt
import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager
from src.utils.logger import log


class TushareData(bt.feeds.PandasData):
    """
    Tushare数据适配器
    
    将Tushare的数据格式转换为Backtrader可用的格式
    """
    
    params = (
        ('datetime', None),    # 日期列（索引）
        ('open', 'open'),      # 开盘价
        ('high', 'high'),      # 最高价
        ('low', 'low'),        # 最低价
        ('close', 'close'),    # 收盘价
        ('volume', 'vol'),     # 成交量
        ('openinterest', -1),  # 持仓量（股票不需要，设为-1）
    )


class DataFeedManager:
    """数据Feed管理器"""
    
    def __init__(self):
        self.dm = DataManager()
    
    def get_data_feed(self, stock_code: str, start_date: str, end_date: str) -> TushareData:
        """
        获取单只股票的数据Feed
        
        Args:
            stock_code: 股票代码，如 '000001.SZ'
            start_date: 开始日期，如 '20200101'
            end_date: 结束日期，如 '20241224'
        
        Returns:
            TushareData: Backtrader数据Feed
        """
        log.info(f"加载数据: {stock_code} ({start_date} - {end_date})")
        
        # 获取日线数据
        df = self.dm.get_daily_data(stock_code, start_date, end_date)
        
        if df.empty:
            log.warning(f"股票 {stock_code} 无数据")
            return None
        
        # 数据预处理
        df = self._prepare_data(df, stock_code)
        
        # 创建Feed
        data_feed = TushareData(
            dataname=df,
            name=stock_code
        )
        
        log.info(f"✓ 数据加载完成: {len(df)} 条记录")
        return data_feed
    
    def get_multiple_feeds(self, stock_codes: list, start_date: str, end_date: str) -> dict:
        """
        获取多只股票的数据Feed
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            dict: {stock_code: data_feed}
        """
        log.info(f"加载多股票数据: {len(stock_codes)} 只股票")
        
        feeds = {}
        for stock_code in stock_codes:
            feed = self.get_data_feed(stock_code, start_date, end_date)
            if feed is not None:
                feeds[stock_code] = feed
        
        log.info(f"✓ 成功加载 {len(feeds)}/{len(stock_codes)} 只股票")
        return feeds
    
    def _prepare_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            df: 原始数据
            stock_code: 股票代码
        
        Returns:
            处理后的数据
        """
        # 确保有必要的列
        required_cols = ['trade_date', 'open', 'high', 'low', 'close', 'vol']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必要列: {col}")
        
        # 转换日期格式
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        
        # 设置日期为索引
        df.set_index('trade_date', inplace=True)
        
        # 按日期排序
        df.sort_index(inplace=True)
        
        # 删除重复日期
        df = df[~df.index.duplicated(keep='first')]
        
        # 处理缺失值
        df.fillna(method='ffill', inplace=True)
        
        # 确保数值类型
        for col in ['open', 'high', 'low', 'close', 'vol']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 删除异常数据（价格或成交量为0）
        df = df[(df['close'] > 0) & (df['vol'] > 0)]
        
        return df


def create_data_feed(stock_code: str, start_date: str, end_date: str) -> TushareData:
    """
    快速创建数据Feed的便捷函数
    
    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        TushareData: 数据Feed
    
    Example:
        >>> feed = create_data_feed('000001.SZ', '20200101', '20241224')
        >>> cerebro.adddata(feed)
    """
    manager = DataFeedManager()
    return manager.get_data_feed(stock_code, start_date, end_date)


if __name__ == '__main__':
    # 测试
    feed = create_data_feed('000001.SZ', '20230101', '20241224')
    print(f"数据Feed创建成功: {feed._name}")

