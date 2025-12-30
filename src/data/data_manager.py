"""
数据管理器 - 统一的数据获取接口

整合 TushareFetcher 和 CacheManager，提供统一的数据访问接口。
支持：自动缓存、限流控制、数据补全等功能。
"""
import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime, timedelta

from src.utils.logger import log
from src.data.fetcher.tushare_fetcher import TushareFetcher
from src.data.storage.cache_manager import CacheManager
from config.settings import settings


class DataManager:
    """数据管理器 - 统一的数据获取接口"""
    
    def __init__(self, source: str = 'tushare', use_cache: bool = True):
        """
        初始化数据管理器
        
        Args:
            source: 数据源（目前仅支持 'tushare'）
            use_cache: 是否使用缓存
        """
        self.source = source
        self.use_cache = use_cache
        
        # 初始化数据获取器
        if source == 'tushare':
            self.fetcher = TushareFetcher()
        else:
            raise ValueError(f"不支持的数据源: {source}")
        
        # 初始化缓存管理器
        if use_cache:
            self.cache = CacheManager()
        else:
            self.cache = None
        
        log.info(f"DataManager 初始化成功 (source={source}, cache={use_cache})")
    
    def get_stock_list(self, list_status: str = 'L') -> pd.DataFrame:
        """
        获取股票列表
        
        Args:
            list_status: 上市状态 ('L'上市, 'D'退市, 'P'暂停上市)
            
        Returns:
            包含股票代码、名称等信息的DataFrame
        """
        cache_key_params = {'list_status': list_status}
        
        # 尝试从缓存获取
        if self.cache:
            cached = self.cache.get('stock_list', **cache_key_params)
            if cached is not None:
                log.info(f"从缓存获取股票列表: {len(cached)} 只")
                return cached
        
        # 从API获取
        df = self.fetcher.get_stock_list(list_status=list_status)
        
        # 存入缓存（股票列表缓存1天）
        if self.cache and not df.empty:
            self.cache.set('stock_list', df, expire_days=1, **cache_key_params)
        
        return df
    
    def get_daily_data(
        self, 
        stock_code: str, 
        start_date: str, 
        end_date: str, 
        adjust: str = 'qfq'
    ) -> pd.DataFrame:
        """
        获取日线数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            adjust: 复权类型 ('qfq'前复权, 'hfq'后复权, ''不复权)
            
        Returns:
            日线数据DataFrame
        """
        cache_key_params = {
            'ts_code': stock_code,
            'start_date': start_date,
            'end_date': end_date,
            'adjust': adjust
        }
        
        # 尝试从缓存获取
        if self.cache:
            cached = self.cache.get('daily', **cache_key_params)
            if cached is not None:
                return cached
        
        # 从API获取
        df = self.fetcher.get_daily_data(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )
        
        # 存入缓存
        if self.cache and not df.empty:
            self.cache.set('daily', df, **cache_key_params)
        
        return df
    
    def get_weekly_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        adjust: str = 'qfq'
    ) -> pd.DataFrame:
        """
        获取周线数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权类型
            
        Returns:
            周线数据DataFrame
        """
        cache_key_params = {
            'ts_code': stock_code,
            'start_date': start_date,
            'end_date': end_date,
            'adjust': adjust
        }
        
        # 尝试从缓存获取
        if self.cache:
            cached = self.cache.get('weekly', **cache_key_params)
            if cached is not None:
                return cached
        
        # 从API获取
        df = self.fetcher.get_weekly_data(
            ts_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )
        
        # 存入缓存
        if self.cache and not df.empty:
            self.cache.set('weekly', df, **cache_key_params)
        
        return df
    
    def get_daily_basic(
        self,
        stock_code: str = None,
        trade_date: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取每日指标（市值、市盈率、换手率等）
        
        Args:
            stock_code: 股票代码（可选）
            trade_date: 交易日期（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            
        Returns:
            每日指标DataFrame
        """
        cache_key_params = {
            'ts_code': stock_code,
            'trade_date': trade_date,
            'start_date': start_date,
            'end_date': end_date
        }
        
        # 尝试从缓存获取
        if self.cache:
            cached = self.cache.get('daily_basic', **cache_key_params)
            if cached is not None:
                return cached
        
        # 从API获取
        df = self.fetcher.get_daily_basic(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            trade_date=trade_date
        )
        
        # 存入缓存
        if self.cache and not df.empty:
            self.cache.set('daily_basic', df, **cache_key_params)
        
        return df
    
    def get_stk_factor(
        self,
        stock_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取技术因子（MACD、KDJ、RSI等）
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            技术因子DataFrame
        """
        cache_key_params = {
            'ts_code': stock_code,
            'start_date': start_date,
            'end_date': end_date
        }
        
        # 尝试从缓存获取
        if self.cache:
            cached = self.cache.get('stk_factor', **cache_key_params)
            if cached is not None:
                return cached
        
        # 从API获取
        df = self.fetcher.get_stk_factor(
            ts_code=stock_code,
            start_date=start_date,
            end_date=end_date
        )
        
        # 存入缓存
        if self.cache and not df.empty:
            self.cache.set('stk_factor', df, **cache_key_params)
        
        return df
    
    def get_complete_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取完整数据（行情 + 每日指标 + 技术因子）
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期（可选，默认今天）
            
        Returns:
            完整数据DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 1. 获取日线数据
        df_daily = self.get_daily_data(stock_code, start_date, end_date)
        
        if df_daily.empty:
            return pd.DataFrame()
        
        # 2. 获取每日指标
        try:
            df_basic = self.get_daily_basic(
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if not df_basic.empty:
                # 确保trade_date是同类型
                df_daily['trade_date'] = pd.to_datetime(df_daily['trade_date'])
                df_basic['trade_date'] = pd.to_datetime(df_basic['trade_date'])
                
                # 合并（保留日线数据的所有行）
                merge_cols = [c for c in df_basic.columns 
                             if c not in df_daily.columns or c == 'trade_date']
                df_daily = pd.merge(
                    df_daily,
                    df_basic[merge_cols],
                    on='trade_date',
                    how='left'
                )
        except Exception as e:
            log.warning(f"获取每日指标失败: {e}")
        
        # 3. 计算MA指标（如果没有）
        if 'ma5' not in df_daily.columns:
            df_daily['ma5'] = df_daily['close'].rolling(window=5).mean()
        if 'ma10' not in df_daily.columns:
            df_daily['ma10'] = df_daily['close'].rolling(window=10).mean()
        if 'ma20' not in df_daily.columns:
            df_daily['ma20'] = df_daily['close'].rolling(window=20).mean()
        
        return df_daily
    
    def get_suspend_info(
        self,
        ts_code: str = None,
        trade_date: str = None,
        suspend_type: str = None
    ) -> pd.DataFrame:
        """
        获取停牌信息
        
        Args:
            ts_code: 股票代码（可选）
            trade_date: 交易日期（可选）
            suspend_type: 停牌类型 ('S'停牌, 'R'复牌)
            
        Returns:
            停牌信息DataFrame
        """
        return self.fetcher.get_suspend_info(
            ts_code=ts_code,
            trade_date=trade_date,
            suspend_type=suspend_type
        )
    
    def get_trade_calendar(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取交易日历
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            交易日历DataFrame
        """
        cache_key_params = {
            'start_date': start_date,
            'end_date': end_date
        }
        
        # 尝试从缓存获取
        if self.cache:
            cached = self.cache.get('trade_calendar', **cache_key_params)
            if cached is not None:
                return cached
        
        # 从API获取
        df = self.fetcher.get_trade_calendar(start_date, end_date)
        
        # 存入缓存（交易日历缓存30天）
        if self.cache and not df.empty:
            self.cache.set('trade_calendar', df, expire_days=30, **cache_key_params)
        
        return df
    
    def batch_get_daily_data(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        adjust: str = 'qfq'
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取日线数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权类型
            
        Returns:
            {股票代码: DataFrame} 字典
        """
        result = {}
        total = len(stock_codes)
        
        for i, code in enumerate(stock_codes):
            if (i + 1) % 50 == 0:
                log.info(f"批量获取日线数据进度: {i+1}/{total}")
            
            df = self.get_daily_data(code, start_date, end_date, adjust)
            if not df.empty:
                result[code] = df
        
        log.info(f"批量获取完成: {len(result)}/{total} 只股票")
        return result
    
    def batch_get_daily_basic(
        self,
        trade_date: str,
        stock_codes: List[str] = None
    ) -> pd.DataFrame:
        """
        批量获取某日所有股票的每日指标
        
        Args:
            trade_date: 交易日期 (YYYYMMDD)
            stock_codes: 股票代码列表（可选）
            
        Returns:
            每日指标DataFrame
        """
        return self.fetcher.batch_get_daily_basic(trade_date, stock_codes)
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        if self.cache:
            return self.cache.get_stats()
        return {}
    
    def clear_cache(self, data_type: str = None):
        """
        清理缓存
        
        Args:
            data_type: 数据类型（如果为None则清除所有）
        """
        if self.cache:
            self.cache.invalidate(data_type)
