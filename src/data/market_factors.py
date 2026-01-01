"""
市场因子模块 - 获取和计算市场整体趋势和相对强度特征

提供以下因子：
1. 市场整体趋势
   - 大盘指数N日涨跌幅
   - 市场情绪指标（涨跌停统计、融资融券等）
   
2. 相对强度特征
   - 个股相对大盘强度
   - 个股相对历史均值强度
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime, timedelta

from src.utils.logger import log


class MarketFactors:
    """市场因子计算器"""
    
    # 主要指数代码
    INDEX_CODES = {
        'sh': '000001.SH',      # 上证指数
        'sz': '399001.SZ',      # 深证成指
        'cyb': '399006.SZ',     # 创业板指
        'hs300': '000300.SH',   # 沪深300
        'zz500': '000905.SH',   # 中证500
    }
    
    def __init__(self, fetcher=None):
        """
        初始化市场因子计算器
        
        Args:
            fetcher: 数据获取器（TushareFetcher实例）
        """
        self.fetcher = fetcher
        self._index_cache = {}  # 缓存指数数据
    
    def get_index_data(
        self,
        index_code: str = '000001.SH',
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取指数日线数据
        
        Args:
            index_code: 指数代码（如 '000001.SH'）
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            
        Returns:
            指数日线数据DataFrame
        """
        if self.fetcher is None:
            raise ValueError("需要提供 fetcher 实例")
        
        # 检查缓存
        cache_key = f"{index_code}_{start_date}_{end_date}"
        if cache_key in self._index_cache:
            return self._index_cache[cache_key]
        
        try:
            self.fetcher.rate_limiter.wait_if_needed()
            
            df = self.fetcher.pro.index_daily(
                ts_code=index_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount'
            )
            
            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').reset_index(drop=True)
                
                # 缓存结果
                self._index_cache[cache_key] = df
                
            return df if df is not None else pd.DataFrame()
            
        except Exception as e:
            log.warning(f"获取指数数据失败 {index_code}: {e}")
            return pd.DataFrame()
    
    def calculate_market_return(
        self,
        index_data: pd.DataFrame,
        window: int = 34
    ) -> pd.DataFrame:
        """
        计算市场N日涨跌幅
        
        Args:
            index_data: 指数日线数据
            window: 计算窗口（天数）
            
        Returns:
            添加了市场涨跌幅的DataFrame
        """
        if index_data.empty:
            return index_data
        
        df = index_data.copy()
        
        # 计算N日涨跌幅
        df[f'market_return_{window}d'] = (
            df['close'] / df['close'].shift(window) - 1
        ) * 100
        
        # 计算N日波动率
        df[f'market_volatility_{window}d'] = (
            df['pct_chg'].rolling(window=window).std()
        )
        
        # 计算N日均线
        df[f'market_ma_{window}d'] = df['close'].rolling(window=window).mean()
        
        # 市场趋势（价格相对均线位置）
        df['market_trend'] = (df['close'] / df[f'market_ma_{window}d'] - 1) * 100
        
        return df
    
    def calculate_relative_strength(
        self,
        stock_data: pd.DataFrame,
        market_data: pd.DataFrame,
        window: int = 34
    ) -> pd.DataFrame:
        """
        计算个股相对大盘的强度
        
        Args:
            stock_data: 个股日线数据（需包含 trade_date, close, pct_chg）
            market_data: 市场指数数据（需包含 trade_date, close, pct_chg）
            window: 计算窗口
            
        Returns:
            添加了相对强度特征的DataFrame
        """
        if stock_data.empty or market_data.empty:
            return stock_data
        
        df = stock_data.copy()
        
        # 确保日期格式一致
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        market_data = market_data.copy()
        market_data['trade_date'] = pd.to_datetime(market_data['trade_date'])
        
        # 合并市场数据
        market_cols = ['trade_date', 'close', 'pct_chg']
        market_subset = market_data[market_cols].copy()
        market_subset.columns = ['trade_date', 'market_close', 'market_pct_chg']
        
        df = pd.merge(df, market_subset, on='trade_date', how='left')
        
        # 1. 单日相对强度（超额收益）
        df['excess_return'] = df['pct_chg'] - df['market_pct_chg']
        
        # 2. N日累计超额收益
        df['stock_return_nd'] = (
            df['close'] / df['close'].shift(window) - 1
        ) * 100
        df['market_return_nd'] = (
            df['market_close'] / df['market_close'].shift(window) - 1
        ) * 100
        df[f'excess_return_{window}d'] = df['stock_return_nd'] - df['market_return_nd']
        
        # 3. 相对强度指数（RS）
        # RS = 个股N日涨幅 / 市场N日涨幅（当市场涨幅为正时）
        df['relative_strength'] = np.where(
            df['market_return_nd'] > 0,
            df['stock_return_nd'] / df['market_return_nd'],
            np.where(
                df['market_return_nd'] < 0,
                -df['stock_return_nd'] / df['market_return_nd'],
                0
            )
        )
        
        # 4. Beta（个股对市场的敏感度）
        # 使用滚动窗口计算
        def calc_beta(group):
            if len(group) < window:
                return np.nan
            stock_returns = group['pct_chg'].values
            market_returns = group['market_pct_chg'].values
            
            # 去除NaN
            mask = ~(np.isnan(stock_returns) | np.isnan(market_returns))
            if mask.sum() < 10:
                return np.nan
            
            stock_returns = stock_returns[mask]
            market_returns = market_returns[mask]
            
            cov = np.cov(stock_returns, market_returns)[0, 1]
            var = np.var(market_returns)
            return cov / var if var > 0 else np.nan
        
        # 简化的Beta计算（使用协方差/方差）
        df['beta'] = (
            df['pct_chg'].rolling(window=window).cov(df['market_pct_chg']) /
            df['market_pct_chg'].rolling(window=window).var()
        )
        
        # 清理临时列
        df = df.drop(columns=['stock_return_nd', 'market_return_nd'], errors='ignore')
        
        return df
    
    def calculate_historical_strength(
        self,
        stock_data: pd.DataFrame,
        lookback_window: int = 250,  # 约1年
        compare_window: int = 34
    ) -> pd.DataFrame:
        """
        计算个股相对历史均值的强度
        
        Args:
            stock_data: 个股日线数据
            lookback_window: 历史回看窗口（天数）
            compare_window: 比较窗口（天数）
            
        Returns:
            添加了历史强度特征的DataFrame
        """
        if stock_data.empty:
            return stock_data
        
        df = stock_data.copy()
        
        # 1. 价格相对历史均值位置
        df['price_vs_hist_mean'] = (
            df['close'] / df['close'].rolling(window=lookback_window).mean() - 1
        ) * 100
        
        # 2. 价格相对历史高点位置
        df['price_vs_hist_high'] = (
            df['close'] / df['close'].rolling(window=lookback_window).max() - 1
        ) * 100
        
        # 3. 价格相对历史低点位置
        df['price_vs_hist_low'] = (
            df['close'] / df['close'].rolling(window=lookback_window).min() - 1
        ) * 100
        
        # 4. 成交量相对历史均值
        if 'vol' in df.columns:
            df['volume_vs_hist_mean'] = (
                df['vol'] / df['vol'].rolling(window=lookback_window).mean()
            )
        
        # 5. 波动率相对历史均值
        current_vol = df['pct_chg'].rolling(window=compare_window).std()
        hist_vol = df['pct_chg'].rolling(window=lookback_window).std()
        df['volatility_vs_hist'] = current_vol / hist_vol
        
        # 6. 当前涨跌幅在历史分布中的位置（百分位）
        def rolling_percentile(series, window):
            result = []
            for i in range(len(series)):
                if i < window - 1:
                    result.append(np.nan)
                else:
                    window_data = series.iloc[i-window+1:i+1]
                    current = series.iloc[i]
                    pct = (window_data < current).sum() / len(window_data) * 100
                    result.append(pct)
            return result
        
        # 计算N日涨幅
        df['return_nd'] = (
            df['close'] / df['close'].shift(compare_window) - 1
        ) * 100
        
        # 当前N日涨幅在历史中的百分位
        df['return_percentile'] = rolling_percentile(
            df['return_nd'], lookback_window
        )
        
        return df
    
    def get_market_sentiment(
        self,
        trade_date: str
    ) -> Dict:
        """
        获取市场情绪指标
        
        Args:
            trade_date: 交易日期 (YYYYMMDD)
            
        Returns:
            市场情绪指标字典
        """
        if self.fetcher is None:
            return {}
        
        sentiment = {}
        
        try:
            self.fetcher.rate_limiter.wait_if_needed()
            
            # 1. 涨跌停统计
            df_limit = self.fetcher.pro.limit_list_d(trade_date=trade_date)
            if df_limit is not None and not df_limit.empty:
                # 涨停数量
                up_limit = len(df_limit[df_limit['limit'] == 'U'])
                # 跌停数量
                down_limit = len(df_limit[df_limit['limit'] == 'D'])
                # 炸板数量（开板后未封住）
                broken_limit = len(df_limit[df_limit['limit'] == 'Z'])
                
                sentiment['up_limit_count'] = up_limit
                sentiment['down_limit_count'] = down_limit
                sentiment['broken_limit_count'] = broken_limit
                sentiment['limit_ratio'] = (
                    up_limit / (up_limit + down_limit) 
                    if (up_limit + down_limit) > 0 else 0.5
                )
            
        except Exception as e:
            log.warning(f"获取涨跌停数据失败: {e}")
        
        try:
            self.fetcher.rate_limiter.wait_if_needed()
            
            # 2. 融资融券数据（市场整体）
            df_margin = self.fetcher.pro.margin(trade_date=trade_date)
            if df_margin is not None and not df_margin.empty:
                # 融资余额（亿元）
                total_rzye = df_margin['rzye'].sum() / 1e8
                # 融券余额（亿元）
                total_rqye = df_margin['rqyl'].sum() / 1e8 if 'rqyl' in df_margin.columns else 0
                
                sentiment['margin_balance'] = total_rzye
                sentiment['short_balance'] = total_rqye
                sentiment['margin_ratio'] = (
                    total_rzye / (total_rzye + total_rqye) 
                    if (total_rzye + total_rqye) > 0 else 0.5
                )
                
        except Exception as e:
            log.warning(f"获取融资融券数据失败: {e}")
        
        return sentiment
    
    def enrich_stock_data_with_market_factors(
        self,
        stock_data: pd.DataFrame,
        index_code: str = '000001.SH',
        window: int = 34
    ) -> pd.DataFrame:
        """
        为个股数据添加市场因子
        
        Args:
            stock_data: 个股日线数据
            index_code: 参考指数代码
            window: 计算窗口
            
        Returns:
            添加了市场因子的DataFrame
        """
        if stock_data.empty:
            return stock_data
        
        # 获取日期范围
        stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
        start_date = stock_data['trade_date'].min()
        end_date = stock_data['trade_date'].max()
        
        # 往前多取一些数据用于计算
        extended_start = start_date - timedelta(days=window * 2)
        
        # 获取市场数据
        market_data = self.get_index_data(
            index_code=index_code,
            start_date=extended_start.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d')
        )
        
        if market_data.empty:
            log.warning("无法获取市场数据，跳过市场因子计算")
            return stock_data
        
        # 计算市场趋势
        market_data = self.calculate_market_return(market_data, window)
        
        # 计算相对强度
        df = self.calculate_relative_strength(stock_data, market_data, window)
        
        # 计算历史强度
        df = self.calculate_historical_strength(df, lookback_window=250, compare_window=window)
        
        # 合并市场趋势数据
        market_cols = [
            'trade_date', 
            f'market_return_{window}d',
            f'market_volatility_{window}d',
            'market_trend'
        ]
        market_subset = market_data[
            [c for c in market_cols if c in market_data.columns]
        ].copy()
        
        df = pd.merge(df, market_subset, on='trade_date', how='left')
        
        return df


def calculate_market_factors_for_samples(
    samples_df: pd.DataFrame,
    fetcher,
    window: int = 34
) -> pd.DataFrame:
    """
    为样本数据批量计算市场因子
    
    Args:
        samples_df: 样本数据（需包含 ts_code, trade_date）
        fetcher: 数据获取器
        window: 计算窗口
        
    Returns:
        添加了市场因子的DataFrame
    """
    mf = MarketFactors(fetcher)
    
    # 获取日期范围
    samples_df['trade_date'] = pd.to_datetime(samples_df['trade_date'])
    start_date = samples_df['trade_date'].min()
    end_date = samples_df['trade_date'].max()
    
    # 往前多取数据
    extended_start = start_date - timedelta(days=window * 2 + 50)
    
    # 获取市场数据
    log.info(f"获取市场指数数据: {extended_start.strftime('%Y%m%d')} - {end_date.strftime('%Y%m%d')}")
    market_data = mf.get_index_data(
        index_code='000001.SH',
        start_date=extended_start.strftime('%Y%m%d'),
        end_date=end_date.strftime('%Y%m%d')
    )
    
    if market_data.empty:
        log.warning("无法获取市场数据")
        return samples_df
    
    # 计算市场趋势
    market_data = mf.calculate_market_return(market_data, window)
    
    # 准备合并的市场数据
    market_cols = [
        'trade_date',
        'pct_chg',
        f'market_return_{window}d',
        f'market_volatility_{window}d',
        'market_trend'
    ]
    market_subset = market_data[
        [c for c in market_cols if c in market_data.columns]
    ].copy()
    market_subset.columns = [
        'trade_date' if c == 'trade_date' else f'market_{c}' if c == 'pct_chg' else c
        for c in market_subset.columns
    ]
    
    # 合并市场数据到样本
    result = pd.merge(samples_df, market_subset, on='trade_date', how='left')
    
    # 计算超额收益
    if 'pct_chg' in result.columns and 'market_pct_chg' in result.columns:
        result['excess_return'] = result['pct_chg'] - result['market_pct_chg']
    
    log.success(f"✓ 市场因子计算完成，新增 {len(market_subset.columns) - 1} 个特征")
    
    return result

