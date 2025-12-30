"""
Tushare Pro 数据获取器

封装 Tushare Pro API，提供统一的数据获取接口。
支持：股票列表、日线数据、周线数据、技术因子、每日指标等。
"""
import os
import time
import pandas as pd
from typing import Optional, List
from datetime import datetime, timedelta
from dotenv import load_dotenv

from src.utils.logger import log
from src.utils.rate_limiter import get_api_limiter, init_rate_limiter

# 加载环境变量
load_dotenv()


class TushareFetcher:
    """Tushare Pro 数据获取器"""
    
    def __init__(self, token: str = None, points: int = None):
        """
        初始化 Tushare 数据获取器
        
        Args:
            token: Tushare Pro Token，如果为None则从环境变量读取
            points: Tushare 积分，用于设置限流级别
        """
        self.token = token or os.getenv('TUSHARE_TOKEN')
        
        if not self.token or self.token == 'YOUR_TUSHARE_TOKEN':
            raise ValueError(
                "请设置有效的 TUSHARE_TOKEN！\n"
                "1. 在 https://tushare.pro/register 注册账号\n"
                "2. 在 .env 文件中设置 TUSHARE_TOKEN=你的token"
            )
        
        # 初始化 Tushare Pro
        import tushare as ts
        ts.set_token(self.token)
        self.pro = ts.pro_api()
        
        # 初始化限流器（根据积分设置）
        if points is None:
            points = int(os.getenv('TUSHARE_POINTS', '120'))
        init_rate_limiter(points)
        self.rate_limiter = get_api_limiter()
        
        log.info(f"TushareFetcher 初始化成功 (积分级别: {points})")
    
    def get_stock_list(
        self, 
        list_status: str = 'L',
        exchange: str = None
    ) -> pd.DataFrame:
        """
        获取股票列表
        
        Args:
            list_status: 上市状态 ('L'上市, 'D'退市, 'P'暂停上市)
            exchange: 交易所 ('SSE'上交所, 'SZSE'深交所, 'BSE'北交所)
            
        Returns:
            股票列表DataFrame，包含 ts_code, name, list_date 等字段
        """
        self.rate_limiter.wait_if_needed()
        
        try:
            df = self.pro.stock_basic(
                list_status=list_status,
                exchange=exchange,
                fields='ts_code,symbol,name,area,industry,list_date,market,is_hs'
            )
            log.info(f"获取股票列表成功: {len(df)} 只")
            return df
        except Exception as e:
            log.error(f"获取股票列表失败: {e}")
            raise
    
    def get_daily_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        adjust: str = 'qfq'
    ) -> pd.DataFrame:
        """
        获取日线数据
        
        Args:
            ts_code: 股票代码 (如 '000001.SZ')
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            adjust: 复权类型 ('qfq'前复权, 'hfq'后复权, ''不复权)
            
        Returns:
            日线数据DataFrame
        """
        self.rate_limiter.wait_if_needed()
        
        try:
            import tushare as ts
            
            # 使用 pro_bar 获取复权数据
            df = ts.pro_bar(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                adj=adjust,
                factors=['tor', 'vr']  # 换手率、量比
            )
            
            if df is not None and not df.empty:
                # 转换日期格式并排序
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').reset_index(drop=True)
                
            return df if df is not None else pd.DataFrame()
            
        except Exception as e:
            log.warning(f"获取日线数据失败 {ts_code}: {e}")
            return pd.DataFrame()
    
    def get_weekly_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        adjust: str = 'qfq'
    ) -> pd.DataFrame:
        """
        获取周线数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            adjust: 复权类型
            
        Returns:
            周线数据DataFrame
        """
        self.rate_limiter.wait_if_needed()
        
        try:
            df = self.pro.weekly(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                adj=adjust
            )
            
            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').reset_index(drop=True)
                
            return df if df is not None else pd.DataFrame()
            
        except Exception as e:
            log.warning(f"获取周线数据失败 {ts_code}: {e}")
            return pd.DataFrame()
    
    def get_daily_basic(
        self,
        ts_code: str = None,
        trade_date: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取每日指标（市值、市盈率、换手率等）
        
        Args:
            ts_code: 股票代码（可选）
            trade_date: 交易日期（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            
        Returns:
            每日指标DataFrame
        """
        self.rate_limiter.wait_if_needed()
        
        try:
            params = {
                'fields': 'ts_code,trade_date,close,turnover_rate,turnover_rate_f,'
                         'volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,'
                         'total_share,float_share,free_share,total_mv,circ_mv'
            }
            
            if ts_code:
                params['ts_code'] = ts_code
            if trade_date:
                params['trade_date'] = trade_date
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            
            df = self.pro.daily_basic(**params)
            
            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').reset_index(drop=True)
                
            return df if df is not None else pd.DataFrame()
            
        except Exception as e:
            log.warning(f"获取每日指标失败: {e}")
            return pd.DataFrame()
    
    def get_stk_factor(
        self,
        ts_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取技术因子（MACD、KDJ、RSI等）
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            技术因子DataFrame
        """
        self.rate_limiter.wait_if_needed()
        
        try:
            df = self.pro.stk_factor(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,close,macd_dif,macd_dea,macd,'
                       'kdj_k,kdj_d,kdj_j,rsi_6,rsi_12,rsi_24,boll_upper,'
                       'boll_mid,boll_lower,cci'
            )
            
            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date').reset_index(drop=True)
                
            return df if df is not None else pd.DataFrame()
            
        except Exception as e:
            log.warning(f"获取技术因子失败 {ts_code}: {e}")
            return pd.DataFrame()
    
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
        self.rate_limiter.wait_if_needed()
        
        try:
            params = {}
            if ts_code:
                params['ts_code'] = ts_code
            if trade_date:
                params['trade_date'] = trade_date
            if suspend_type:
                params['suspend_type'] = suspend_type
            
            df = self.pro.suspend_d(**params)
            return df if df is not None else pd.DataFrame()
            
        except Exception as e:
            log.warning(f"获取停牌信息失败: {e}")
            return pd.DataFrame()
    
    def get_trade_calendar(
        self,
        start_date: str,
        end_date: str,
        exchange: str = 'SSE'
    ) -> pd.DataFrame:
        """
        获取交易日历
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            exchange: 交易所（默认上交所）
            
        Returns:
            交易日历DataFrame
        """
        self.rate_limiter.wait_if_needed()
        
        try:
            df = self.pro.trade_cal(
                exchange=exchange,
                start_date=start_date,
                end_date=end_date,
                fields='exchange,cal_date,is_open,pretrade_date'
            )
            
            if df is not None and not df.empty:
                df['cal_date'] = pd.to_datetime(df['cal_date'])
                df = df.sort_values('cal_date').reset_index(drop=True)
                
            return df if df is not None else pd.DataFrame()
            
        except Exception as e:
            log.warning(f"获取交易日历失败: {e}")
            return pd.DataFrame()
    
    def batch_get_daily_basic(
        self,
        trade_date: str,
        stock_codes: List[str] = None
    ) -> pd.DataFrame:
        """
        批量获取某日所有股票的每日指标
        
        Args:
            trade_date: 交易日期 (YYYYMMDD)
            stock_codes: 股票代码列表（可选，为None则获取所有）
            
        Returns:
            每日指标DataFrame
        """
        df = self.get_daily_basic(trade_date=trade_date)
        
        if df.empty:
            return df
        
        if stock_codes:
            df = df[df['ts_code'].isin(stock_codes)]
        
        return df
