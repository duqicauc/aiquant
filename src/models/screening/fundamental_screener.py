"""
基本面筛选器 - 筛选被错杀的优质公司

筛选条件（v3版本：国九+红利+审计筛选）：
1. 市值筛选：market_cap在10-100亿之间
2. 财务筛选：营业收入>1e8, 净利润>2000000
3. 盈利能力筛选：ROE>0, ROA>0
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime, timedelta
from src.utils.logger import log


class FundamentalScreener:
    """基本面筛选器"""
    
    def __init__(self, data_manager, config: Optional[Dict] = None):
        """
        初始化基本面筛选器
        
        Args:
            data_manager: 数据管理器实例
            config: 配置字典，包含筛选条件
        """
        self.dm = data_manager
        
        # 默认配置（v3版本：国九+红利+审计筛选）
        # 推荐使用标准方案（方案2），详见 docs/reference/FUNDAMENTAL_SCREENING_THRESHOLDS.md
        self.config = {
            # 市值筛选（单位：万元）
            'market_cap_min': 100000,      # 10亿 = 100000万元
            'market_cap_max': 1000000,     # 100亿 = 1000000万元
            
            # 财务筛选（单位：元）
            # 标准方案（推荐）：
            # - 营业收入>5亿：确保公司有足够的业务规模，市销率在2-20倍之间
            # - 净利润>500万：最低盈利能力要求
            # 保守方案：revenue_min=3e8, net_profit_min=3000000
            # 严格方案：revenue_min=10e8, net_profit_min=10000000
            'revenue_min': 5e8,            # 营业收入>5亿（单位：元）- 标准方案
            'net_profit_min': 5000000,    # 净利润>500万（单位：元）- 标准方案
            
            # 盈利能力筛选（单位：百分比%）
            # 标准方案（推荐）：
            # - ROE>5%：高于A股平均水平，但不过于严格
            # - ROA>2%：接近A股平均水平，确保资产使用效率
            # 保守方案：roe_min=2, roa_min=1
            # 严格方案：roe_min=8, roa_min=3
            'roe_min': 5,                  # ROE>5%（标准方案）
            'roa_min': 2,                  # ROA>2%（标准方案）
            
            # 是否启用筛选
            'enabled': False,
        }
        
        # 合并用户配置
        if config:
            self.config.update(config)
    
    def get_latest_financial_data(self, ts_code: str, trade_date: str) -> Optional[Dict]:
        """
        获取最新的财务数据
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期（YYYYMMDD）
            
        Returns:
            财务数据字典，包含：market_cap, revenue, net_profit, roe, roa
            如果获取失败返回None
        """
        try:
            # 1. 获取市值（从daily_basic）
            df_basic = self.dm.get_daily_basic(
                stock_code=ts_code,
                trade_date=trade_date
            )
            
            if df_basic.empty:
                # 尝试获取最近的数据
                end_date = trade_date
                start_date = (datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=30)).strftime('%Y%m%d')
                df_basic = self.dm.get_daily_basic(
                    stock_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
            
            if df_basic.empty:
                return None
            
            # 获取最新的市值数据
            latest_basic = df_basic.iloc[-1]
            market_cap = latest_basic.get('total_mv', 0)  # 总市值（万元）
            
            # 2. 获取财务数据（从fina_indicator和income）
            # 注意：财务数据按报告期（end_date）发布，不是按交易日期
            # 获取最近一年的财务数据，然后选择最新的报告期
            trade_dt = datetime.strptime(trade_date, '%Y%m%d')
            
            # 获取最近一年的财务数据（按报告期end_date）
            # 财务报告通常有延迟，所以获取最近一年确保能拿到最新数据
            end_date = trade_date
            start_date = (trade_dt - timedelta(days=365)).strftime('%Y%m%d')
            
            # 获取财务指标（包含ROE、ROA，单位：百分比%）
            fina_data = self._get_fina_indicator(ts_code, start_date, end_date)
            
            # 获取利润表（包含营业收入、净利润，单位：元）
            income_data = self._get_income(ts_code, start_date, end_date)
            
            # 合并数据
            result = {
                'market_cap': market_cap,
                'revenue': income_data.get('revenue', 0) if income_data else 0,
                'net_profit': income_data.get('n_income', 0) if income_data else 0,
                'roe': fina_data.get('roe', 0) if fina_data else 0,
                'roa': fina_data.get('roa', 0) if fina_data else 0,
            }
            
            return result
            
        except Exception as e:
            log.debug(f"获取财务数据失败 {ts_code}: {e}")
            return None
    
    def _get_fina_indicator(self, ts_code: str, start_date: str, end_date: str) -> Optional[Dict]:
        """
        获取财务指标（ROE、ROA）
        
        注意：财务数据按报告期发布，不是按交易日期。这里获取最近一年的数据，
        然后选择最新的报告期数据。
        """
        try:
            # 使用data_manager的fetcher来获取数据
            fetcher = self.dm.fetcher
            fetcher.rate_limiter.wait_if_needed()
            
            # 获取财务指标（按报告期end_date获取最近一年的数据）
            df = fetcher.pro.fina_indicator(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,end_date,roe,roa'
            )
            
            if df is not None and not df.empty:
                # 获取最新的报告期数据（按end_date排序）
                df = df.sort_values('end_date', ascending=False)
                latest = df.iloc[0]
                
                # 检查数据有效性（ROE和ROA应该是百分比，正常范围0-100）
                roe = latest.get('roe', 0)
                roa = latest.get('roa', 0)
                
                # 如果数据异常（可能是空值或异常值），返回None
                if pd.isna(roe) or pd.isna(roa) or not np.isfinite(roe) or not np.isfinite(roa):
                    return None
                
                return {
                    'roe': float(roe),
                    'roa': float(roa),
                }
            
            return None
        except Exception as e:
            log.debug(f"获取财务指标失败 {ts_code}: {e}")
            return None
    
    def _get_income(self, ts_code: str, start_date: str, end_date: str) -> Optional[Dict]:
        """
        获取利润表（营业收入、净利润）
        
        注意：
        1. revenue和n_income的单位是"元"（不是千元或万元）
        2. 财务数据按报告期发布，这里获取最新报告期的数据
        """
        try:
            # 使用data_manager的fetcher来获取数据
            fetcher = self.dm.fetcher
            fetcher.rate_limiter.wait_if_needed()
            
            # 获取利润表（按报告期end_date获取最近一年的数据）
            df = fetcher.pro.income(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,end_date,revenue,n_income'
            )
            
            if df is not None and not df.empty:
                # 获取最新的报告期数据（按end_date排序）
                df = df.sort_values('end_date', ascending=False)
                latest = df.iloc[0]
                
                # 检查数据有效性
                revenue = latest.get('revenue', 0)
                n_income = latest.get('n_income', 0)
                
                # 如果数据异常（可能是空值或异常值），返回None
                if pd.isna(revenue) or pd.isna(n_income) or not np.isfinite(revenue) or not np.isfinite(n_income):
                    return None
                
                return {
                    'revenue': float(revenue),  # 单位：元
                    'n_income': float(n_income),  # 单位：元
                }
            
            return None
        except Exception as e:
            log.debug(f"获取利润表失败 {ts_code}: {e}")
            return None
    
    def screen_stocks(self, stocks_df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        """
        对股票进行基本面筛选
        
        Args:
            stocks_df: 待筛选的股票DataFrame，必须包含ts_code列
            trade_date: 交易日期（YYYYMMDD）
            
        Returns:
            筛选后的股票DataFrame，添加了fundamental_pass列表示是否通过筛选
        """
        if not self.config.get('enabled', False):
            log.info("基本面筛选未启用，跳过筛选")
            stocks_df['fundamental_pass'] = True
            return stocks_df
        
        log.info("="*80)
        log.info("基本面筛选器 - 筛选被错杀的优质公司")
        log.info("="*80)
        log.info(f"筛选条件:")
        log.info(f"  市值范围: {self.config['market_cap_min']/10000:.0f}亿 - {self.config['market_cap_max']/10000:.0f}亿")
        log.info(f"  营业收入 > {self.config['revenue_min']/1e8:.2f}亿")
        log.info(f"  净利润 > {self.config['net_profit_min']/1e6:.2f}百万")
        log.info(f"  ROE > {self.config['roe_min']:.2f}%")
        log.info(f"  ROA > {self.config['roa_min']:.2f}%")
        log.info("")
        
        result_df = stocks_df.copy()
        result_df['fundamental_pass'] = False
        result_df['fundamental_reason'] = ''
        
        total = len(result_df)
        passed_count = 0
        failed_count = 0
        
        for idx, row in result_df.iterrows():
            ts_code = row['ts_code']
            
            # 获取财务数据
            fina_data = self.get_latest_financial_data(ts_code, trade_date)
            
            if fina_data is None:
                result_df.at[idx, 'fundamental_reason'] = '财务数据缺失'
                failed_count += 1
                continue
            
            # 检查筛选条件
            reasons = []
            
            # 1. 市值筛选
            market_cap = fina_data.get('market_cap', 0)
            if market_cap < self.config['market_cap_min']:
                reasons.append(f"市值过小({market_cap/10000:.2f}亿)")
            elif market_cap > self.config['market_cap_max']:
                reasons.append(f"市值过大({market_cap/10000:.2f}亿)")
            
            # 2. 财务筛选
            revenue = fina_data.get('revenue', 0)
            if revenue < self.config['revenue_min']:
                reasons.append(f"营收不足({revenue/1e8:.2f}亿)")
            
            net_profit = fina_data.get('net_profit', 0)
            if net_profit < self.config['net_profit_min']:
                reasons.append(f"净利润不足({net_profit/1e6:.2f}百万)")
            
            # 3. 盈利能力筛选
            roe = fina_data.get('roe', 0)
            if roe <= self.config['roe_min']:
                reasons.append(f"ROE不足({roe:.2f}%)")
            
            roa = fina_data.get('roa', 0)
            if roa <= self.config['roa_min']:
                reasons.append(f"ROA不足({roa:.2f}%)")
            
            # 判断是否通过
            if not reasons:
                result_df.at[idx, 'fundamental_pass'] = True
                passed_count += 1
            else:
                result_df.at[idx, 'fundamental_reason'] = '; '.join(reasons)
                failed_count += 1
            
            # 进度提示
            if (idx + 1) % 100 == 0:
                log.info(f"进度: {idx+1}/{total} | 通过: {passed_count}, 未通过: {failed_count}")
        
        log.info("")
        log.info(f"基本面筛选完成:")
        log.info(f"  总股票数: {total}")
        log.info(f"  通过筛选: {passed_count} ({passed_count/total*100:.1f}%)")
        log.info(f"  未通过筛选: {failed_count} ({failed_count/total*100:.1f}%)")
        
        return result_df
    
    def filter_stocks(self, stocks_df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        """
        过滤股票，只保留通过基本面筛选的股票
        
        Args:
            stocks_df: 待筛选的股票DataFrame
            trade_date: 交易日期
            
        Returns:
            过滤后的股票DataFrame
        """
        screened_df = self.screen_stocks(stocks_df, trade_date)
        
        if not self.config.get('enabled', False):
            return screened_df
        
        filtered_df = screened_df[screened_df['fundamental_pass'] == True].copy()
        log.info(f"基本面筛选后剩余: {len(filtered_df)} 只股票（原始: {len(stocks_df)}）")
        
        return filtered_df
