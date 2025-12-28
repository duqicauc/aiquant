"""
财务指标筛选器
用于在模型预测后进行基本面筛选，过滤财务状况不佳的股票
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
from src.utils.logger import log
from src.utils.rate_limiter import safe_api_call


class FinancialFilter:
    """
    财务指标筛选器
    
    筛选条件（财务退市指标）：
    1. 营收 > 3亿
    2. 连续三年净利润 > 0
    3. 净资产 > 0
    """
    
    def __init__(self, data_manager):
        """
        初始化财务筛选器
        
        Args:
            data_manager: 数据管理器实例
        """
        self.dm = data_manager
        self.fetcher = data_manager.fetcher
    
    def filter_stocks(
        self,
        df_stocks: pd.DataFrame,
        revenue_threshold: float = 3.0,  # 营收阈值（亿元）
        profit_years: int = 3  # 连续盈利年数
    ) -> pd.DataFrame:
        """
        对股票列表进行财务筛选
        
        Args:
            df_stocks: 包含股票代码的DataFrame（必须有'股票代码'或'ts_code'列）
            revenue_threshold: 营收阈值（亿元）
            profit_years: 连续盈利年数
            
        Returns:
            通过筛选的股票DataFrame，增加筛选原因列
        """
        log.info("="*80)
        log.info("🔍 开始财务指标筛选")
        log.info("="*80)
        log.info(f"\n筛选条件：")
        log.info(f"  1. 营收 > {revenue_threshold}亿元")
        log.info(f"  2. 连续{profit_years}年净利润 > 0")
        log.info(f"  3. 净资产 > 0")
        log.info("")
        
        # 确定股票代码列名
        code_col = '股票代码' if '股票代码' in df_stocks.columns else 'ts_code'
        
        results = []
        total = len(df_stocks)
        
        for idx, row in df_stocks.iterrows():
            ts_code = row[code_col]
            name = row.get('股票名称', row.get('name', ''))
            
            if (idx + 1) % 10 == 0:
                log.info(f"进度: {idx+1}/{total} ({(idx+1)/total*100:.1f}%)")
            
            # 检查财务指标
            check_result = self.check_financial_indicators(
                ts_code,
                revenue_threshold=revenue_threshold,
                profit_years=profit_years
            )
            
            if check_result['passed']:
                # 通过筛选
                result_row = row.copy()
                result_row['财务状况'] = '良好'
                result_row['营收(亿)'] = check_result['revenue']
                result_row['连续盈利年数'] = check_result['consecutive_profit_years']
                result_row['净资产(亿)'] = check_result['net_assets']
                results.append(result_row)
                
                log.debug(f"  ✓ {name} 通过筛选")
            else:
                log.warning(f"  ✗ {name} 未通过: {check_result['reason']}")
        
        df_filtered = pd.DataFrame(results)
        
        log.info("\n" + "="*80)
        log.info("📊 筛选结果")
        log.info("="*80)
        log.info(f"原始数量: {total}")
        log.info(f"通过筛选: {len(df_filtered)}")
        log.info(f"剔除数量: {total - len(df_filtered)}")
        if total > 0:
            log.info(f"通过率: {len(df_filtered)/total*100:.1f}%")
        else:
            log.warning("⚠️  原始数量为0，无法计算通过率")
        
        return df_filtered
    
    def check_financial_indicators(
        self,
        ts_code: str,
        revenue_threshold: float = 3.0,
        profit_years: int = 3
    ) -> Dict:
        """
        检查单个股票的财务指标
        
        Args:
            ts_code: 股票代码
            revenue_threshold: 营收阈值（亿元）
            profit_years: 连续盈利年数
            
        Returns:
            检查结果字典
        """
        try:
            # 获取财务数据
            financial_data = self.get_financial_data(ts_code)
            
            if financial_data is None:
                return {
                    'passed': False,
                    'reason': '无法获取财务数据',
                    'revenue': None,
                    'consecutive_profit_years': None,
                    'net_assets': None
                }
            
            # 检查1: 营收 > 3亿
            latest_revenue = financial_data.get('latest_revenue', 0)
            if latest_revenue <= revenue_threshold:
                return {
                    'passed': False,
                    'reason': f'营收{latest_revenue:.2f}亿 <= {revenue_threshold}亿',
                    'revenue': latest_revenue,
                    'consecutive_profit_years': None,
                    'net_assets': None
                }
            
            # 检查2: 连续N年净利润 > 0
            consecutive_years = financial_data.get('consecutive_profit_years', 0)
            if consecutive_years < profit_years:
                return {
                    'passed': False,
                    'reason': f'连续盈利{consecutive_years}年 < {profit_years}年',
                    'revenue': latest_revenue,
                    'consecutive_profit_years': consecutive_years,
                    'net_assets': None
                }
            
            # 检查3: 净资产 > 0
            net_assets = financial_data.get('net_assets', 0)
            if net_assets <= 0:
                return {
                    'passed': False,
                    'reason': f'净资产{net_assets:.2f}亿 <= 0',
                    'revenue': latest_revenue,
                    'consecutive_profit_years': consecutive_years,
                    'net_assets': net_assets
                }
            
            # 全部通过
            return {
                'passed': True,
                'reason': 'OK',
                'revenue': latest_revenue,
                'consecutive_profit_years': consecutive_years,
                'net_assets': net_assets
            }
        
        except Exception as e:
            log.warning(f"检查{ts_code}财务指标失败: {e}")
            return {
                'passed': False,
                'reason': f'检查失败: {str(e)}',
                'revenue': None,
                'consecutive_profit_years': None,
                'net_assets': None
            }
    
    @safe_api_call(max_retries=3, base_delay=1.0)
    def get_financial_data(self, ts_code: str) -> Dict:
        """
        获取股票的财务数据（带限流）
        
        Args:
            ts_code: 股票代码
            
        Returns:
            财务数据字典
        """
        try:
            # 获取利润表（营收、净利润）
            income_df = self.fetcher.pro.income(
                ts_code=ts_code,
                fields='ts_code,end_date,revenue,n_income'
            )
            
            if income_df is None or income_df.empty:
                log.warning(f"{ts_code} 无利润表数据")
                return None
            
            # 按报告期排序（降序）
            income_df = income_df.sort_values('end_date', ascending=False)
            
            # 提取年报数据（报告期以1231结尾）
            annual_income = income_df[income_df['end_date'].str.endswith('1231')].head(5)
            
            if len(annual_income) < 3:
                log.warning(f"{ts_code} 年报数据不足3年")
                return None
            
            # 最新营收（亿元）
            latest_revenue = annual_income.iloc[0]['revenue'] / 1e8 if annual_income.iloc[0]['revenue'] else 0
            
            # 计算连续盈利年数
            consecutive_profit_years = 0
            for _, row in annual_income.iterrows():
                net_profit = row['n_income']
                if net_profit and net_profit > 0:
                    consecutive_profit_years += 1
                else:
                    break
            
            # 获取资产负债表（净资产）
            balance_df = self.fetcher.pro.balancesheet(
                ts_code=ts_code,
                fields='ts_code,end_date,total_assets,total_liab'
            )
            
            net_assets = 0
            if balance_df is not None and not balance_df.empty:
                balance_df = balance_df.sort_values('end_date', ascending=False)
                # 最新净资产 = 总资产 - 总负债
                latest_balance = balance_df.iloc[0]
                total_assets = latest_balance['total_assets'] if latest_balance['total_assets'] else 0
                total_liab = latest_balance['total_liab'] if latest_balance['total_liab'] else 0
                net_assets = (total_assets - total_liab) / 1e8  # 转换为亿元
            
            return {
                'latest_revenue': latest_revenue,
                'consecutive_profit_years': consecutive_profit_years,
                'net_assets': net_assets,
            }
        
        except Exception as e:
            log.error(f"获取{ts_code}财务数据失败: {e}")
            return None


if __name__ == '__main__':
    # 测试
    from src.data.data_manager import DataManager
    
    log.info("测试财务筛选器")
    
    dm = DataManager()
    filter_obj = FinancialFilter(dm)
    
    # 测试单个股票
    test_codes = ['000001.SZ', '600000.SH', '000002.SZ']
    
    for code in test_codes:
        log.info(f"\n测试: {code}")
        result = filter_obj.check_financial_indicators(code)
        log.info(f"结果: {result}")

