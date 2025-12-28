"""
财务筛选器测试
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.strategy.screening.financial_filter import FinancialFilter


class TestFinancialFilter:
    """FinancialFilter测试类"""
    
    def test_init(self, mock_data_manager):
        """测试初始化"""
        filter_obj = FinancialFilter(mock_data_manager)
        assert filter_obj.dm == mock_data_manager
        assert filter_obj.fetcher == mock_data_manager.fetcher
    
    def test_filter_stocks_basic(self, mock_data_manager, sample_stocks_df):
        """测试基本筛选功能"""
        filter_obj = FinancialFilter(mock_data_manager)
        
        # 模拟财务数据检查通过
        with patch.object(filter_obj, 'check_financial_indicators') as mock_check:
            mock_check.return_value = {
                'passed': True,
                'revenue': 5.0,
                'consecutive_profit_years': 3,
                'net_assets': 10.0
            }
            
            result = filter_obj.filter_stocks(
                sample_stocks_df,
                revenue_threshold=3.0,
                profit_years=3
            )
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
    
    def test_filter_stocks_with_failures(self, mock_data_manager, sample_stocks_df):
        """测试筛选失败的情况"""
        filter_obj = FinancialFilter(mock_data_manager)
        
        # 模拟财务数据检查失败
        with patch.object(filter_obj, 'check_financial_indicators') as mock_check:
            mock_check.return_value = {
                'passed': False,
                'reason': '营收不足'
            }
            
            result = filter_obj.filter_stocks(
                sample_stocks_df,
                revenue_threshold=3.0,
                profit_years=3
            )
            
            # 所有股票都应该被过滤掉
            assert len(result) == 0
    
    def test_check_financial_indicators(self, mock_data_manager):
        """测试财务指标检查"""
        filter_obj = FinancialFilter(mock_data_manager)
        
        # 模拟获取财务数据
        with patch.object(filter_obj.fetcher, 'get_financial_data') as mock_financial:
            mock_financial.return_value = pd.DataFrame({
                'end_date': ['20231231', '20221231', '20211231'],
                'revenue': [5.0, 4.5, 4.0],  # 亿元
                'net_profit': [0.5, 0.4, 0.3],
                'total_assets': [100.0, 90.0, 80.0],
                'total_liab': [50.0, 45.0, 40.0],
            })
            
            result = filter_obj.check_financial_indicators(
                '000001.SZ',
                revenue_threshold=3.0,
                profit_years=3
            )
            
            assert isinstance(result, dict)
            assert 'passed' in result
    
    def test_filter_stocks_column_names(self, mock_data_manager):
        """测试不同列名的处理"""
        filter_obj = FinancialFilter(mock_data_manager)
        
        # 测试使用'ts_code'列
        df1 = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'name': ['测试股票'],
            '牛股概率': [0.8]
        })
        
        # 测试使用'股票代码'列
        df2 = pd.DataFrame({
            '股票代码': ['000001.SZ'],
            '股票名称': ['测试股票'],
            '牛股概率': [0.8]
        })
        
        with patch.object(filter_obj, 'check_financial_indicators') as mock_check:
            mock_check.return_value = {
                'passed': True,
                'revenue': 5.0,
                'consecutive_profit_years': 3,
                'net_assets': 10.0
            }
            
            result1 = filter_obj.filter_stocks(df1)
            result2 = filter_obj.filter_stocks(df2)
            
            assert len(result1) > 0
            assert len(result2) > 0
    
    @pytest.mark.api
    def test_get_financial_data_real(self, mock_data_manager):
        """测试真实获取财务数据"""
        try:
            from config.data_source import data_source_config
            data_source_config.validate_tushare()
        except Exception as e:
            pytest.skip(f"Tushare配置无效: {e}")
        
        from src.data.data_manager import DataManager
        from src.strategy.screening.financial_filter import FinancialFilter
        
        dm = DataManager(source='tushare')
        filter_obj = FinancialFilter(dm)
        
        # 测试获取财务数据
        result = filter_obj.get_financial_data('000001.SZ')
        
        if result is not None:
            assert isinstance(result, dict)
            assert 'latest_revenue' in result
            assert 'consecutive_profit_years' in result
            assert 'net_assets' in result
    
    @pytest.mark.api
    def test_check_financial_indicators_real(self, mock_data_manager):
        """测试真实检查财务指标"""
        try:
            from config.data_source import data_source_config
            data_source_config.validate_tushare()
        except Exception as e:
            pytest.skip(f"Tushare配置无效: {e}")
        
        from src.data.data_manager import DataManager
        from src.strategy.screening.financial_filter import FinancialFilter
        
        dm = DataManager(source='tushare')
        filter_obj = FinancialFilter(dm)
        
        # 测试检查财务指标
        result = filter_obj.check_financial_indicators('000001.SZ')
        
        assert isinstance(result, dict)
        assert 'passed' in result
        assert 'reason' in result
    
    def test_check_financial_indicators_no_data(self, mock_data_manager):
        """测试无法获取财务数据的情况"""
        filter_obj = FinancialFilter(mock_data_manager)
        
        with patch.object(filter_obj, 'get_financial_data') as mock_get:
            mock_get.return_value = None
            
            result = filter_obj.check_financial_indicators('000001.SZ')
            
            assert result['passed'] == False
            assert '无法获取财务数据' in result['reason']
    
    def test_check_financial_indicators_low_revenue(self, mock_data_manager):
        """测试营收不足的情况"""
        filter_obj = FinancialFilter(mock_data_manager)
        
        with patch.object(filter_obj, 'get_financial_data') as mock_get:
            mock_get.return_value = {
                'latest_revenue': 2.0,  # 低于3亿
                'consecutive_profit_years': 3,
                'net_assets': 10.0,
            }
            
            result = filter_obj.check_financial_indicators('000001.SZ', revenue_threshold=3.0)
            
            assert result['passed'] == False
            assert '营收' in result['reason']
    
    def test_check_financial_indicators_insufficient_profit(self, mock_data_manager):
        """测试连续盈利年数不足的情况"""
        filter_obj = FinancialFilter(mock_data_manager)
        
        with patch.object(filter_obj, 'get_financial_data') as mock_get:
            mock_get.return_value = {
                'latest_revenue': 5.0,
                'consecutive_profit_years': 2,  # 少于3年
                'net_assets': 10.0,
            }
            
            result = filter_obj.check_financial_indicators('000001.SZ', profit_years=3)
            
            assert result['passed'] == False
            assert '连续盈利' in result['reason']
    
    def test_check_financial_indicators_negative_assets(self, mock_data_manager):
        """测试净资产为负的情况"""
        filter_obj = FinancialFilter(mock_data_manager)
        
        with patch.object(filter_obj, 'get_financial_data') as mock_get:
            mock_get.return_value = {
                'latest_revenue': 5.0,
                'consecutive_profit_years': 3,
                'net_assets': -1.0,  # 负资产
            }
            
            result = filter_obj.check_financial_indicators('000001.SZ')
            
            assert result['passed'] == False
            assert '净资产' in result['reason']
    
    def test_check_financial_indicators_exception(self, mock_data_manager):
        """测试异常处理"""
        filter_obj = FinancialFilter(mock_data_manager)
        
        with patch.object(filter_obj, 'get_financial_data') as mock_get:
            mock_get.side_effect = Exception("测试异常")
            
            result = filter_obj.check_financial_indicators('000001.SZ')
            
            assert result['passed'] == False
            assert '检查失败' in result['reason']

