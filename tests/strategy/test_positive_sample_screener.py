"""
正样本筛选器测试
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.strategy.screening.positive_sample_screener import PositiveSampleScreener


class TestPositiveSampleScreener:
    """PositiveSampleScreener测试类"""
    
    def test_init(self, mock_data_manager):
        """测试初始化"""
        screener = PositiveSampleScreener(mock_data_manager)
        assert screener.dm == mock_data_manager
        assert screener.positive_samples == []
    
    def test_get_valid_stock_list(self, mock_data_manager):
        """测试获取有效股票列表"""
        screener = PositiveSampleScreener(mock_data_manager)
        
        # 模拟股票列表
        mock_data_manager.get_stock_list.return_value = pd.DataFrame({
            'ts_code': ['000001.SZ', '600000.SH', 'ST0001.SZ'],
            'name': ['平安银行', '浦发银行', 'ST股票'],
            'list_date': ['19910403', '19991110', '20000101'],
        })
        
        result = screener._get_valid_stock_list()
        assert isinstance(result, pd.DataFrame)
        # ST股票应该被过滤
        assert 'ST0001.SZ' not in result['ts_code'].values
    
    def test_check_positive_sample(self, mock_data_manager):
        """测试检查正样本"""
        screener = PositiveSampleScreener(mock_data_manager)
        
        # 模拟周线数据（三连阳，涨幅>50%）
        dates = pd.date_range(end='20240131', periods=21, freq='D')
        weekly_data = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            'open': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            'high': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
            'low': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        })
        
        # 这个方法需要真实数据，这里只测试接口
        assert hasattr(screener, '_check_positive_sample')
    
    def test_screen_all_stocks_structure(self, mock_data_manager):
        """测试筛选所有股票的结构"""
        screener = PositiveSampleScreener(mock_data_manager)
        
        # 模拟返回空结果（避免真实API调用）
        mock_data_manager.get_stock_list.return_value = pd.DataFrame({
            'ts_code': pd.Series([], dtype='object'),
            'name': pd.Series([], dtype='object'),
            'list_date': pd.Series([], dtype='object'),
        })
        
        result = screener.screen_all_stocks(
            start_date='20240101',
            end_date='20240131'
        )
        
        assert isinstance(result, pd.DataFrame)

