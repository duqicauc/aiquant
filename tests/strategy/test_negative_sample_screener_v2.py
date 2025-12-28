"""
负样本筛选器V2测试
"""
import pytest
import pandas as pd
from unittest.mock import Mock
from src.strategy.screening.negative_sample_screener_v2 import NegativeSampleScreenerV2


class TestNegativeSampleScreenerV2:
    """NegativeSampleScreenerV2测试类"""
    
    def test_init(self, mock_data_manager):
        """测试初始化"""
        screener = NegativeSampleScreenerV2(mock_data_manager)
        assert screener.dm == mock_data_manager
    
    def test_screen_negative_samples_structure(self, mock_data_manager):
        """测试筛选负样本的结构"""
        screener = NegativeSampleScreenerV2(mock_data_manager)
        
        # 创建模拟正样本
        positive_samples = pd.DataFrame({
            'ts_code': ['000001.SZ', '600000.SH'],
            't1_date': ['20240115', '20240116'],
            't0_date': ['20240101', '20240102'],
        })
        
        # 模拟股票列表
        mock_data_manager.get_stock_list.return_value = pd.DataFrame({
            'ts_code': ['000002.SZ', '600001.SH', '000001.SZ'],
            'name': ['万科A', '邯郸钢铁', '平安银行'],
            'list_date': ['19910129', '19940114', '19910403'],
        })
        
        # 模拟日线数据
        dates = pd.date_range(end='20240115', periods=34, freq='D')
        mock_data_manager.get_daily_data.return_value = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'close': range(10, 44),
        })
        
        result = screener.screen_negative_samples(
            positive_samples_df=positive_samples,
            samples_per_positive=1
        )
        
        assert isinstance(result, pd.DataFrame)
    
    def test_get_valid_stock_list(self, mock_data_manager):
        """测试获取有效股票列表"""
        screener = NegativeSampleScreenerV2(mock_data_manager)
        
        mock_data_manager.get_stock_list.return_value = pd.DataFrame({
            'ts_code': ['000001.SZ', 'ST0001.SZ'],
            'name': ['平安银行', 'ST股票'],
            'list_date': ['19910403', '20000101'],
        })
        
        result = screener._get_valid_stock_list()
        assert isinstance(result, pd.DataFrame)
        # ST股票应该被过滤
        assert 'ST0001.SZ' not in result['ts_code'].values

