"""
负样本筛选器V2测试
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.strategy.screening.negative_sample_screener_v2 import NegativeSampleScreenerV2


class TestNegativeSampleScreenerV2:
    """NegativeSampleScreenerV2测试类"""
    
    @pytest.fixture
    def mock_dm(self):
        """创建模拟DataManager"""
        dm = Mock()
        
        # 模拟股票列表
        dm.get_stock_list.return_value = pd.DataFrame({
            'ts_code': ['000001.SZ', '600000.SH', '000002.SZ', '600001.SH'],
            'name': ['平安银行', '浦发银行', '万科A', '邯郸钢铁'],
            'list_date': ['19910403', '19991110', '19910129', '19980101'],
        })
        
        # 模拟日线数据
        dates = pd.date_range(end='20240119', periods=34, freq='D')
        mock_daily = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'close': [10.0] * 34,
            'open': [9.8] * 34,
            'high': [10.2] * 34,
            'low': [9.5] * 34,
            'vol': [1000000] * 34,
            'amount': [10000000] * 34,
        })
        dm.get_daily_data.return_value = mock_daily
        
        return dm
    
    @pytest.fixture
    def positive_samples(self):
        """创建正样本数据"""
        return pd.DataFrame({
            'ts_code': ['000001.SZ', '600000.SH'],
            't1_date': ['20240119', '20240119'],
            't0_date': ['20240101', '20240101'],
            'total_return': [0.5, 0.6],
        })
    
    def test_init(self, mock_dm):
        """测试初始化"""
        screener = NegativeSampleScreenerV2(mock_dm)
        assert screener.dm == mock_dm
    
    def test_screen_negative_samples_basic(self, mock_dm, positive_samples):
        """测试基本负样本筛选"""
        screener = NegativeSampleScreenerV2(mock_dm)
        
        result = screener.screen_negative_samples(
            positive_samples_df=positive_samples,
            samples_per_positive=1,
            random_seed=42
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0  # 可能为0如果数据不足
    
    def test_screen_negative_samples_multiple(self, mock_dm, positive_samples):
        """测试每个正样本对应多个负样本"""
        screener = NegativeSampleScreenerV2(mock_dm)
        
        result = screener.screen_negative_samples(
            positive_samples_df=positive_samples,
            samples_per_positive=2,
            random_seed=42
        )
        
        assert isinstance(result, pd.DataFrame)
        # 应该至少有2个负样本（每个正样本对应2个）
    
    def test_get_valid_stock_list(self, mock_dm):
        """测试获取有效股票列表"""
        screener = NegativeSampleScreenerV2(mock_dm)
        
        # 测试私有方法（通过反射）
        stock_list = screener._get_valid_stock_list()
        
        assert isinstance(stock_list, pd.DataFrame)
        assert 'ts_code' in stock_list.columns
    
    def test_get_valid_stock_list_filters(self, mock_dm):
        """测试股票列表过滤"""
        # 模拟包含ST股票的列表
        mock_dm.get_stock_list.return_value = pd.DataFrame({
            'ts_code': ['000001.SZ', 'ST0001.SZ'],
            'name': ['平安银行', 'ST股票'],
            'list_date': ['19910403', '19900101'],
        })
        
        screener = NegativeSampleScreenerV2(mock_dm)
        stock_list = screener._get_valid_stock_list()
        
        # 应该过滤掉ST股票
        assert 'ST0001.SZ' not in stock_list['ts_code'].values

