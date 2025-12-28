"""
回测数据Feed测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# 检查backtrader是否可用
try:
    import backtrader as bt
    from src.backtest.data_feed import DataFeedManager, TushareData
    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False
    # 创建Mock类用于测试
    class TushareData:
        params = (
            ('datetime', None),
            ('open', 'open'),
            ('high', 'high'),
            ('low', 'low'),
            ('close', 'close'),
            ('volume', 'vol'),
            ('openinterest', -1),
        )
    
    class DataFeedManager:
        def __init__(self):
            pass


@pytest.mark.skipif(not BACKTRADER_AVAILABLE, reason="backtrader not installed")
class TestDataFeedManager:
    """DataFeedManager测试类"""
    
    @pytest.fixture
    def feed_manager(self, mock_data_manager):
        """创建数据Feed管理器实例"""
        manager = DataFeedManager()
        manager.dm = mock_data_manager
        return manager
    
    @pytest.fixture
    def sample_daily_data(self):
        """创建示例日线数据"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        return pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'open': np.random.uniform(10, 12, 100),
            'high': np.random.uniform(11, 13, 100),
            'low': np.random.uniform(9, 11, 100),
            'close': np.random.uniform(10, 12, 100),
            'vol': np.random.uniform(1000000, 5000000, 100),
            'amount': np.random.uniform(10000000, 50000000, 100),
        })
    
    def test_init(self, feed_manager):
        """测试初始化"""
        assert feed_manager.dm is not None
    
    def test_get_data_feed_success(self, feed_manager, sample_daily_data):
        """测试成功获取数据Feed"""
        feed_manager.dm.get_daily_data.return_value = sample_daily_data
        
        feed = feed_manager.get_data_feed(
            stock_code='000001.SZ',
            start_date='20240101',
            end_date='20240131'
        )
        
        # 应该返回TushareData对象
        assert feed is not None
        assert isinstance(feed, TushareData)
    
    def test_get_data_feed_empty_data(self, feed_manager):
        """测试空数据的情况"""
        feed_manager.dm.get_daily_data.return_value = pd.DataFrame()
        
        feed = feed_manager.get_data_feed(
            stock_code='999999.SZ',
            start_date='20240101',
            end_date='20240131'
        )
        
        # 应该返回None
        assert feed is None
    
    def test_get_multiple_feeds(self, feed_manager, sample_daily_data):
        """测试获取多个数据Feed"""
        feed_manager.dm.get_daily_data.return_value = sample_daily_data
        
        stock_codes = ['000001.SZ', '600000.SH']
        feeds = feed_manager.get_multiple_feeds(
            stock_codes=stock_codes,
            start_date='20240101',
            end_date='20240131'
        )
        
        assert isinstance(feeds, dict)
        assert len(feeds) == len(stock_codes)
        for code in stock_codes:
            assert code in feeds
            assert isinstance(feeds[code], TushareData)
    
    def test_get_multiple_feeds_partial_success(self, feed_manager, sample_daily_data):
        """测试部分成功的情况"""
        # 第一个股票有数据，第二个没有
        def side_effect(stock_code, *args, **kwargs):
            if stock_code == '000001.SZ':
                return sample_daily_data
            else:
                return pd.DataFrame()
        
        feed_manager.dm.get_daily_data.side_effect = side_effect
        
        stock_codes = ['000001.SZ', '999999.SZ']
        feeds = feed_manager.get_multiple_feeds(
            stock_codes=stock_codes,
            start_date='20240101',
            end_date='20240131'
        )
        
        assert isinstance(feeds, dict)
        assert len(feeds) == 1  # 只有1个成功
        assert '000001.SZ' in feeds
        assert '999999.SZ' not in feeds
    
    def test_prepare_data(self, feed_manager, sample_daily_data):
        """测试数据预处理"""
        result = feed_manager._prepare_data(sample_daily_data, '000001.SZ')
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        # 检查必要的列是否存在
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'vol' in result.columns


@pytest.mark.skipif(not BACKTRADER_AVAILABLE, reason="backtrader not installed")
class TestTushareData:
    """TushareData测试类"""
    
    def test_tushare_data_params(self):
        """测试TushareData参数配置"""
        # 检查参数是否正确配置
        assert hasattr(TushareData, 'params')
        params = TushareData.params
        
        # 检查必要的参数
        param_names = [p[0] for p in params]
        assert 'open' in param_names
        assert 'high' in param_names
        assert 'low' in param_names
        assert 'close' in param_names
        assert 'volume' in param_names

