"""
回测数据适配器测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.backtest.data_feed import (
    TushareData,
    DataFeedManager,
    create_data_feed
)


class TestTushareData:
    """TushareData测试类"""

    @pytest.mark.unit
    def test_tushare_data_params(self):
        """测试TushareData参数配置"""
        assert TushareData.params is not None
        assert hasattr(TushareData.params, 'datetime')
        assert hasattr(TushareData.params, 'open')
        assert hasattr(TushareData.params, 'close')


class TestDataFeedManager:
    """DataFeedManager测试类"""

    @pytest.fixture
    def mock_data_manager(self):
        """模拟DataManager"""
        mock_dm = Mock()
        
        # 创建示例数据
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        mock_data = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'open': np.random.uniform(10, 20, 30),
            'high': np.random.uniform(15, 25, 30),
            'low': np.random.uniform(8, 18, 30),
            'close': np.random.uniform(10, 20, 30),
            'vol': np.random.uniform(1000000, 10000000, 30),
        })
        
        mock_dm.get_daily_data.return_value = mock_data
        return mock_dm

    @pytest.fixture
    def data_feed_manager(self, mock_data_manager):
        """创建DataFeedManager实例"""
        with patch('src.backtest.data_feed.DataManager', return_value=mock_data_manager):
            manager = DataFeedManager()
            manager.dm = mock_data_manager
            return manager

    @pytest.mark.unit
    def test_get_data_feed_success(self, data_feed_manager, mock_data_manager):
        """测试成功获取数据Feed"""
        feed = data_feed_manager.get_data_feed(
            stock_code='000001.SZ',
            start_date='20240101',
            end_date='20241231'
        )
        
        assert feed is not None
        assert isinstance(feed, TushareData)
        mock_data_manager.get_daily_data.assert_called_once()

    @pytest.mark.unit
    def test_get_data_feed_empty_data(self, data_feed_manager, mock_data_manager):
        """测试获取空数据的情况"""
        mock_data_manager.get_daily_data.return_value = pd.DataFrame()
        
        feed = data_feed_manager.get_data_feed(
            stock_code='000001.SZ',
            start_date='20240101',
            end_date='20241231'
        )
        
        assert feed is None

    @pytest.mark.unit
    def test_get_multiple_feeds(self, data_feed_manager, mock_data_manager):
        """测试获取多只股票的数据Feed"""
        stock_codes = ['000001.SZ', '600000.SH', '000002.SZ']
        
        # 确保返回的数据包含trade_date列
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        mock_daily_with_date = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'open': np.random.uniform(10, 20, 30),
            'high': np.random.uniform(15, 25, 30),
            'low': np.random.uniform(8, 18, 30),
            'close': np.random.uniform(10, 20, 30),
            'vol': np.random.uniform(1000000, 10000000, 30),
        })
        mock_data_manager.get_daily_data.return_value = mock_daily_with_date
        
        feeds = data_feed_manager.get_multiple_feeds(
            stock_codes=stock_codes,
            start_date='20240101',
            end_date='20241231'
        )
        
        assert len(feeds) == len(stock_codes)
        assert all(isinstance(feed, TushareData) for feed in feeds.values())
        assert mock_data_manager.get_daily_data.call_count == len(stock_codes)

    @pytest.mark.unit
    def test_prepare_data(self, data_feed_manager):
        """测试数据预处理"""
        # 创建原始数据
        dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
        raw_data = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'open': [10.0] * 10,
            'high': [15.0] * 10,
            'low': [8.0] * 10,
            'close': [12.0] * 10,
            'vol': [1000000] * 10,
        })
        
        # 预处理
        processed_data = data_feed_manager._prepare_data(raw_data, '000001.SZ')
        
        # 检查结果
        assert isinstance(processed_data.index, pd.DatetimeIndex)
        assert len(processed_data) == 10
        assert 'open' in processed_data.columns
        assert 'close' in processed_data.columns
        assert processed_data.index.name == 'trade_date' or processed_data.index.name is None

    @pytest.mark.unit
    def test_prepare_data_missing_columns(self, data_feed_manager):
        """测试缺少必要列的情况"""
        incomplete_data = pd.DataFrame({
            'trade_date': ['20240101'],
            'open': [10.0],
        })
        
        with pytest.raises(ValueError, match="缺少必要列"):
            data_feed_manager._prepare_data(incomplete_data, '000001.SZ')

    @pytest.mark.unit
    def test_prepare_data_duplicate_dates(self, data_feed_manager):
        """测试处理重复日期"""
        dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
        raw_data = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d').tolist() + [dates.strftime('%Y%m%d')[0]],
            'open': [10.0] * 6,
            'high': [15.0] * 6,
            'low': [8.0] * 6,
            'close': [12.0] * 6,
            'vol': [1000000] * 6,
        })
        
        processed_data = data_feed_manager._prepare_data(raw_data, '000001.SZ')
        
        # 应该去重
        assert len(processed_data) <= len(raw_data)

    @pytest.mark.unit
    def test_prepare_data_invalid_data(self, data_feed_manager):
        """测试处理无效数据（价格为0）"""
        dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
        raw_data = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'open': [10.0, 0.0, 10.0, 10.0, 10.0],
            'high': [15.0, 15.0, 15.0, 15.0, 15.0],
            'low': [8.0, 8.0, 8.0, 8.0, 8.0],
            'close': [12.0, 0.0, 12.0, 12.0, 12.0],
            'vol': [1000000, 0, 1000000, 1000000, 1000000],
        })
        
        processed_data = data_feed_manager._prepare_data(raw_data, '000001.SZ')
        
        # 应该过滤掉价格为0或成交量为0的数据
        assert len(processed_data) < len(raw_data)
        assert all(processed_data['close'] > 0)
        assert all(processed_data['vol'] > 0)


class TestCreateDataFeed:
    """create_data_feed函数测试类"""

    @pytest.mark.unit
    @patch('src.backtest.data_feed.DataFeedManager')
    def test_create_data_feed(self, mock_manager_class):
        """测试create_data_feed便捷函数"""
        mock_manager = Mock()
        mock_feed = Mock()
        mock_manager.get_data_feed.return_value = mock_feed
        mock_manager_class.return_value = mock_manager
        
        feed = create_data_feed('000001.SZ', '20240101', '20241231')
        
        assert feed == mock_feed
        mock_manager.get_data_feed.assert_called_once_with(
            '000001.SZ', '20240101', '20241231'
        )
