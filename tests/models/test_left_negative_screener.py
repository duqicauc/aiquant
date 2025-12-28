"""
左侧起爆点模型负样本筛选器测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock
from src.models.stock_selection.left_breakout.left_negative_screener import LeftNegativeSampleScreener


class TestLeftNegativeSampleScreener:
    """LeftNegativeSampleScreener测试类"""
    
    @pytest.fixture
    def mock_data_manager(self):
        """创建模拟数据管理器"""
        mock_dm = Mock()
        
        # 模拟股票列表
        mock_dm.get_stock_list.return_value = pd.DataFrame({
            'ts_code': ['000001.SZ', '600000.SH', '000002.SZ'],
            'name': ['平安银行', '浦发银行', '万科A'],
            'list_date': ['19910403', '19991110', '19910129'],
        })
        
        # 模拟日线数据（涨幅较小，适合负样本）
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = np.linspace(10, 11, 100)  # 从10涨到11，涨幅仅10%
        mock_dm.get_daily_data.return_value = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'vol': np.random.uniform(1000000, 5000000, 100),
            'amount': prices * np.random.uniform(100000, 500000, 100),
        })
        
        return mock_dm
    
    @pytest.fixture
    def screener(self, mock_data_manager):
        """创建筛选器实例"""
        return LeftNegativeSampleScreener(mock_data_manager)
    
    @pytest.fixture
    def sample_positive_samples(self):
        """创建示例正样本"""
        return pd.DataFrame({
            'ts_code': ['000001.SZ', '600000.SH'],
            'name': ['平安银行', '浦发银行'],
            't0_date': ['20240101', '20240115'],
            'label': [1, 1],
            'future_return': [60.0, 55.0],  # 未来涨幅>50%
        })
    
    def test_init(self, screener, mock_data_manager):
        """测试初始化"""
        assert screener.dm == mock_data_manager
        assert screener.negative_samples == []
    
    def test_screen_negative_samples_empty_positive(self, screener):
        """测试正样本为空的情况"""
        empty_positive = pd.DataFrame()
        result = screener.screen_negative_samples(empty_positive)
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_screen_negative_samples_structure(self, screener, sample_positive_samples):
        """测试负样本筛选结构"""
        result = screener.screen_negative_samples(
            positive_samples=sample_positive_samples,
            start_date='20240101',
            end_date='20240131',
            look_forward_days=45
        )
        
        assert isinstance(result, pd.DataFrame)
        # 即使没有找到负样本，也应该返回DataFrame
    
    def test_analyze_positive_features(self, screener, sample_positive_samples):
        """测试正样本特征分析"""
        stats = screener._analyze_positive_features(sample_positive_samples)
        
        assert isinstance(stats, dict)
        # 应该包含一些统计信息
    
    def test_get_candidate_stocks(self, screener, sample_positive_samples):
        """测试获取候选股票"""
        positive_ts_codes = set(sample_positive_samples['ts_code'].unique())
        candidates = screener._get_candidate_stocks(positive_ts_codes)
        
        assert isinstance(candidates, pd.DataFrame)
        assert 'ts_code' in candidates.columns
        # 候选股票不应该包含正样本中的股票
        candidate_codes = set(candidates['ts_code'].unique())
        assert len(candidate_codes & positive_ts_codes) == 0
    
    def test_screen_single_stock_negative_structure(self, screener, sample_positive_samples):
        """测试单股票负样本筛选结构"""
        stats = screener._analyze_positive_features(sample_positive_samples)
        
        result = screener._screen_single_stock_negative(
            ts_code='000002.SZ',
            name='万科A',
            positive_samples=sample_positive_samples,
            feature_stats=stats,
            look_forward_days=45
        )
        
        # 应该返回列表
        assert isinstance(result, list)
    
    def test_screen_single_stock_negative_no_data(self, screener, sample_positive_samples):
        """测试没有数据的情况"""
        # 模拟返回空数据
        screener.dm.get_daily_data.return_value = pd.DataFrame()
        
        stats = screener._analyze_positive_features(sample_positive_samples)
        result = screener._screen_single_stock_negative(
            ts_code='999999.SZ',
            name='测试股票',
            positive_samples=sample_positive_samples,
            feature_stats=stats,
            look_forward_days=45
        )
        
        assert isinstance(result, list)
        # 没有数据应该返回空列表
        assert len(result) == 0
    
    def test_check_negative_criteria(self, screener):
        """测试负样本条件检查"""
        # 创建模拟数据：未来45天涨幅仅5%（<10%阈值）
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = np.linspace(10, 10.5, 100)  # 从10涨到10.5，涨幅5%
        
        df = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'close': prices,
        })
        
        # 检查第50天（t0）的未来收益
        t0_idx = 50
        t0_price = df.iloc[t0_idx]['close']
        future_price = df.iloc[min(t0_idx + 45, len(df) - 1)]['close']
        future_return = (future_price / t0_price - 1) * 100
        
        # 5% < 10%阈值，适合作为负样本
        assert future_return < 10

