"""
负样本筛选器测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.strategy.screening.negative_sample_screener import NegativeSampleScreener


class TestNegativeSampleScreener:
    """负样本筛选器测试类"""

    @pytest.fixture
    def mock_data_manager(self):
        """模拟DataManager"""
        mock_dm = Mock()
        
        # 模拟日线数据
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        mock_daily_data = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'open': np.random.uniform(10, 20, 100),
            'high': np.random.uniform(15, 25, 100),
            'low': np.random.uniform(8, 18, 100),
            'close': np.random.uniform(10, 20, 100),
            'vol': np.random.uniform(1000000, 10000000, 100),
        })
        mock_dm.get_daily_data.return_value = mock_daily_data
        
        # 模拟日线基础数据
        mock_daily_basic = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'volume_ratio': np.random.uniform(0.5, 3.0, 100),
        })
        mock_dm.get_daily_basic.return_value = mock_daily_basic
        
        return mock_dm

    @pytest.fixture
    def screener(self, mock_data_manager):
        """创建筛选器实例"""
        return NegativeSampleScreener(mock_data_manager)

    @pytest.fixture
    def sample_positive_features(self):
        """示例正样本特征数据"""
        dates = pd.date_range(end=datetime.now(), periods=34, freq='D')
        return pd.DataFrame({
            'sample_id': ['sample_1'] * 34,
            'ts_code': ['000001.SZ'] * 34,
            'trade_date': dates.strftime('%Y%m%d'),
            'volume_ratio': np.random.uniform(1.5, 3.0, 34),
            'macd_dif': np.random.uniform(-0.5, 0.5, 34),
            'pct_chg': np.random.uniform(-5, 10, 34),
        })

    @pytest.mark.unit
    def test_init(self, screener, mock_data_manager):
        """测试初始化"""
        assert screener.dm == mock_data_manager
        assert screener.negative_samples == []

    @pytest.mark.unit
    def test_analyze_positive_features(self, screener, sample_positive_features):
        """测试分析正样本特征"""
        stats = screener.analyze_positive_features(sample_positive_features)
        
        assert 'total_samples' in stats
        assert 'volume_ratio_gt2_count' in stats
        assert 'macd_turn_positive_count' in stats
        assert stats['total_samples'] == 1

    @pytest.mark.unit
    def test_analyze_positive_features_empty(self, screener):
        """测试分析空数据"""
        empty_df = pd.DataFrame()
        stats = screener.analyze_positive_features(empty_df)
        
        assert stats['total_samples'] == 0

    @pytest.mark.unit
    def test_analyze_positive_features_multiple_samples(self, screener):
        """测试分析多个正样本"""
        # 创建多个样本
        dates = pd.date_range(end=datetime.now(), periods=34, freq='D')
        samples = []
        
        for i in range(3):
            sample_data = pd.DataFrame({
                'sample_id': [f'sample_{i}'] * 34,
                'ts_code': [f'00000{i}.SZ'] * 34,
                'trade_date': dates.strftime('%Y%m%d'),
                'volume_ratio': np.random.uniform(1.5, 3.0, 34),
                'macd_dif': np.random.uniform(-0.5, 0.5, 34),
                'pct_chg': np.random.uniform(-5, 10, 34),
            })
            samples.append(sample_data)
        
        combined_df = pd.concat(samples, ignore_index=True)
        stats = screener.analyze_positive_features(combined_df)
        
        assert stats['total_samples'] == 3

    @pytest.mark.unit
    def test_analyze_positive_features_missing_columns(self, screener):
        """测试缺少某些列的情况"""
        dates = pd.date_range(end=datetime.now(), periods=34, freq='D')
        incomplete_df = pd.DataFrame({
            'sample_id': ['sample_1'] * 34,
            'ts_code': ['000001.SZ'] * 34,
            'trade_date': dates.strftime('%Y%m%d'),
            'volume_ratio': np.random.uniform(1.5, 3.0, 34),
            # 缺少macd_dif和pct_chg
        })
        
        # 应该能正常处理
        stats = screener.analyze_positive_features(incomplete_df)
        assert stats['total_samples'] == 1

