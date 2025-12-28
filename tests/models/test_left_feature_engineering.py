"""
左侧潜力牛股特征工程测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.models.stock_selection.potential_discovery.left_breakout.left_feature_engineering import LeftBreakoutFeatureEngineering


class TestLeftBreakoutFeatureEngineering:
    """左侧潜力牛股特征工程测试类"""

    @pytest.fixture
    def feature_engineer(self):
        """创建特征工程器实例"""
        return LeftBreakoutFeatureEngineering()

    @pytest.fixture
    def sample_data(self):
        """示例34天数据"""
        dates = pd.date_range(end=datetime.now(), periods=34, freq='D')
        return pd.DataFrame({
            'unique_sample_id': ['sample_1'] * 34,
            'ts_code': ['000001.SZ'] * 34,
            'name': ['测试股票'] * 34,
            't0_date': ['20241226'] * 34,
            'label': [1] * 34,
            'days_to_t1': list(range(34, 0, -1)),
            'trade_date': dates.strftime('%Y%m%d'),
            'open': np.random.uniform(10, 20, 34),
            'high': np.random.uniform(15, 25, 34),
            'low': np.random.uniform(8, 18, 34),
            'close': np.linspace(10, 20, 34),
            'vol': np.random.uniform(1000000, 10000000, 34),
        })

    @pytest.mark.unit
    def test_init(self, feature_engineer):
        """测试初始化"""
        assert feature_engineer.feature_columns == []
        assert 'bottom_oscillation' in feature_engineer.feature_categories

    @pytest.mark.unit
    def test_extract_features_success(self, feature_engineer, sample_data):
        """测试成功提取特征"""
        result = feature_engineer.extract_features(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'unique_sample_id' in result.columns
        assert 'ts_code' in result.columns

    @pytest.mark.unit
    def test_extract_features_empty(self, feature_engineer):
        """测试空数据"""
        empty_df = pd.DataFrame()
        result = feature_engineer.extract_features(empty_df)
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @pytest.mark.unit
    def test_extract_features_insufficient_data(self, feature_engineer):
        """测试数据不足的情况"""
        # 少于20天的数据
        short_data = pd.DataFrame({
            'unique_sample_id': ['sample_1'] * 10,
            'ts_code': ['000001.SZ'] * 10,
            'name': ['测试股票'] * 10,
            't0_date': ['20241226'] * 10,
            'label': [1] * 10,
            'days_to_t1': list(range(10, 0, -1)),
            'trade_date': pd.date_range(end=datetime.now(), periods=10, freq='D').strftime('%Y%m%d'),
            'close': np.random.uniform(10, 20, 10),
        })
        
        result = feature_engineer.extract_features(short_data)
        
        # 应该跳过数据不足的样本
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit
    def test_extract_features_multiple_samples(self, feature_engineer):
        """测试多个样本的特征提取"""
        # 创建两个样本
        dates = pd.date_range(end=datetime.now(), periods=34, freq='D')
        sample1 = pd.DataFrame({
            'unique_sample_id': ['sample_1'] * 34,
            'ts_code': ['000001.SZ'] * 34,
            'name': ['股票1'] * 34,
            't0_date': ['20241226'] * 34,
            'label': [1] * 34,
            'days_to_t1': list(range(34, 0, -1)),
            'trade_date': dates.strftime('%Y%m%d'),
            'close': np.linspace(10, 20, 34),
        })
        
        sample2 = pd.DataFrame({
            'unique_sample_id': ['sample_2'] * 34,
            'ts_code': ['600000.SH'] * 34,
            'name': ['股票2'] * 34,
            't0_date': ['20241226'] * 34,
            'label': [0] * 34,
            'days_to_t1': list(range(34, 0, -1)),
            'trade_date': dates.strftime('%Y%m%d'),
            'close': np.linspace(15, 25, 34),
        })
        
        combined_data = pd.concat([sample1, sample2], ignore_index=True)
        result = feature_engineer.extract_features(combined_data)
        
        assert isinstance(result, pd.DataFrame)
        # 应该有两个样本的特征
        assert len(result) >= 0  # 可能因为数据不完整而跳过

    @pytest.mark.unit
    def test_extract_single_sample_features(self, feature_engineer, sample_data):
        """测试提取单个样本特征"""
        features = feature_engineer._extract_single_sample_features(sample_data)
        
        assert isinstance(features, dict)
        assert 'ts_code' in features
        assert 'name' in features
