"""
左侧起爆点模型特征工程测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock
from src.models.stock_selection.left_breakout.left_feature_engineering import LeftBreakoutFeatureEngineering


class TestLeftBreakoutFeatureEngineering:
    """LeftBreakoutFeatureEngineering测试类"""
    
    @pytest.fixture
    def feature_engineer(self):
        """创建特征工程器实例"""
        return LeftBreakoutFeatureEngineering()
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据（34天）"""
        dates = pd.date_range(end=datetime.now(), periods=34, freq='D')
        data = pd.DataFrame({
            'unique_sample_id': ['sample_001'] * 34,
            'ts_code': ['000001.SZ'] * 34,
            'name': ['平安银行'] * 34,
            't0_date': ['20240101'] * 34,
            'label': [1] * 34,
            'days_to_t1': list(range(34, 0, -1)),  # 从34天前到1天前
            'trade_date': dates.strftime('%Y%m%d'),
            'open': np.random.uniform(10, 12, 34),
            'high': np.random.uniform(11, 13, 34),
            'low': np.random.uniform(9, 11, 34),
            'close': np.random.uniform(10, 12, 34),
            'vol': np.random.uniform(1000000, 5000000, 34),
            'amount': np.random.uniform(10000000, 50000000, 34),
        })
        return data
    
    def test_init(self, feature_engineer):
        """测试初始化"""
        assert feature_engineer.feature_columns == []
        assert 'bottom_oscillation' in feature_engineer.feature_categories
        assert 'breakout_signals' in feature_engineer.feature_categories
        assert 'volume_price' in feature_engineer.feature_categories
        assert 'technical_indicators' in feature_engineer.feature_categories
        assert 'market_context' in feature_engineer.feature_categories
    
    def test_extract_features_empty(self, feature_engineer):
        """测试空数据提取特征"""
        empty_df = pd.DataFrame()
        result = feature_engineer.extract_features(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_extract_features_basic(self, feature_engineer, sample_data):
        """测试基本特征提取"""
        result = feature_engineer.extract_features(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'unique_sample_id' in result.columns
        assert 'ts_code' in result.columns
        assert 'name' in result.columns
        assert len(feature_engineer.feature_columns) > 0
    
    def test_extract_features_insufficient_data(self, feature_engineer):
        """测试数据不足的情况"""
        # 只有10天数据，少于20天要求
        dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
        data = pd.DataFrame({
            'unique_sample_id': ['sample_001'] * 10,
            'ts_code': ['000001.SZ'] * 10,
            'name': ['平安银行'] * 10,
            't0_date': ['20240101'] * 10,
            'label': [1] * 10,
            'days_to_t1': list(range(10, 0, -1)),
            'trade_date': dates.strftime('%Y%m%d'),
            'close': np.random.uniform(10, 12, 10),
        })
        
        result = feature_engineer.extract_features(data)
        # 应该返回空DataFrame，因为数据不足
        assert isinstance(result, pd.DataFrame)
    
    def test_extract_single_sample_features(self, feature_engineer, sample_data):
        """测试单样本特征提取"""
        sample = sample_data[sample_data['unique_sample_id'] == 'sample_001']
        features = feature_engineer._extract_single_sample_features(sample)
        
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_extract_bottom_oscillation_features(self, feature_engineer, sample_data):
        """测试底部震荡特征提取"""
        sample = sample_data[sample_data['unique_sample_id'] == 'sample_001']
        features = feature_engineer._extract_bottom_oscillation_features(sample)
        
        assert isinstance(features, dict)
        # 检查是否包含底部震荡相关特征
        assert any('bottom' in key.lower() or 'oscillation' in key.lower() 
                   for key in features.keys())
    
    def test_extract_breakout_signal_features(self, feature_engineer, sample_data):
        """测试预转信号特征提取"""
        sample = sample_data[sample_data['unique_sample_id'] == 'sample_001']
        features = feature_engineer._extract_breakout_signal_features(sample)
        
        assert isinstance(features, dict)
    
    def test_extract_volume_price_features(self, feature_engineer, sample_data):
        """测试量价配合特征提取"""
        sample = sample_data[sample_data['unique_sample_id'] == 'sample_001']
        features = feature_engineer._extract_volume_price_features(sample)
        
        assert isinstance(features, dict)
        # 检查是否包含量价相关特征
        assert any('volume' in key.lower() or 'price' in key.lower() 
                   for key in features.keys())
    
    def test_extract_technical_indicator_features(self, feature_engineer, sample_data):
        """测试技术指标特征提取"""
        sample = sample_data[sample_data['unique_sample_id'] == 'sample_001']
        features = feature_engineer._extract_technical_indicator_features(sample)
        
        assert isinstance(features, dict)
    
    def test_calculate_skewness(self, feature_engineer):
        """测试偏度计算"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        skewness = feature_engineer._calculate_skewness(data)
        assert isinstance(skewness, (int, float))
        assert not np.isnan(skewness)
    
    def test_calculate_kurtosis(self, feature_engineer):
        """测试峰度计算"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        kurtosis = feature_engineer._calculate_kurtosis(data)
        assert isinstance(kurtosis, (int, float))
        assert not np.isnan(kurtosis)
    
    def test_calculate_ma_convergence(self, feature_engineer, sample_data):
        """测试均线收敛计算"""
        sample = sample_data[sample_data['unique_sample_id'] == 'sample_001']
        result = feature_engineer._calculate_ma_convergence(sample)
        
        assert isinstance(result, dict)
    
    def test_calculate_bollinger_contraction(self, feature_engineer, sample_data):
        """测试布林带收缩计算"""
        sample = sample_data[sample_data['unique_sample_id'] == 'sample_001']
        result = feature_engineer._calculate_bollinger_contraction(sample)
        
        assert isinstance(result, dict)
    
    def test_extract_features_multiple_samples(self, feature_engineer):
        """测试多个样本的特征提取"""
        # 创建两个样本的数据
        dates1 = pd.date_range(end=datetime.now(), periods=34, freq='D')
        dates2 = pd.date_range(end=datetime.now() - timedelta(days=50), periods=34, freq='D')
        
        data1 = pd.DataFrame({
            'unique_sample_id': ['sample_001'] * 34,
            'ts_code': ['000001.SZ'] * 34,
            'name': ['平安银行'] * 34,
            't0_date': ['20240101'] * 34,
            'label': [1] * 34,
            'days_to_t1': list(range(34, 0, -1)),
            'trade_date': dates1.strftime('%Y%m%d'),
            'close': np.random.uniform(10, 12, 34),
            'vol': np.random.uniform(1000000, 5000000, 34),
        })
        
        data2 = pd.DataFrame({
            'unique_sample_id': ['sample_002'] * 34,
            'ts_code': ['600000.SH'] * 34,
            'name': ['浦发银行'] * 34,
            't0_date': ['20240115'] * 34,
            'label': [0] * 34,
            'days_to_t1': list(range(34, 0, -1)),
            'trade_date': dates2.strftime('%Y%m%d'),
            'close': np.random.uniform(8, 10, 34),
            'vol': np.random.uniform(1000000, 5000000, 34),
        })
        
        combined_data = pd.concat([data1, data2], ignore_index=True)
        result = feature_engineer.extract_features(combined_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # 两个样本
        assert len(result['unique_sample_id'].unique()) == 2

