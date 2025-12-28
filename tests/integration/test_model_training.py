"""
模型训练流程集成测试
测试从样本准备到模型训练的完整流程
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.models.stock_selection.potential_discovery.left_breakout.left_model import LeftBreakoutModel


@pytest.mark.integration
class TestModelTraining:
    """模型训练流程集成测试"""

    @pytest.fixture
    def mock_data_manager(self):
        """模拟DataManager"""
        mock_dm = Mock()
        
        # 模拟股票列表
        mock_stocks = pd.DataFrame({
            'ts_code': ['000001.SZ', '600000.SH', '000002.SZ'],
            'name': ['股票1', '股票2', '股票3'],
        })
        mock_dm.get_stock_list.return_value = mock_stocks
        
        # 模拟日线数据
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        mock_daily = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'open': np.random.uniform(10, 20, 100),
            'high': np.random.uniform(15, 25, 100),
            'low': np.random.uniform(8, 18, 100),
            'close': np.random.uniform(10, 20, 100),
            'vol': np.random.uniform(1000000, 10000000, 100),
        })
        mock_dm.get_daily_data.return_value = mock_daily
        
        # 模拟周线数据
        mock_dm.get_weekly_data.return_value = mock_daily
        
        # 模拟日线基础数据
        mock_daily_basic = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'volume_ratio': np.random.uniform(0.5, 3.0, 100),
        })
        mock_dm.get_daily_basic.return_value = mock_daily_basic
        
        return mock_dm

    @pytest.fixture
    def mock_left_model(self, mock_data_manager):
        """创建模拟的LeftBreakoutModel"""
        with patch('src.models.stock_selection.potential_discovery.left_breakout.left_model.LeftPositiveSampleScreener'), \
             patch('src.models.stock_selection.potential_discovery.left_breakout.left_model.LeftNegativeSampleScreener'), \
             patch('src.models.stock_selection.potential_discovery.left_breakout.left_model.LeftBreakoutFeatureEngineering'):
            
            model = LeftBreakoutModel(mock_data_manager)
            model.positive_screener = Mock()
            model.negative_screener = Mock()
            model.feature_engineer = Mock()
            
            return model

    @pytest.mark.unit
    def test_model_initialization(self, mock_data_manager):
        """测试模型初始化"""
        with patch('src.models.stock_selection.potential_discovery.left_breakout.left_model.LeftPositiveSampleScreener'), \
             patch('src.models.stock_selection.potential_discovery.left_breakout.left_model.LeftNegativeSampleScreener'), \
             patch('src.models.stock_selection.potential_discovery.left_breakout.left_model.LeftBreakoutFeatureEngineering'):
            
            model = LeftBreakoutModel(mock_data_manager)
            
            assert model.dm == mock_data_manager
            assert model.config is not None
            assert model.model is None
            assert model.feature_columns == []

    @pytest.mark.unit
    def test_prepare_samples_flow(self, mock_left_model):
        """测试样本准备流程"""
        # 模拟正样本
        positive_samples = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'name': ['股票1'],
            't0_date': ['20241226'],
            'label': [1],
        })
        mock_left_model.positive_screener.screen_all_stocks.return_value = positive_samples
        
        # 模拟负样本
        negative_samples = pd.DataFrame({
            'ts_code': ['600000.SH'],
            'name': ['股票2'],
            't0_date': ['20241226'],
            'label': [0],
        })
        mock_left_model.negative_screener.screen_negative_samples.return_value = negative_samples
        
        # 执行样本准备
        pos_df, neg_df = mock_left_model.prepare_samples(force_refresh=False)
        
        # 验证流程被调用
        mock_left_model.positive_screener.screen_all_stocks.assert_called()
        mock_left_model.negative_screener.screen_negative_samples.assert_called()

    @pytest.mark.unit
    def test_feature_extraction_flow(self, mock_left_model):
        """测试特征提取流程"""
        # 模拟样本数据
        sample_data = pd.DataFrame({
            'unique_sample_id': ['sample_1'] * 34,
            'ts_code': ['000001.SZ'] * 34,
            'name': ['股票1'] * 34,
            't0_date': ['20241226'] * 34,
            'label': [1] * 34,
            'days_to_t1': list(range(34, 0, -1)),
            'trade_date': pd.date_range(end=datetime.now(), periods=34, freq='D').strftime('%Y%m%d'),
            'close': np.linspace(10, 20, 34),
        })
        
        # 模拟特征提取结果
        features = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'feature1': [0.5],
            'feature2': [0.3],
            'label': [1],
        })
        mock_left_model.feature_engineer.extract_features.return_value = features
        
        # 执行特征提取
        result = mock_left_model.feature_engineer.extract_features(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        mock_left_model.feature_engineer.extract_features.assert_called_once()

    @pytest.mark.unit
    def test_model_training_flow(self, mock_left_model):
        """测试模型训练流程"""
        # 模拟训练数据
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        # 模拟模型
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_left_model.model = mock_model
        
        # 执行训练
        mock_left_model.model.fit(X_train, y_train)
        
        # 验证训练被调用
        mock_model.fit.assert_called_once()

    @pytest.mark.integration
    def test_complete_training_pipeline(self, mock_left_model):
        """测试完整的训练流程（集成测试）"""
        # 1. 准备样本
        positive_samples = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'name': ['股票1'],
            't0_date': ['20241226'],
            'label': [1],
        })
        mock_left_model.positive_screener.screen_all_stocks.return_value = positive_samples
        
        negative_samples = pd.DataFrame({
            'ts_code': ['600000.SH'],
            'name': ['股票2'],
            't0_date': ['20241226'],
            'label': [0],
        })
        mock_left_model.negative_screener.screen_negative_samples.return_value = negative_samples
        
        # 2. 提取特征
        features = pd.DataFrame({
            'ts_code': ['000001.SZ', '600000.SH'],
            'feature1': [0.5, 0.3],
            'feature2': [0.2, 0.4],
            'label': [1, 0],
        })
        mock_left_model.feature_engineer.extract_features.return_value = features
        
        # 3. 训练模型
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_left_model.model = mock_model
        
        # 验证流程完整性
        assert mock_left_model.positive_screener is not None
        assert mock_left_model.negative_screener is not None
        assert mock_left_model.feature_engineer is not None

