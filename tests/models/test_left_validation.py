"""
左侧潜力牛股模型验证器测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.models.stock_selection.potential_discovery.left_breakout.left_validation import LeftBreakoutValidator


class TestLeftBreakoutValidator:
    """左侧潜力牛股模型验证器测试类"""

    @pytest.fixture
    def mock_left_model(self):
        """模拟LeftBreakoutModel"""
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.feature_columns = ['feature1', 'feature2', 'feature3']
        
        # 模拟预测结果
        mock_model.model.predict_proba.return_value = np.array([[0.3, 0.7], [0.2, 0.8]])
        mock_model.model.predict.return_value = np.array([1, 1])
        
        return mock_model

    @pytest.fixture
    def validator(self, mock_left_model):
        """创建验证器实例"""
        return LeftBreakoutValidator(mock_left_model)

    @pytest.fixture
    def sample_features_df(self):
        """示例特征DataFrame"""
        dates = pd.date_range(start='20200101', periods=1000, freq='D')
        return pd.DataFrame({
            'unique_sample_id': [f'sample_{i}' for i in range(1000)],
            'ts_code': [f'00000{i%10}.SZ' for i in range(1000)],
            'name': [f'股票{i}' for i in range(1000)],
            't0_date': dates.strftime('%Y%m%d'),
            'label': np.random.randint(0, 2, 1000),
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000),
        })

    @pytest.mark.unit
    def test_init(self, validator, mock_left_model):
        """测试初始化"""
        assert validator.model == mock_left_model
        assert validator.validation_results == {}

    @pytest.mark.unit
    def test_walk_forward_validation_success(self, validator, sample_features_df):
        """测试成功执行Walk-Forward验证"""
        result = validator.walk_forward_validation(
            features_df=sample_features_df,
            n_splits=3,
            min_train_samples=100
        )
        
        assert isinstance(result, dict)
        # 应该有验证结果

    @pytest.mark.unit
    def test_walk_forward_validation_insufficient_samples(self, validator, sample_features_df):
        """测试样本不足的情况"""
        small_df = sample_features_df.head(50)
        
        result = validator.walk_forward_validation(
            features_df=small_df,
            n_splits=3,
            min_train_samples=100
        )
        
        assert isinstance(result, dict)
        # 应该返回空结果或错误信息

    @pytest.mark.unit
    def test_walk_forward_validation_empty_data(self, validator):
        """测试空数据"""
        empty_df = pd.DataFrame()
        
        result = validator.walk_forward_validation(
            features_df=empty_df,
            n_splits=3
        )
        
        assert isinstance(result, dict)

