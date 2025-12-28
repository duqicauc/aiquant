"""
左侧起爆点模型测试
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.models.stock_selection.left_breakout import LeftBreakoutModel


class TestLeftBreakoutModel:
    """LeftBreakoutModel测试类"""
    
    def test_init(self, mock_data_manager):
        """测试初始化"""
        config = {
            'model': {
                'version': 'v1',
                'training': {
                    'test_size': 0.2,
                    'time_series_split': True,
                    'n_splits': 5
                },
                'parameters': {
                    'n_estimators': 100,
                    'max_depth': 5
                }
            },
            'save': {
                'directory': 'data/models/left_breakout'
            }
        }
        
        model = LeftBreakoutModel(mock_data_manager, config)
        assert model.dm == mock_data_manager
        assert model.config == config
        assert model.positive_screener is not None
        assert model.negative_screener is not None
        assert model.feature_engineer is not None
    
    def test_get_default_config(self, mock_data_manager):
        """测试获取默认配置"""
        model = LeftBreakoutModel(mock_data_manager)
        assert model.config is not None
        assert 'model' in model.config
    
    def test_predict_stocks_structure(self, mock_data_manager):
        """测试预测股票结构"""
        model = LeftBreakoutModel(mock_data_manager)
        
        # 模拟特征数据
        features_df = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'name': ['平安银行'],
            'feature1': [0.5],
            'feature2': [0.3],
        })
        
        # 如果没有加载模型，应该返回空DataFrame
        result = model.predict_stocks(features_df)
        assert isinstance(result, pd.DataFrame)
    
    def test_get_feature_importance_structure(self, mock_data_manager):
        """测试获取特征重要性结构"""
        model = LeftBreakoutModel(mock_data_manager)
        
        # 如果没有模型，应该返回空DataFrame
        result = model.get_feature_importance()
        assert isinstance(result, pd.DataFrame)

