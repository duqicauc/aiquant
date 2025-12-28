"""
左侧起爆点模型预测器测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

# 延迟导入，避免在导入时触发dotenv
# LeftBreakoutPredictor将在测试函数中导入


class TestLeftBreakoutPredictor:
    """LeftBreakoutPredictor测试类"""
    
    @pytest.fixture
    def mock_left_model(self):
        """创建模拟左侧模型"""
        mock_model = Mock()
        
        # 模拟特征工程器
        mock_feature_engineer = Mock()
        mock_feature_engineer.extract_features.return_value = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'name': ['平安银行'],
            'feature1': [0.5],
            'feature2': [0.3],
        })
        
        mock_model.feature_engineer = mock_feature_engineer
        
        # 模拟模型（XGBoost）
        mock_xgb_model = Mock()
        mock_xgb_model.predict_proba.return_value = np.array([[0.3, 0.7]])  # 70%概率为正样本
        
        mock_model.model = mock_xgb_model
        mock_model.feature_columns = ['feature1', 'feature2']
        
        # 模拟数据管理器
        mock_model.dm = Mock()
        mock_model.dm.get_stock_list.return_value = pd.DataFrame({
            'ts_code': ['000001.SZ', '600000.SH'],
            'name': ['平安银行', '浦发银行'],
        })
        
        return mock_model
    
    @pytest.fixture
    def predictor(self, mock_left_model):
        """创建预测器实例"""
        # 在fixture中导入，此时conftest的mock已生效
        from src.models.stock_selection.left_breakout.left_predictor import LeftBreakoutPredictor
        return LeftBreakoutPredictor(mock_left_model)
    
    def test_init(self, predictor, mock_left_model):
        """测试初始化"""
        assert predictor.model == mock_left_model
        assert predictor.feature_engineer == mock_left_model.feature_engineer
    
    def test_get_market_stocks(self, predictor):
        """测试获取市场股票列表"""
        stocks = predictor._get_market_stocks()
        
        assert isinstance(stocks, pd.DataFrame)
        assert 'ts_code' in stocks.columns
        assert 'name' in stocks.columns
    
    def test_get_market_stocks_empty(self, predictor):
        """测试获取空股票列表"""
        # 模拟返回空列表
        predictor.model.dm.get_stock_list.return_value = pd.DataFrame()
        
        stocks = predictor._get_market_stocks()
        assert isinstance(stocks, pd.DataFrame)
    
    def test_extract_stock_features_structure(self, predictor):
        """测试提取股票特征结构"""
        result = predictor._extract_stock_features(
            ts_code='000001.SZ',
            name='平安银行',
            prediction_date='20240101'
        )
        
        assert isinstance(result, pd.DataFrame)
        # 即使没有数据，也应该返回DataFrame
    
    def test_extract_stock_features_no_data(self, predictor):
        """测试没有数据的情况"""
        # 模拟特征工程返回空DataFrame
        predictor.feature_engineer.extract_features.return_value = pd.DataFrame()
        
        result = predictor._extract_stock_features(
            ts_code='999999.SZ',
            name='测试股票',
            prediction_date='20240101'
        )
        
        assert isinstance(result, pd.DataFrame)
    
    def test_predict_current_market_structure(self, predictor):
        """测试预测当前市场结构"""
        result = predictor.predict_current_market(
            prediction_date='20240101',
            top_n=10,
            min_probability=0.1,
            max_stocks=5
        )
        
        assert isinstance(result, pd.DataFrame)
        # 即使没有预测结果，也应该返回DataFrame
    
    def test_predict_current_market_empty_stocks(self, predictor):
        """测试股票列表为空的情况"""
        # 模拟返回空股票列表
        predictor.model.dm.get_stock_list.return_value = pd.DataFrame()
        
        result = predictor.predict_current_market(
            prediction_date='20240101',
            top_n=10
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_predict_current_market_with_max_stocks(self, predictor):
        """测试限制最大股票数"""
        result = predictor.predict_current_market(
            prediction_date='20240101',
            top_n=10,
            max_stocks=1  # 只处理1只股票
        )
        
        assert isinstance(result, pd.DataFrame)
    
    def test_predict_current_market_no_features(self, predictor):
        """测试没有提取到特征的情况"""
        # 模拟所有股票特征提取都失败
        predictor.feature_engineer.extract_features.return_value = pd.DataFrame()
        
        result = predictor.predict_current_market(
            prediction_date='20240101',
            top_n=10
        )
        
        assert isinstance(result, pd.DataFrame)
        # 应该返回空DataFrame
    
    def test_predict_current_market_with_model(self, predictor):
        """测试使用模型进行预测"""
        # 确保模型存在
        predictor.model.model = Mock()
        predictor.model.model.predict_proba.return_value = np.array([[0.2, 0.8]])
        predictor.model.feature_columns = ['feature1', 'feature2']
        
        # 创建特征数据
        predictor.feature_engineer.extract_features.return_value = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'name': ['平安银行'],
            'feature1': [0.5],
            'feature2': [0.3],
        })
        
        result = predictor.predict_current_market(
            prediction_date='20240101',
            top_n=10
        )
        
        assert isinstance(result, pd.DataFrame)
    
    def test_predict_current_market_no_model(self, predictor):
        """测试没有模型的情况"""
        # 模拟模型为None
        predictor.model.model = None
        
        result = predictor.predict_current_market(
            prediction_date='20240101',
            top_n=10
        )
        
        assert isinstance(result, pd.DataFrame)
        # 没有模型时应该返回空DataFrame或提示信息

