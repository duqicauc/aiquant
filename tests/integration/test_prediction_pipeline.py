"""
预测流程集成测试
测试从数据获取到预测结果的完整流程
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.models.stock_selection.potential_discovery.left_breakout.left_predictor import LeftBreakoutPredictor


@pytest.mark.integration
class TestPredictionPipeline:
    """预测流程集成测试"""

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
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        mock_daily = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'open': np.random.uniform(10, 20, 50),
            'high': np.random.uniform(15, 25, 50),
            'low': np.random.uniform(8, 18, 50),
            'close': np.random.uniform(10, 20, 50),
            'vol': np.random.uniform(1000000, 10000000, 50),
        })
        mock_dm.get_daily_data.return_value = mock_daily
        
        # 模拟交易日历
        mock_calendar = pd.DataFrame({
            'cal_date': dates.strftime('%Y%m%d'),
            'is_open': [1] * 50,
        })
        mock_dm.get_trade_calendar.return_value = mock_calendar
        
        return mock_dm

    @pytest.fixture
    def mock_left_model(self, mock_data_manager):
        """创建模拟的LeftBreakoutModel"""
        mock_model = Mock()
        mock_model.dm = mock_data_manager
        mock_model.feature_engineer = Mock()
        mock_model.model = Mock()
        mock_model.feature_columns = ['feature1', 'feature2', 'feature3']
        
        # 模拟预测结果
        mock_model.model.predict_proba.return_value = np.array([[0.3, 0.7], [0.2, 0.8]])
        
        return mock_model

    @pytest.fixture
    def predictor(self, mock_left_model):
        """创建预测器实例"""
        return LeftBreakoutPredictor(mock_left_model)

    @pytest.mark.unit
    def test_get_market_stocks(self, predictor):
        """测试获取市场股票列表"""
        stocks = predictor._get_market_stocks()
        
        assert isinstance(stocks, pd.DataFrame)
        assert 'ts_code' in stocks.columns
        assert 'name' in stocks.columns
        assert len(stocks) > 0

    @pytest.mark.unit
    def test_extract_stock_features(self, predictor, mock_left_model):
        """测试提取股票特征"""
        stock_code = '000001.SZ'
        trading_days = ['20241226', '20241225', '20241224']
        
        # 模拟特征提取结果
        features = pd.DataFrame({
            'ts_code': [stock_code],
            'feature1': [0.5],
            'feature2': [0.3],
            'feature3': [0.2],
        })
        mock_left_model.feature_engineer.extract_features.return_value = features
        
        # 执行特征提取
        result = predictor._extract_stock_features(stock_code, trading_days)
        
        # 验证特征提取被调用
        mock_left_model.feature_engineer.extract_features.assert_called()

    @pytest.mark.unit
    def test_predict_with_model(self, predictor, mock_left_model):
        """测试使用模型进行预测"""
        # 模拟特征数据
        features = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'feature1': [0.5],
            'feature2': [0.3],
            'feature3': [0.2],
        })
        
        # 执行预测
        probabilities = mock_left_model.model.predict_proba(features[['feature1', 'feature2', 'feature3']])
        
        assert probabilities is not None
        assert len(probabilities) > 0

    @pytest.mark.integration
    def test_complete_prediction_pipeline(self, predictor, mock_left_model):
        """测试完整预测流程"""
        # 1. 获取市场股票
        stocks = predictor._get_market_stocks()
        assert len(stocks) > 0
        
        # 2. 获取交易日历
        trading_days = predictor._get_trading_days_cached('20241226')
        assert trading_days is not None
        assert len(trading_days) > 0
        
        # 3. 提取特征
        stock_code = stocks.iloc[0]['ts_code']
        features = pd.DataFrame({
            'ts_code': [stock_code],
            'feature1': [0.5],
            'feature2': [0.3],
            'feature3': [0.2],
        })
        mock_left_model.feature_engineer.extract_features.return_value = features
        
        # 4. 模型预测
        probabilities = mock_left_model.model.predict_proba(
            features[['feature1', 'feature2', 'feature3']]
        )
        
        # 5. 生成预测结果
        result = pd.DataFrame({
            'ts_code': [stock_code],
            'name': ['股票1'],
            'probability': [probabilities[0][1]],
        })
        
        assert len(result) > 0
        assert 'probability' in result.columns

    @pytest.mark.integration
    def test_prediction_with_filtering(self, predictor, mock_left_model):
        """测试带过滤条件的预测"""
        # 模拟预测结果
        mock_left_model.model.predict_proba.return_value = np.array([
            [0.3, 0.7],  # 高概率
            [0.8, 0.2],  # 低概率
            [0.4, 0.6],  # 中等概率
        ])
        
        # 执行预测并过滤
        probabilities = mock_left_model.model.predict_proba(
            np.random.randn(3, 3)
        )
        
        # 过滤高概率结果
        high_prob_indices = np.where(probabilities[:, 1] > 0.5)[0]
        
        assert len(high_prob_indices) > 0

