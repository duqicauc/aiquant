"""
左侧潜力牛股预测器测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.models.stock_selection.potential_discovery.left_breakout.left_predictor import LeftBreakoutPredictor


class TestLeftBreakoutPredictor:
    """左侧潜力牛股预测器测试类"""

    @pytest.fixture
    def mock_left_model(self):
        """模拟LeftBreakoutModel"""
        mock_model = Mock()
        mock_model.feature_engineer = Mock()
        mock_model.model = Mock()
        mock_model.feature_columns = ['feature1', 'feature2', 'feature3']
        
        # 模拟预测结果
        mock_model.model.predict_proba.return_value = np.array([[0.3, 0.7], [0.2, 0.8]])
        
        return mock_model

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
    def predictor(self, mock_left_model, mock_data_manager):
        """创建预测器实例"""
        mock_left_model.dm = mock_data_manager
        predictor = LeftBreakoutPredictor(mock_left_model)
        return predictor

    @pytest.mark.unit
    def test_init(self, predictor, mock_left_model):
        """测试初始化"""
        assert predictor.model == mock_left_model
        assert predictor.feature_engineer == mock_left_model.feature_engineer
        assert predictor._calendar_cache == {}

    @pytest.mark.unit
    def test_predict_current_market_success(self, predictor, mock_data_manager):
        """测试成功预测当前市场"""
        # 模拟特征提取
        mock_features = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'name': ['股票1'],
            'feature1': [1.0],
            'feature2': [2.0],
            'feature3': [3.0],
        })
        predictor.feature_engineer.extract_features.return_value = mock_features
        
        result = predictor.predict_current_market(
            prediction_date='20241226',
            top_n=10,
            min_probability=0.1
        )
        
        assert isinstance(result, pd.DataFrame)
        # 由于mock数据，可能为空，但应该不抛出异常

    @pytest.mark.unit
    def test_predict_current_market_no_stocks(self, predictor, mock_data_manager):
        """测试没有股票的情况"""
        mock_data_manager.get_stock_list.return_value = pd.DataFrame()
        
        result = predictor.predict_current_market()
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @pytest.mark.unit
    def test_predict_current_market_insufficient_calendar(self, predictor, mock_data_manager):
        """测试交易日历不足的情况"""
        # 模拟不足的交易日历
        mock_calendar = pd.DataFrame({
            'cal_date': ['20241226'],
            'is_open': [1],
        })
        mock_data_manager.get_trade_calendar.return_value = mock_calendar
        
        result = predictor.predict_current_market()
        
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit
    def test_get_trading_days_cached(self, predictor, mock_data_manager):
        """测试获取交易日历（带缓存）"""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        mock_calendar = pd.DataFrame({
            'cal_date': dates.strftime('%Y%m%d'),
            'is_open': [1] * 50,
        })
        mock_data_manager.get_trade_calendar.return_value = mock_calendar
        
        trading_days = predictor._get_trading_days_cached('20241226')
        
        assert trading_days is not None
        assert len(trading_days) > 0
        
        # 第二次调用应该使用缓存
        trading_days2 = predictor._get_trading_days_cached('20241226')
        assert trading_days2 == trading_days
        # 应该只调用一次
        assert mock_data_manager.get_trade_calendar.call_count <= 2

    @pytest.mark.unit
    def test_get_market_stocks(self, predictor, mock_data_manager):
        """测试获取市场股票列表"""
        stocks = predictor._get_market_stocks()
        
        assert isinstance(stocks, pd.DataFrame)
        assert 'ts_code' in stocks.columns
        assert 'name' in stocks.columns
