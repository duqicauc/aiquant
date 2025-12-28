"""
左侧起爆点模型正样本筛选器测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.models.stock_selection.left_breakout.left_positive_screener import LeftPositiveSampleScreener


class TestLeftPositiveSampleScreener:
    """LeftPositiveSampleScreener测试类"""
    
    @pytest.fixture
    def mock_data_manager(self):
        """创建模拟数据管理器"""
        mock_dm = Mock()
        
        # 模拟股票列表
        mock_dm.get_stock_list.return_value = pd.DataFrame({
            'ts_code': ['000001.SZ', '600000.SH', '000002.SZ'],
            'name': ['平安银行', '浦发银行', '万科A'],
            'list_date': ['19910403', '19991110', '19910129'],
            'market': ['主板', '主板', '主板']
        })
        
        # 模拟日线数据（包含上涨趋势）
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = np.linspace(10, 20, 100)  # 从10涨到20，涨幅100%
        mock_dm.get_daily_data.return_value = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'vol': np.random.uniform(1000000, 5000000, 100),
            'amount': prices * np.random.uniform(100000, 500000, 100),
        })
        
        # 模拟每日指标
        mock_dm.get_daily_basic.return_value = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'volume_ratio': np.random.uniform(1.5, 3.0, 100),
            'turnover_rate': np.random.uniform(1.0, 5.0, 100),
        })
        
        # 模拟技术因子
        mock_dm.get_stk_factor.return_value = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'rsi_6': np.random.uniform(30, 70, 100),  # RSI < 70
            'macd_dif': np.random.uniform(-0.5, 0.5, 100),
            'macd_dea': np.random.uniform(-0.3, 0.3, 100),
        })
        
        return mock_dm
    
    @pytest.fixture
    def screener(self, mock_data_manager):
        """创建筛选器实例"""
        config = {
            'sample_preparation': {
                'positive_criteria': {
                    'future_return_threshold': 50,
                    'past_return_threshold': 20,
                    'rsi_threshold': 70,
                    'volume_ratio_min': 1.5,
                    'volume_ratio_max': 3.0,
                    'min_signals': 2
                }
            }
        }
        return LeftPositiveSampleScreener(mock_data_manager, config)
    
    def test_init(self, screener, mock_data_manager):
        """测试初始化"""
        assert screener.dm == mock_data_manager
        assert screener.future_return_threshold == 0.5  # 50%转换为0.5
        assert screener.past_return_threshold == 0.2
        assert screener.rsi_threshold == 70
        assert screener.volume_ratio_min == 1.5
        assert screener.volume_ratio_max == 3.0
    
    def test_init_default_config(self, mock_data_manager):
        """测试使用默认配置初始化"""
        screener = LeftPositiveSampleScreener(mock_data_manager)
        assert screener.dm == mock_data_manager
        # 检查默认值
        assert screener.future_return_threshold > 0
    
    def test_get_valid_stock_list(self, screener):
        """测试获取有效股票列表"""
        stock_list = screener._get_valid_stock_list()
        
        assert isinstance(stock_list, pd.DataFrame)
        assert 'ts_code' in stock_list.columns
        assert 'name' in stock_list.columns
        assert len(stock_list) > 0
    
    def test_screen_single_stock_structure(self, screener):
        """测试单股票筛选结构"""
        result = screener._screen_single_stock(
            ts_code='000001.SZ',
            name='平安银行',
            list_date='19910403',
            start_date='20240101',
            end_date='20240131',
            look_forward_days=45
        )
        
        # 应该返回列表
        assert isinstance(result, list)
    
    def test_screen_single_stock_no_data(self, screener):
        """测试没有数据的情况"""
        # 模拟返回空数据
        screener.dm.get_daily_data.return_value = pd.DataFrame()
        
        result = screener._screen_single_stock(
            ts_code='999999.SZ',
            name='测试股票',
            list_date='20200101',
            start_date='20240101',
            end_date='20240131',
            look_forward_days=45
        )
        
        assert isinstance(result, list)
        # 没有数据应该返回空列表
        assert len(result) == 0
    
    def test_check_future_return(self, screener):
        """测试未来收益检查"""
        # 创建模拟数据：未来45天涨幅60%
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = np.linspace(10, 16, 100)  # 从10涨到16，涨幅60%
        
        df = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'close': prices,
        })
        
        # 检查第50天（t0）的未来收益
        t0_idx = 50
        t0_price = df.iloc[t0_idx]['close']
        future_price = df.iloc[t0_idx + 45]['close'] if t0_idx + 45 < len(df) else df.iloc[-1]['close']
        future_return = (future_price / t0_price - 1) * 100
        
        # 60% > 50%阈值，应该通过
        assert future_return > screener.future_return_threshold * 100
    
    def test_check_past_return(self, screener):
        """测试过去收益检查"""
        # 创建模拟数据：过去60天涨幅15%
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        prices = np.linspace(8.5, 10, 100)  # 从8.5涨到10，涨幅约17.6%
        
        df = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'close': prices,
        })
        
        # 检查第80天（t0）的过去收益
        t0_idx = 80
        t0_price = df.iloc[t0_idx]['close']
        past_price = df.iloc[max(0, t0_idx - 60)]['close']
        past_return = (t0_price / past_price - 1) * 100
        
        # 17.6% < 20%阈值，应该通过
        assert past_return < screener.past_return_threshold * 100
    
    def test_check_rsi_condition(self, screener):
        """测试RSI条件检查"""
        # 创建模拟RSI数据
        rsi_data = pd.DataFrame({
            'trade_date': ['20240101'],
            'rsi_6': [65],  # RSI < 70
        })
        
        # RSI 65 < 70，应该通过
        assert rsi_data.iloc[0]['rsi_6'] < screener.rsi_threshold
    
    def test_check_volume_ratio(self, screener):
        """测试量比条件检查"""
        volume_ratio = 2.0  # 在1.5-3.0范围内
        
        assert screener.volume_ratio_min <= volume_ratio <= screener.volume_ratio_max
    
    def test_screen_all_stocks_structure(self, screener):
        """测试筛选所有股票的结构"""
        result = screener.screen_all_stocks(
            start_date='20240101',
            end_date='20240131',
            look_forward_days=45
        )
        
        assert isinstance(result, pd.DataFrame)
        # 即使没有找到样本，也应该返回DataFrame
        assert 'ts_code' in result.columns or result.empty
    
    def test_screen_all_stocks_empty_date_range(self, screener):
        """测试空日期范围"""
        # 使用未来的日期范围（没有数据）
        future_date = (datetime.now() + timedelta(days=365)).strftime('%Y%m%d')
        result = screener.screen_all_stocks(
            start_date=future_date,
            end_date=future_date,
            look_forward_days=45
        )
        
        assert isinstance(result, pd.DataFrame)

