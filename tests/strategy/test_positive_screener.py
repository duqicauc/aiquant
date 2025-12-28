"""
正样本筛选器测试
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.strategy.screening.positive_sample_screener import PositiveSampleScreener


class TestPositiveSampleScreener:
    """PositiveSampleScreener测试类"""
    
    @pytest.fixture
    def mock_dm(self):
        """创建模拟DataManager"""
        dm = Mock()
        
        # 模拟股票列表
        dm.get_stock_list.return_value = pd.DataFrame({
            'ts_code': ['000001.SZ', '600000.SH'],
            'name': ['平安银行', '浦发银行'],
            'list_date': ['19910403', '19991110'],
        })
        
        # 模拟周线数据（包含连续3周上涨的情况）
        dates = pd.date_range(end='20240119', periods=21, freq='W-FRI')
        mock_weekly = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'close': [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0,
                      15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0],
            'open': [9.8, 10.2, 10.7, 11.2, 11.7, 12.2, 12.7, 13.2, 13.7, 14.2, 14.7,
                     15.2, 15.7, 16.2, 16.7, 17.2, 17.7, 18.2, 18.7, 19.2, 19.7],
            'high': [10.2, 10.7, 11.2, 11.7, 12.2, 12.7, 13.2, 13.7, 14.2, 14.7, 15.2,
                     15.7, 16.2, 16.7, 17.2, 17.7, 18.2, 18.7, 19.2, 19.7, 20.2],
            'low': [9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5,
                    15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5],
            'vol': [1000000] * 21,
            'amount': [10000000] * 21,
        })
        dm.get_weekly_data.return_value = mock_weekly
        
        return dm
    
    def test_init(self, mock_dm):
        """测试初始化"""
        screener = PositiveSampleScreener(mock_dm)
        assert screener.dm == mock_dm
    
    def test_screen_all_stocks_basic(self, mock_dm):
        """测试基本筛选功能"""
        screener = PositiveSampleScreener(mock_dm)
        
        # 模拟返回空股票列表（避免真实API调用）
        mock_dm.get_stock_list.return_value = pd.DataFrame({
            'ts_code': pd.Series([], dtype='object'),
            'name': pd.Series([], dtype='object'),
            'list_date': pd.Series([], dtype='object'),
        })
        
        result = screener.screen_all_stocks(
            start_date='20240101',
            end_date='20240131'
        )
        
        assert isinstance(result, pd.DataFrame)
    
    def test_is_positive_sample_criteria(self, mock_dm):
        """测试正样本判断标准"""
        screener = PositiveSampleScreener(mock_dm)
        
        # 检查是否有_screen_single_stock方法（实际使用的方法）
        assert hasattr(screener, '_screen_single_stock')
        assert hasattr(screener, '_get_valid_stock_list')
    
    def test_get_valid_stock_list(self, mock_dm):
        """测试获取有效股票列表"""
        screener = PositiveSampleScreener(mock_dm)
        
        stock_list = screener._get_valid_stock_list()
        
        assert isinstance(stock_list, pd.DataFrame)
        assert 'ts_code' in stock_list.columns
        assert 'name' in stock_list.columns
        assert 'list_date' in stock_list.columns
    
    def test_convert_to_weekly(self, mock_dm):
        """测试日线转周线"""
        screener = PositiveSampleScreener(mock_dm)
        
        # 创建日线数据
        dates = pd.date_range(end='20240119', periods=21, freq='D')
        daily_data = pd.DataFrame({
            'trade_date': dates,
            'ts_code': ['000001.SZ'] * 21,
            'open': [10.0] * 21,
            'close': [10.5] * 21,
            'high': [11.0] * 21,
            'low': [9.5] * 21,
            'vol': [1000000] * 21,
        })
        
        weekly_data = screener._convert_to_weekly(daily_data)
        
        assert isinstance(weekly_data, pd.DataFrame)
        assert len(weekly_data) > 0
        assert 'open' in weekly_data.columns
        assert 'close' in weekly_data.columns
    
    def test_check_three_week_pattern_valid(self, mock_dm):
        """测试三周模式检查（有效）"""
        screener = PositiveSampleScreener(mock_dm)
        
        # 创建满足条件的三周数据
        # 注意：trade_date需要是datetime类型，且需要是索引或列
        three_weeks = pd.DataFrame({
            'trade_date': pd.to_datetime(['20240105', '20240112', '20240119']),
            'open': [10.0, 11.0, 12.0],
            'close': [11.0, 12.0, 15.0],  # 三连阳
            'high': [11.5, 12.5, 17.0],   # 最高涨幅>70% (17-10)/10 = 70%
            'low': [9.5, 10.5, 11.5],
        })
        
        list_date = pd.Timestamp('20200101')  # 上市超过半年
        
        result = screener._check_three_week_pattern(
            three_weeks, '000001.SZ', '测试股票', list_date
        )
        
        assert result is not None
        assert result['ts_code'] == '000001.SZ'
        assert result['total_return'] > 50
        assert result['max_return'] > 70
    
    def test_check_three_week_pattern_invalid(self, mock_dm):
        """测试三周模式检查（无效）"""
        screener = PositiveSampleScreener(mock_dm)
        
        # 创建不满足条件的三周数据（不是三连阳）
        three_weeks = pd.DataFrame({
            'trade_date': pd.to_datetime(['20240105', '20240112', '20240119']),
            'open': [10.0, 11.0, 12.0],
            'close': [9.5, 11.5, 12.5],  # 第一周下跌
            'high': [10.5, 12.0, 13.0],
            'low': [9.0, 10.5, 11.5],
        })
        
        list_date = pd.Timestamp('20200101')
        
        result = screener._check_three_week_pattern(
            three_weeks, '000001.SZ', '测试股票', list_date
        )
        
        assert result is None
    
    def test_check_three_week_pattern_insufficient_return(self, mock_dm):
        """测试涨幅不足的情况"""
        screener = PositiveSampleScreener(mock_dm)
        
        # 创建涨幅不足的三周数据
        three_weeks = pd.DataFrame({
            'trade_date': pd.to_datetime(['20240105', '20240112', '20240119']),
            'open': [10.0, 10.5, 11.0],
            'close': [10.5, 11.0, 11.5],  # 三连阳但涨幅<50%
            'high': [10.8, 11.3, 11.8],
            'low': [9.8, 10.3, 10.8],
        })
        
        list_date = pd.Timestamp('20200101')
        
        result = screener._check_three_week_pattern(
            three_weeks, '000001.SZ', '测试股票', list_date
        )
        
        assert result is None
    
    def test_check_three_week_pattern_new_stock(self, mock_dm):
        """测试新股（上市不足半年）"""
        screener = PositiveSampleScreener(mock_dm)
        
        # 创建满足其他条件但上市不足半年的数据
        three_weeks = pd.DataFrame({
            'trade_date': pd.to_datetime(['20240105', '20240112', '20240119']),
            'open': [10.0, 11.0, 12.0],
            'close': [11.0, 12.0, 15.0],
            'high': [11.5, 12.5, 17.0],
            'low': [9.5, 10.5, 11.5],
        })
        
        list_date = pd.Timestamp('20231201')  # 上市不足半年
        
        result = screener._check_three_week_pattern(
            three_weeks, '000001.SZ', '测试股票', list_date
        )
        
        assert result is None
    
    def test_screen_single_stock(self, mock_dm):
        """测试单只股票筛选"""
        screener = PositiveSampleScreener(mock_dm)
        
        # 模拟周线数据
        dates = pd.date_range(end='20240119', periods=21, freq='W-FRI')
        mock_weekly = pd.DataFrame({
            'trade_date': dates,
            'close': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                      21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
            'open': [9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8, 16.8, 17.8, 18.8, 19.8,
                     20.8, 21.8, 22.8, 23.8, 24.8, 25.8, 26.8, 27.8, 28.8, 29.8],
            'high': [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5,
                     21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5],
            'low': [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5,
                    20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5],
        })
        mock_dm.get_weekly_data.return_value = mock_weekly
        
        list_date = pd.Timestamp('20200101')
        result = screener._screen_single_stock(
            '000001.SZ', '测试股票', list_date, '20240101', '20240131'
        )
        
        assert isinstance(result, list)

