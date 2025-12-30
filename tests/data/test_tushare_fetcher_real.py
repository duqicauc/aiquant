"""
Tushare数据获取器真实数据测试
使用真实API调用进行测试
"""
import pytest
import os
import pandas as pd
from datetime import datetime, timedelta


@pytest.mark.api
@pytest.mark.slow
class TestTushareFetcherReal:
    """TushareFetcher真实数据测试类"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """检查是否有Tushare Token"""
        token = os.getenv('TUSHARE_TOKEN')
        if not token:
            pytest.skip("需要设置TUSHARE_TOKEN环境变量")
    
    def test_get_stock_list_real(self):
        """测试获取股票列表（真实数据）"""
        from src.data.data_manager import DataManager
        
        dm = DataManager(source='tushare')
        result = dm.get_stock_list()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'ts_code' in result.columns
        assert 'name' in result.columns
        assert 'list_date' in result.columns
    
    def test_get_daily_data_real(self):
        """测试获取日线数据（真实数据）"""
        from src.data.data_manager import DataManager
        
        dm = DataManager(source='tushare')
        
        # 获取最近30天的数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        result = dm.get_daily_data(
            stock_code='000001.SZ',
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d')
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'trade_date' in result.columns
        assert 'open' in result.columns
        assert 'close' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
    
    def test_get_daily_basic_real(self):
        """测试获取每日指标（真实数据）"""
        from src.data.data_manager import DataManager
        
        dm = DataManager(source='tushare')
        
        # 获取最近的数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        result = dm.get_daily_basic(
            stock_code='000001.SZ',
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d')
        )
        
        assert isinstance(result, pd.DataFrame)
        # 即使没有数据，也应该返回DataFrame
        assert isinstance(result, pd.DataFrame)
    
    def test_get_complete_data_real(self):
        """测试获取完整数据（真实数据）"""
        from src.data.data_manager import DataManager
        
        dm = DataManager(source='tushare')
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        result = dm.get_complete_data(
            stock_code='000001.SZ',
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d')
        )
        
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            # 验证包含价格数据
            assert 'close' in result.columns
            # 可能包含指标数据
            assert 'trade_date' in result.columns
    
    def test_get_trade_calendar_real(self):
        """测试获取交易日历（真实数据）"""
        from src.data.data_manager import DataManager
        
        dm = DataManager(source='tushare')
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        result = dm.get_trade_calendar(
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d')
        )
        
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert 'cal_date' in result.columns or 'trade_date' in result.columns
    
    def test_batch_get_daily_data_real(self):
        """测试批量获取日线数据（真实数据）"""
        from src.data.data_manager import DataManager
        
        dm = DataManager(source='tushare')
        
        # 只测试2只股票，避免API限制
        stock_codes = ['000001.SZ', '600000.SH']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        result = dm.batch_get_daily_data(
            stock_codes=stock_codes,
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d')
        )
        
        assert isinstance(result, dict)
        assert len(result) == len(stock_codes)
        for code in stock_codes:
            assert code in result
            assert isinstance(result[code], pd.DataFrame)

