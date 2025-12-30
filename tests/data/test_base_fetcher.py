"""
BaseFetcher测试
"""
import pytest
from unittest.mock import Mock
from src.data.fetcher.base_fetcher import BaseFetcher


class TestBaseFetcher:
    """BaseFetcher测试类"""
    
    def test_format_stock_code_with_suffix(self):
        """测试已带后缀的股票代码"""
        # 创建一个具体的实现类用于测试
        class TestFetcher(BaseFetcher):
            def _init_connection(self):
                pass
            def get_stock_list(self, **kwargs):
                pass
            def get_daily_data(self, stock_code, start_date, end_date=None, adjust='qfq'):
                pass
            def get_minute_data(self, stock_code, freq='5min', start_date=None, end_date=None):
                pass
            def get_fundamental_data(self, stock_code, start_date=None, end_date=None):
                pass
            def get_daily_basic(self, stock_code, start_date, end_date=None):
                pass
        
        fetcher = TestFetcher('test')
        assert fetcher.format_stock_code('000001.SZ') == '000001.SZ'
        assert fetcher.format_stock_code('600000.SH') == '600000.SH'
        assert fetcher.format_stock_code('000001.sz') == '000001.SZ'  # 转大写
    
    def test_format_stock_code_without_suffix(self):
        """测试不带后缀的股票代码"""
        class TestFetcher(BaseFetcher):
            def _init_connection(self):
                pass
            def get_stock_list(self, **kwargs):
                pass
            def get_daily_data(self, stock_code, start_date, end_date=None, adjust='qfq'):
                pass
            def get_minute_data(self, stock_code, freq='5min', start_date=None, end_date=None):
                pass
            def get_fundamental_data(self, stock_code, start_date=None, end_date=None):
                pass
            def get_daily_basic(self, stock_code, start_date, end_date=None):
                pass
        
        fetcher = TestFetcher('test')
        assert fetcher.format_stock_code('000001') == '000001.SZ'
        assert fetcher.format_stock_code('600000') == '600000.SH'
        assert fetcher.format_stock_code('300001') == '300001.SZ'
        assert fetcher.format_stock_code('430001') == '430001.BJ'
        assert fetcher.format_stock_code('830001') == '830001.BJ'
        assert fetcher.format_stock_code('930001') == '930001.BJ'
    
    def test_format_stock_code_invalid(self):
        """测试无效股票代码"""
        class TestFetcher(BaseFetcher):
            def _init_connection(self):
                pass
            def get_stock_list(self, **kwargs):
                pass
            def get_daily_data(self, stock_code, start_date, end_date=None, adjust='qfq'):
                pass
            def get_minute_data(self, stock_code, freq='5min', start_date=None, end_date=None):
                pass
            def get_fundamental_data(self, stock_code, start_date=None, end_date=None):
                pass
            def get_daily_basic(self, stock_code, start_date, end_date=None):
                pass
        
        fetcher = TestFetcher('test')
        # 使用一个真正无法识别的代码（不以6, 0, 3, 4, 8, 9开头）
        with pytest.raises(ValueError, match="无法识别的股票代码"):
            fetcher.format_stock_code('123456')
    
    def test_format_date(self):
        """测试日期格式化"""
        class TestFetcher(BaseFetcher):
            def _init_connection(self):
                pass
            def get_stock_list(self, **kwargs):
                pass
            def get_daily_data(self, stock_code, start_date, end_date=None, adjust='qfq'):
                pass
            def get_minute_data(self, stock_code, freq='5min', start_date=None, end_date=None):
                pass
            def get_fundamental_data(self, stock_code, start_date=None, end_date=None):
                pass
            def get_daily_basic(self, stock_code, start_date, end_date=None):
                pass
        
        fetcher = TestFetcher('test')
        assert fetcher.format_date('2024-01-01') == '20240101'
        assert fetcher.format_date('2024/01/01') == '20240101'
        assert fetcher.format_date('20240101') == '20240101'
    
    def test_init(self):
        """测试初始化"""
        class TestFetcher(BaseFetcher):
            def _init_connection(self):
                self.connected = True
            def get_stock_list(self, **kwargs):
                pass
            def get_daily_data(self, stock_code, start_date, end_date=None, adjust='qfq'):
                pass
            def get_minute_data(self, stock_code, freq='5min', start_date=None, end_date=None):
                pass
            def get_fundamental_data(self, stock_code, start_date=None, end_date=None):
                pass
            def get_daily_basic(self, stock_code, start_date, end_date=None):
                pass
        
        fetcher = TestFetcher('test_source')
        assert fetcher.source_name == 'test_source'
        assert fetcher.connected == True

