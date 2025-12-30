"""
数据管理器测试
"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime, timedelta
from src.data.data_manager import DataManager


class TestDataManager:
    """DataManager测试类"""
    
    def test_init_default_source(self):
        """测试默认数据源初始化"""
        with patch('src.data.data_manager.data_source_config') as mock_config:
            mock_config.DEFAULT_SOURCE = 'tushare'
            with patch('src.data.data_manager.TushareFetcher'):
                dm = DataManager()
                assert dm.source == 'tushare'
    
    def test_init_custom_source(self):
        """测试自定义数据源初始化"""
        with patch('src.data.data_manager.TushareFetcher'):
            dm = DataManager(source='tushare')
            assert dm.source == 'tushare'
    
    def test_init_unsupported_source(self):
        """测试不支持的数据源"""
        with pytest.raises(ValueError, match="不支持的数据源"):
            DataManager(source='unsupported')
    
    def test_get_stock_list(self, mock_data_manager):
        """测试获取股票列表"""
        result = mock_data_manager.get_stock_list()
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'ts_code' in result.columns
    
    def test_get_daily_data(self, mock_data_manager):
        """测试获取日线数据"""
        result = mock_data_manager.get_daily_data(
            stock_code='000001.SZ',
            start_date='20240101',
            end_date='20240131'
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_get_fundamental_data(self, mock_data_manager):
        """测试获取基本面数据"""
        result = mock_data_manager.get_fundamental_data('000001.SZ')
        assert isinstance(result, pd.DataFrame)
    
    def test_get_complete_data(self, mock_data_manager):
        """测试获取完整数据"""
        # 模拟get_daily_data和get_daily_basic
        mock_data_manager.get_daily_data.return_value = pd.DataFrame({
            'trade_date': ['20240101'],
            'close': [10.0],
        })
        mock_data_manager.get_daily_basic.return_value = pd.DataFrame({
            'trade_date': ['20240101'],
            'volume_ratio': [1.2],
        })
        
        result = mock_data_manager.get_complete_data(
            stock_code='000001.SZ',
            start_date='20240101',
            end_date='20240101'
        )
        assert isinstance(result, pd.DataFrame)
    
    def test_get_weekly_data(self, mock_data_manager):
        """测试获取周线数据"""
        mock_data_manager.get_weekly_data.return_value = pd.DataFrame({
            'trade_date': ['20240105'],
            'close': [10.5],
        })
        
        result = mock_data_manager.get_weekly_data(
            stock_code='000001.SZ',
            start_date='20240101',
            end_date='20240105'
        )
        assert isinstance(result, pd.DataFrame)
    
    def test_get_daily_basic(self, mock_data_manager):
        """测试获取每日指标"""
        mock_data_manager.get_daily_basic.return_value = pd.DataFrame({
            'trade_date': ['20240101'],
            'volume_ratio': [1.2],
        })
        
        result = mock_data_manager.get_daily_basic(
            stock_code='000001.SZ',
            start_date='20240101',
            end_date='20240101'
        )
        assert isinstance(result, pd.DataFrame)
    
    def test_get_stk_factor(self, mock_data_manager):
        """测试获取技术因子"""
        mock_data_manager.get_stk_factor.return_value = pd.DataFrame({
            'trade_date': ['20240101'],
            'macd': [0.5],
        })
        
        result = mock_data_manager.get_stk_factor(
            stock_code='000001.SZ',
            start_date='20240101',
            end_date='20240101'
        )
        assert isinstance(result, pd.DataFrame)
    
    def test_get_trade_calendar(self, mock_data_manager):
        """测试获取交易日历"""
        mock_data_manager.get_trade_calendar.return_value = pd.DataFrame({
            'cal_date': ['20240101'],
            'is_open': [1],
        })
        
        result = mock_data_manager.get_trade_calendar(
            start_date='20240101',
            end_date='20240131'
        )
        assert isinstance(result, pd.DataFrame)
    
    def test_batch_get_daily_data(self, mock_data_manager):
        """测试批量获取日线数据"""
        mock_data_manager.get_daily_data.return_value = pd.DataFrame({
            'trade_date': ['20240101'],
            'close': [10.0],
        })
        
        stock_codes = ['000001.SZ', '600000.SH']
        result = mock_data_manager.batch_get_daily_data(
            stock_codes=stock_codes,
            start_date='20240101',
            end_date='20240101'
        )
        
        assert isinstance(result, dict)
        assert len(result) == 2
        assert '000001.SZ' in result
        assert '600000.SH' in result
    
    def test_batch_get_daily_basic(self, mock_data_manager):
        """测试批量获取每日指标"""
        mock_data_manager.batch_get_daily_basic.return_value = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240101'],
            'volume_ratio': [1.2],
        })
        
        result = mock_data_manager.batch_get_daily_basic('20240101')
        assert isinstance(result, pd.DataFrame)
    
    def test_get_minute_data(self, mock_data_manager):
        """测试获取分钟数据"""
        mock_data_manager.fetcher.get_minute_data.return_value = pd.DataFrame({
            'trade_time': ['20240101 09:30:00'],
            'close': [10.0],
        })
        
        result = mock_data_manager.get_minute_data('000001.SZ', freq='5min')
        assert isinstance(result, pd.DataFrame)
    
    def test_get_cyq_perf(self, mock_data_manager):
        """测试获取筹码数据"""
        mock_data_manager.fetcher.get_cyq_perf.return_value = pd.DataFrame({
            'trade_date': ['20240101'],
            'cyq_ratio': [0.5],
        })
        
        result = mock_data_manager.get_cyq_perf('000001.SZ', '20240101')
        assert isinstance(result, pd.DataFrame)
    
    def test_get_complete_data_empty_daily(self):
        """测试获取完整数据 - 日线数据为空"""
        with patch('src.data.data_manager.TushareFetcher') as mock_fetcher_class:
            mock_fetcher = Mock()
            mock_fetcher.get_daily_data.return_value = pd.DataFrame()
            mock_fetcher_class.return_value = mock_fetcher
            
            dm = DataManager(source='tushare')
            result = dm.get_complete_data('000001.SZ', '20240101', '20240131')
            assert isinstance(result, pd.DataFrame)
            assert result.empty
    
    def test_get_complete_data_daily_basic_empty(self):
        """测试获取完整数据 - 每日指标为空"""
        with patch('src.data.data_manager.TushareFetcher') as mock_fetcher_class:
            mock_fetcher = Mock()
            mock_fetcher.get_daily_data.return_value = pd.DataFrame({
                'trade_date': ['20240101'],
                'close': [10.0],
            })
            mock_fetcher.get_daily_basic.return_value = pd.DataFrame()
            mock_fetcher_class.return_value = mock_fetcher
            
            dm = DataManager(source='tushare')
            result = dm.get_complete_data('000001.SZ', '20240101', '20240131')
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert 'close' in result.columns
    
    def test_get_complete_data_daily_basic_exception(self):
        """测试获取完整数据 - 每日指标获取异常"""
        with patch('src.data.data_manager.TushareFetcher') as mock_fetcher_class:
            mock_fetcher = Mock()
            mock_fetcher.get_daily_data.return_value = pd.DataFrame({
                'trade_date': ['20240101'],
                'close': [10.0],
            })
            mock_fetcher.get_daily_basic.side_effect = Exception("API错误")
            mock_fetcher_class.return_value = mock_fetcher
            
            dm = DataManager(source='tushare')
            result = dm.get_complete_data('000001.SZ', '20240101', '20240131')
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
    
    def test_get_complete_data_daily_data_exception(self):
        """测试获取完整数据 - 日线数据获取异常"""
        with patch('src.data.data_manager.TushareFetcher') as mock_fetcher_class:
            mock_fetcher = Mock()
            mock_fetcher.get_daily_data.side_effect = Exception("API错误")
            mock_fetcher_class.return_value = mock_fetcher
            
            dm = DataManager(source='tushare')
            result = dm.get_complete_data('000001.SZ', '20240101', '20240131')
            assert isinstance(result, pd.DataFrame)
            assert result.empty
    
    def test_get_complete_data_with_merge(self):
        """测试获取完整数据 - 成功合并日线和每日指标"""
        with patch('src.data.data_manager.TushareFetcher') as mock_fetcher_class:
            mock_fetcher = Mock()
            mock_fetcher.get_daily_data.return_value = pd.DataFrame({
                'trade_date': ['20240101'],
                'close': [10.0],
                'open': [9.8],
            })
            mock_fetcher.get_daily_basic.return_value = pd.DataFrame({
                'trade_date': ['20240101'],
                'volume_ratio': [1.2],
                'total_mv': [1000000.0],
                'circ_mv': [800000.0],
                'pe': [15.5],
                'pb': [1.8],
            })
            mock_fetcher_class.return_value = mock_fetcher
            
            dm = DataManager(source='tushare')
            result = dm.get_complete_data('000001.SZ', '20240101', '20240131')
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert 'close' in result.columns
            assert 'volume_ratio' in result.columns
    
    def test_batch_get_daily_data_with_error(self):
        """测试批量获取日线数据 - 部分股票失败"""
        with patch('src.data.data_manager.TushareFetcher') as mock_fetcher_class:
            mock_fetcher = Mock()
            
            def side_effect(code, *args, **kwargs):
                if code == '000001.SZ':
                    return pd.DataFrame({'trade_date': ['20240101'], 'close': [10.0]})
                else:
                    raise Exception("API错误")
            
            mock_fetcher.get_daily_data.side_effect = side_effect
            mock_fetcher_class.return_value = mock_fetcher
            
            dm = DataManager(source='tushare')
            result = dm.batch_get_daily_data(['000001.SZ', '600000.SH'], '20240101', '20240131')
            
            assert isinstance(result, dict)
            assert len(result) == 2
            assert '000001.SZ' in result
            assert '600000.SH' in result
            assert not result['000001.SZ'].empty
            assert result['600000.SH'].empty
    
    def test_batch_get_daily_data_large_batch(self):
        """测试批量获取日线数据 - 大批量（测试进度日志）"""
        with patch('src.data.data_manager.TushareFetcher') as mock_fetcher_class:
            mock_fetcher = Mock()
            mock_fetcher.get_daily_data.return_value = pd.DataFrame({
                'trade_date': ['20240101'],
                'close': [10.0],
            })
            mock_fetcher_class.return_value = mock_fetcher
            
            dm = DataManager(source='tushare')
            # 创建100只股票，测试进度日志
            stock_codes = [f'{i:06d}.SZ' for i in range(1, 101)]
            result = dm.batch_get_daily_data(stock_codes, '20240101', '20240131')
            
            assert isinstance(result, dict)
            assert len(result) == 100
    
    def test_init_rqdata_source(self):
        """测试初始化RQData数据源"""
        with pytest.raises(NotImplementedError, match="RQData支持即将推出"):
            DataManager(source='rqdata')
    
    def test_init_jqdata_source(self):
        """测试初始化JQData数据源"""
        with pytest.raises(NotImplementedError, match="JQData支持即将推出"):
            DataManager(source='jqdata')


@pytest.mark.api
@pytest.mark.slow
class TestDataManagerRealAPI:
    """DataManager真实API测试"""
    
    @pytest.fixture(scope="class")
    def dm(self):
        """创建DataManager实例"""
        try:
            from config.data_source import data_source_config
            data_source_config.validate_tushare()
        except Exception as e:
            pytest.skip(f"Tushare配置无效: {e}")
        
        return DataManager(source='tushare')
    
    @pytest.mark.api
    def test_get_stock_list_real(self, dm):
        """测试真实获取股票列表"""
        df = dm.get_stock_list()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'ts_code' in df.columns
    
    @pytest.mark.api
    def test_get_daily_data_real(self, dm):
        """测试真实获取日线数据"""
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        df = dm.get_daily_data('000001.SZ', start_date, end_date)
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert 'trade_date' in df.columns
            assert 'close' in df.columns
    
    @pytest.mark.api
    def test_get_complete_data_real(self, dm):
        """测试真实获取完整数据"""
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
        
        df = dm.get_complete_data('000001.SZ', start_date, end_date)
        assert isinstance(df, pd.DataFrame)
    
    @pytest.mark.api
    def test_batch_get_daily_data_real(self, dm):
        """测试真实批量获取数据"""
        stock_codes = ['000001.SZ', '600000.SH']
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
        
        result = dm.batch_get_daily_data(stock_codes, start_date, end_date)
        assert isinstance(result, dict)
        assert len(result) == 2
