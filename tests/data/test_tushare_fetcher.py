"""
TushareFetcher测试 - 使用真实API
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.data.fetcher.tushare_fetcher import TushareFetcher
from config.data_source import data_source_config


@pytest.mark.api
@pytest.mark.slow
class TestTushareFetcher:
    """TushareFetcher测试类 - 使用真实API"""
    
    @pytest.fixture(scope="class")
    def fetcher(self):
        """创建TushareFetcher实例"""
        # 验证配置
        try:
            data_source_config.validate_tushare()
        except Exception as e:
            pytest.skip(f"Tushare配置无效: {e}")
        
        return TushareFetcher(use_cache=True, points=5000)
    
    def test_init(self):
        """测试初始化"""
        fetcher = TushareFetcher(use_cache=False, points=120)
        assert fetcher.use_cache == False
        assert fetcher.points == 120
        assert fetcher.cache is None
    
    def test_init_with_cache(self):
        """测试带缓存的初始化"""
        fetcher = TushareFetcher(use_cache=True, points=2000)
        assert fetcher.use_cache == True
        assert fetcher.cache is not None
    
    @pytest.mark.api
    def test_get_stock_list(self, fetcher):
        """测试获取股票列表"""
        df = fetcher.get_stock_list()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'ts_code' in df.columns
        assert 'name' in df.columns
        assert 'list_date' in df.columns
        
        # 验证数据格式
        assert df['ts_code'].dtype == 'object'
        assert len(df['ts_code'].iloc[0]) > 0
    
    @pytest.mark.api
    def test_get_stock_list_with_exchange(self, fetcher):
        """测试按交易所获取股票列表"""
        # 上交所
        sse_df = fetcher.get_stock_list(exchange='SSE')
        assert isinstance(sse_df, pd.DataFrame)
        if len(sse_df) > 0:
            assert all('SH' in code for code in sse_df['ts_code'].head(10))
        
        # 深交所
        szse_df = fetcher.get_stock_list(exchange='SZSE')
        assert isinstance(szse_df, pd.DataFrame)
        if len(szse_df) > 0:
            assert all('SZ' in code for code in szse_df['ts_code'].head(10))
    
    @pytest.mark.api
    def test_get_daily_data(self, fetcher):
        """测试获取日线数据"""
        # 使用一个常见的股票代码
        stock_code = '000001.SZ'
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        df = fetcher.get_daily_data(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert 'trade_date' in df.columns
            assert 'open' in df.columns
            assert 'high' in df.columns
            assert 'low' in df.columns
            assert 'close' in df.columns
            assert 'vol' in df.columns
            
            # 验证数据合理性
            assert (df['high'] >= df['low']).all()
            assert (df['high'] >= df['open']).all()
            assert (df['high'] >= df['close']).all()
    
    @pytest.mark.api
    def test_get_weekly_data(self, fetcher):
        """测试获取周线数据"""
        stock_code = '000001.SZ'
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
        
        df = fetcher.get_weekly_data(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert 'trade_date' in df.columns
            assert 'close' in df.columns
    
    @pytest.mark.api
    def test_get_daily_basic(self, fetcher):
        """测试获取每日指标"""
        trade_date = (datetime.now() - timedelta(days=5)).strftime('%Y%m%d')
        
        df = fetcher.get_daily_basic(trade_date=trade_date)
        
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert 'ts_code' in df.columns
            assert 'trade_date' in df.columns
    
    @pytest.mark.api
    def test_get_trading_calendar(self, fetcher):
        """测试获取交易日历"""
        start_date = '20240101'
        end_date = '20240131'
        
        df = fetcher.get_trading_calendar(start_date=start_date, end_date=end_date)
        
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert 'cal_date' in df.columns
            assert 'is_open' in df.columns
    
    @pytest.mark.api
    def test_get_fundamental_data(self, fetcher):
        """测试获取基本面数据"""
        stock_code = '000001.SZ'
        
        df = fetcher.get_fundamental_data(stock_code)
        
        assert isinstance(df, pd.DataFrame)
        # 基本面数据可能为空，这是正常的
    
    def test_cache_functionality(self):
        """测试缓存功能"""
        fetcher = TushareFetcher(use_cache=True, points=120)
        
        # 第一次调用（会写入缓存）
        df1 = fetcher.get_stock_list()
        
        # 第二次调用（应该从缓存读取）
        df2 = fetcher.get_stock_list()
        
        # 验证数据一致性
        assert len(df1) == len(df2)
        assert df1.equals(df2)
    
    @pytest.mark.api
    def test_get_minute_data(self, fetcher):
        """测试获取分钟数据"""
        stock_code = '000001.SZ'
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y%m%d')
        
        df = fetcher.get_minute_data(
            stock_code=stock_code,
            freq='5min',
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(df, pd.DataFrame)
        # 分钟数据可能为空，这是正常的
    
    @pytest.mark.api
    def test_get_stk_factor(self, fetcher):
        """测试获取技术因子"""
        stock_code = '000001.SZ'
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        df = fetcher.get_stk_factor(stock_code, start_date, end_date)
        
        assert isinstance(df, pd.DataFrame)
    
    @pytest.mark.api
    def test_get_cyq_perf(self, fetcher):
        """测试获取筹码数据"""
        stock_code = '000001.SZ'
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        df = fetcher.get_cyq_perf(stock_code, start_date, end_date)
        
        assert isinstance(df, pd.DataFrame)
    
    @pytest.mark.api
    def test_batch_get_daily_basic(self, fetcher):
        """测试批量获取每日指标"""
        trade_date = (datetime.now() - timedelta(days=5)).strftime('%Y%m%d')
        
        df = fetcher.batch_get_daily_basic(trade_date=trade_date)
        
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert 'ts_code' in df.columns
            assert 'trade_date' in df.columns
    
    def test_format_stock_code(self, fetcher):
        """测试股票代码格式化"""
        assert fetcher.format_stock_code('000001') == '000001.SZ'
        assert fetcher.format_stock_code('600000') == '600000.SH'
        assert fetcher.format_stock_code('000001.SZ') == '000001.SZ'
    
    def test_format_date(self, fetcher):
        """测试日期格式化"""
        assert fetcher.format_date('2024-01-01') == '20240101'
        assert fetcher.format_date('2024/01/01') == '20240101'
        assert fetcher.format_date('20240101') == '20240101'
    
    @pytest.mark.api
    def test_get_st_list(self, fetcher):
        """测试获取ST股票列表"""
        df = fetcher.get_st_list()
        assert isinstance(df, pd.DataFrame)
    
    @pytest.mark.api
    def test_get_adj_factor(self, fetcher):
        """测试获取复权因子"""
        stock_code = '000001.SZ'
        df = fetcher.get_adj_factor(stock_code)
        assert isinstance(df, pd.DataFrame)
    
    @pytest.mark.api
    def test_get_daily_data_with_cache(self):
        """测试带缓存的日线数据获取"""
        fetcher = TushareFetcher(use_cache=True, points=120)
        
        stock_code = '000001.SZ'
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        # 第一次获取（写入缓存）
        df1 = fetcher.get_daily_data(stock_code, start_date, end_date)
        
        # 第二次获取（从缓存读取）
        df2 = fetcher.get_daily_data(stock_code, start_date, end_date)
        
        # 验证数据一致性
        if len(df1) > 0 and len(df2) > 0:
            assert len(df1) == len(df2)
    
    @pytest.mark.api
    def test_get_weekly_data_with_cache(self):
        """测试带缓存的周线数据获取"""
        fetcher = TushareFetcher(use_cache=True, points=120)
        
        stock_code = '000001.SZ'
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
        
        # 第一次获取
        df1 = fetcher.get_weekly_data(stock_code, start_date, end_date)
        
        # 第二次获取（从缓存）
        df2 = fetcher.get_weekly_data(stock_code, start_date, end_date)
        
        if len(df1) > 0 and len(df2) > 0:
            assert len(df1) == len(df2)
    
    @pytest.mark.api
    def test_get_daily_basic_with_cache(self):
        """测试带缓存的每日指标获取"""
        fetcher = TushareFetcher(use_cache=True, points=120)
        
        stock_code = '000001.SZ'
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        # 第一次获取
        df1 = fetcher.get_daily_basic(stock_code, start_date, end_date)
        
        # 第二次获取（从缓存）
        df2 = fetcher.get_daily_basic(stock_code, start_date, end_date)
        
        if len(df1) > 0 and len(df2) > 0:
            assert len(df1) == len(df2)
    
    @pytest.mark.api
    def test_get_daily_data_different_adjust(self, fetcher):
        """测试不同复权类型的日线数据"""
        stock_code = '000001.SZ'
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        # 测试前复权
        df_qfq = fetcher.get_daily_data(stock_code, start_date, end_date, adjust='qfq')
        assert isinstance(df_qfq, pd.DataFrame)
        
        # 测试后复权
        df_hfq = fetcher.get_daily_data(stock_code, start_date, end_date, adjust='hfq')
        assert isinstance(df_hfq, pd.DataFrame)
        
        # 测试不复权
        df_none = fetcher.get_daily_data(stock_code, start_date, end_date, adjust='none')
        assert isinstance(df_none, pd.DataFrame)
    
    @pytest.mark.api
    def test_get_adj_factor(self, fetcher):
        """测试获取复权因子"""
        stock_code = '000001.SZ'
        df = fetcher.get_adj_factor(stock_code)
        assert isinstance(df, pd.DataFrame)
    
    @pytest.mark.api
    def test_get_daily_data_empty_result(self, fetcher):
        """测试获取不存在股票的数据"""
        # 使用一个不存在的股票代码（如果可能）
        # 或者使用一个很久以前的日期
        result = fetcher.get_daily_data(
            stock_code='999999.SZ',
            start_date='20000101',
            end_date='20000131'
        )
        # 应该返回空DataFrame或抛出异常
        assert isinstance(result, pd.DataFrame)
    
    @pytest.mark.api
    def test_get_daily_data_incremental_cache(self):
        """测试增量缓存更新"""
        fetcher = TushareFetcher(use_cache=True, points=120)
        
        stock_code = '000001.SZ'
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y%m%d')
        
        # 第一次获取部分数据
        df1 = fetcher.get_daily_data(stock_code, start_date, end_date)
        
        # 第二次获取扩展范围的数据（应该使用增量更新）
        extended_start = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
        df2 = fetcher.get_daily_data(stock_code, extended_start, end_date)
        
        # 验证数据完整性
        if len(df1) > 0 and len(df2) > 0:
            assert len(df2) >= len(df1)

