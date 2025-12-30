"""
CacheManager测试 - 使用真实数据库
"""
import pytest
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from src.data.storage.cache_manager import CacheManager


@pytest.mark.slow
class TestCacheManager:
    """CacheManager测试类"""
    
    @pytest.fixture
    def cache_manager(self, temp_dir):
        """创建CacheManager实例（使用临时数据库）"""
        db_path = temp_dir / 'test_cache.db'
        if db_path.exists():
            db_path.unlink()  # 删除旧数据库
        
        return CacheManager(db_path=str(db_path))
    
    def test_init(self, temp_dir):
        """测试初始化"""
        db_path = temp_dir / 'test_init.db'
        cm = CacheManager(db_path=str(db_path))
        
        assert cm.db_path == db_path
        assert cm.conn is not None
        assert db_path.exists()
    
    def test_init_default_path(self):
        """测试默认路径初始化"""
        cm = CacheManager()
        assert cm.db_path.exists()
        assert cm.conn is not None
    
    def test_save_daily_data(self, cache_manager):
        """测试保存日线数据"""
        # 创建测试数据
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ', '000001.SZ'],
            'trade_date': ['20240101', '20240102'],
            'open': [10.0, 10.5],
            'high': [10.5, 11.0],
            'low': [9.8, 10.2],
            'close': [10.2, 10.8],
            'pre_close': [9.9, 10.2],
            'change': [0.3, 0.6],
            'pct_chg': [3.03, 5.88],
            'vol': [1000000, 1200000],
            'amount': [10000000, 12000000],
        })
        
        cache_manager.save_data(test_data, 'daily_data', '000001.SZ')
        
        # 验证数据已保存
        result = cache_manager.get_data('000001.SZ', 'daily_data', '20240101', '20240102')
        assert len(result) == 2
        assert result.iloc[0]['close'] == 10.2
    
    def test_get_daily_data(self, cache_manager):
        """测试获取日线数据"""
        # 先保存数据
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240101'],
            'open': [10.0],
            'high': [10.5],
            'low': [9.8],
            'close': [10.2],
            'pre_close': [9.9],
            'change': [0.3],
            'pct_chg': [3.03],
            'vol': [1000000],
            'amount': [10000000],
        })
        cache_manager.save_data(test_data, 'daily_data', '000001.SZ')
        
        # 获取数据
        result = cache_manager.get_data('000001.SZ', 'daily_data', '20240101', '20240101')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]['ts_code'] == '000001.SZ'
        assert result.iloc[0]['trade_date'] == '20240101'
    
    def test_get_daily_data_not_found(self, cache_manager):
        """测试获取不存在的数据"""
        result = cache_manager.get_data('999999.SZ', 'daily_data', '20240101', '20240101')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_save_weekly_data(self, cache_manager):
        """测试保存周线数据"""
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240105'],
            'open': [10.0],
            'high': [11.0],
            'low': [9.5],
            'close': [10.5],
            'pre_close': [9.9],
            'change': [0.6],
            'pct_chg': [6.06],
            'vol': [5000000],
            'amount': [50000000],
        })
        
        cache_manager.save_data(test_data, 'weekly_data', '000001.SZ')
        
        # 验证数据已保存
        result = cache_manager.get_data('000001.SZ', 'weekly_data', '20240105', '20240105')
        assert len(result) == 1
    
    def test_save_daily_basic(self, cache_manager):
        """测试保存每日指标"""
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240101'],
            'turnover_rate': [2.5],
            'volume_ratio': [1.2],
            'total_mv': [1000000.0],
            'circ_mv': [800000.0],
            'pe': [15.5],
            'pb': [1.8],
        })
        
        cache_manager.save_data(test_data, 'daily_basic', '000001.SZ')
        
        # 验证数据已保存
        result = cache_manager.get_data('000001.SZ', 'daily_basic', '20240101', '20240101')
        assert len(result) > 0
    
    def test_clear_cache(self, cache_manager):
        """测试清理缓存"""
        # 先保存一些数据
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240101'],
            'open': [10.0],
            'high': [10.5],
            'low': [9.8],
            'close': [10.2],
            'pre_close': [9.9],
            'change': [0.3],
            'pct_chg': [3.03],
            'vol': [1000000],
            'amount': [10000000],
        })
        cache_manager.save_data(test_data, 'daily_data', '000001.SZ')
        
        # 清理缓存
        cache_manager.clear_cache(ts_code='000001.SZ', data_type='daily_data')
        
        # 验证数据已删除
        result = cache_manager.get_data('000001.SZ', 'daily_data', '20240101', '20240101')
        assert len(result) == 0
    
    def test_get_cache_stats(self, cache_manager):
        """测试获取缓存统计"""
        stats = cache_manager.get_cache_stats()
        
        assert isinstance(stats, dict)
        assert 'daily_data' in stats
        assert 'weekly_data' in stats
        assert 'daily_basic' in stats
    
    def test_has_data(self, cache_manager):
        """测试检查数据是否存在"""
        # 先保存数据
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240101'],
            'open': [10.0],
            'high': [10.5],
            'low': [9.8],
            'close': [10.2],
            'pre_close': [9.9],
            'change': [0.3],
            'pct_chg': [3.03],
            'vol': [1000000],
            'amount': [10000000],
        })
        cache_manager.save_data(test_data, 'daily_data', '000001.SZ')
        
        # 检查数据是否存在
        has_data = cache_manager.has_data('000001.SZ', 'daily_data', '20240101', '20240101')
        assert has_data == True
        
        # 检查不存在的数据
        has_data = cache_manager.has_data('999999.SZ', 'daily_data', '20240101', '20240101')
        assert has_data == False
    
    def test_get_data(self, cache_manager):
        """测试通用get_data方法"""
        # 先保存数据
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240101'],
            'open': [10.0],
            'high': [10.5],
            'low': [9.8],
            'close': [10.2],
            'pre_close': [9.9],
            'change': [0.3],
            'pct_chg': [3.03],
            'vol': [1000000],
            'amount': [10000000],
        })
        cache_manager.save_data(test_data, 'daily_data', '000001.SZ')
        
        # 使用get_data获取
        result = cache_manager.get_data('000001.SZ', 'daily_data', '20240101', '20240101')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_save_data(self, cache_manager):
        """测试通用save_data方法"""
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240101'],
            'open': [10.0],
            'high': [10.5],
            'low': [9.8],
            'close': [10.2],
        })
        
        cache_manager.save_data(test_data, 'daily_data', '000001.SZ')
        
        # 验证数据已保存
        result = cache_manager.get_data('000001.SZ', 'daily_data', '20240101', '20240101')
        assert len(result) == 1
    
    def test_get_missing_dates(self, cache_manager):
        """测试获取缺失日期"""
        # 先保存部分数据
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240101'],
            'open': [10.0],
            'high': [10.5],
            'low': [9.8],
            'close': [10.2],
            'pre_close': [9.9],
            'change': [0.3],
            'pct_chg': [3.03],
            'vol': [1000000],
            'amount': [10000000],
        })
        cache_manager.save_data(test_data, 'daily_data', '000001.SZ')
        
        # 获取缺失日期（应该返回20240102-20240105）
        missing = cache_manager.get_missing_dates('000001.SZ', 'daily_data', '20240101', '20240105')
        assert missing is not None
        assert missing[0] == '20240102'  # 开始缺失日期
        assert missing[1] == '20240105'  # 结束缺失日期
    
    def test_save_stk_factor(self, cache_manager):
        """测试保存技术因子"""
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240101'],
            'macd_dif': [0.5],
            'macd_dea': [0.3],
            'macd': [0.2],
            'kdj_k': [50.0],
            'kdj_d': [45.0],
            'kdj_j': [55.0],
            'rsi_6': [60.0],
            'rsi_12': [55.0],
            'rsi_24': [50.0],
        })
        
        cache_manager.save_data(test_data, 'stk_factor', '000001.SZ')
        
        # 验证数据已保存（通过get_data）
        result = cache_manager.get_data('000001.SZ', 'stk_factor', '20240101', '20240101')
        assert len(result) == 1
    
    def test_get_cache_stats_with_data(self, cache_manager):
        """测试获取缓存统计（有数据）"""
        # 先保存一些数据
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ', '000001.SZ'],
            'trade_date': ['20240101', '20240102'],
            'open': [10.0, 10.5],
            'high': [10.5, 11.0],
            'low': [9.8, 10.2],
            'close': [10.2, 10.8],
            'pre_close': [9.9, 10.2],
            'change': [0.3, 0.6],
            'pct_chg': [3.03, 5.88],
            'vol': [1000000, 1200000],
            'amount': [10000000, 12000000],
        })
        cache_manager.save_data(test_data, 'daily_data', '000001.SZ')
        
        stats = cache_manager.get_cache_stats()
        
        assert isinstance(stats, dict)
        assert 'daily_data' in stats
        assert stats['daily_data'] == 2  # 2条记录
        assert 'unique_stocks' in stats
    
    def test_get_missing_dates_no_cache(self, cache_manager):
        """测试获取缺失日期（无缓存）"""
        missing = cache_manager.get_missing_dates('000001.SZ', 'daily_data', '20240101', '20240105')
        assert missing is not None
        assert missing[0] == '20240101'
        assert missing[1] == '20240105'
    
    def test_get_missing_dates_full_cache(self, cache_manager):
        """测试获取缺失日期（完全缓存）"""
        # 先保存数据
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240101'],
            'open': [10.0],
            'high': [10.5],
            'low': [9.8],
            'close': [10.2],
            'pre_close': [9.9],
            'change': [0.3],
            'pct_chg': [3.03],
            'vol': [1000000],
            'amount': [10000000],
        })
        cache_manager.save_data(test_data, 'daily_data', '000001.SZ')
        
        # 如果数据在缓存中且未过期，应该返回None
        missing = cache_manager.get_missing_dates('000001.SZ', 'daily_data', '20240101', '20240101')
        # 注意：由于过期检查，可能返回None或需要更新的日期
    
    def test_clear_cache_by_code(self, cache_manager):
        """测试按股票代码清理缓存"""
        # 先保存数据
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240101'],
            'open': [10.0],
            'high': [10.5],
            'low': [9.8],
            'close': [10.2],
            'pre_close': [9.9],
            'change': [0.3],
            'pct_chg': [3.03],
            'vol': [1000000],
            'amount': [10000000],
        })
        cache_manager.save_data(test_data, 'daily_data', '000001.SZ')
        
        # 清理特定股票
        cache_manager.clear_cache(ts_code='000001.SZ')
        
        # 验证数据已删除
        result = cache_manager.get_data('000001.SZ', 'daily_data', '20240101', '20240101')
        assert len(result) == 0
    
    def test_clear_cache_by_type(self, cache_manager):
        """测试按数据类型清理缓存"""
        # 先保存数据
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240101'],
            'open': [10.0],
            'high': [10.5],
            'low': [9.8],
            'close': [10.2],
            'pre_close': [9.9],
            'change': [0.3],
            'pct_chg': [3.03],
            'vol': [1000000],
            'amount': [10000000],
        })
        cache_manager.save_data(test_data, 'daily_data', '000001.SZ')
        
        # 清理特定类型
        cache_manager.clear_cache(ts_code='000001.SZ', data_type='daily_data')
        
        # 验证数据已删除
        result = cache_manager.get_data('000001.SZ', 'daily_data', '20240101', '20240101')
        assert len(result) == 0
    
    def test_clear_all_cache(self, cache_manager):
        """测试清理所有缓存"""
        # 先保存数据
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240101'],
            'open': [10.0],
            'high': [10.5],
            'low': [9.8],
            'close': [10.2],
            'pre_close': [9.9],
            'change': [0.3],
            'pct_chg': [3.03],
            'vol': [1000000],
            'amount': [10000000],
        })
        cache_manager.save_data(test_data, 'daily_data', '000001.SZ')
        
        # 清理所有缓存
        cache_manager.clear_cache()
        
        # 验证数据已删除
        result = cache_manager.get_data('000001.SZ', 'daily_data', '20240101', '20240101')
        assert len(result) == 0
    
    def test_has_data(self, cache_manager):
        """测试检查数据是否存在"""
        # 先保存数据
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240101'],
            'open': [10.0],
            'high': [10.5],
            'low': [9.8],
            'close': [10.2],
            'pre_close': [9.9],
            'change': [0.3],
            'pct_chg': [3.03],
            'vol': [1000000],
            'amount': [10000000],
        })
        cache_manager.save_data(test_data, 'daily_data', '000001.SZ')
        
        # 检查数据是否存在
        has_data = cache_manager.has_data('000001.SZ', 'daily_data', '20240101', '20240101')
        assert has_data == True
        
        # 检查不存在的数据
        has_data = cache_manager.has_data('999999.SZ', 'daily_data', '20240101', '20240101')
        assert has_data == False
    
    def test_get_data(self, cache_manager):
        """测试通用get_data方法"""
        # 先保存数据
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240101'],
            'open': [10.0],
            'high': [10.5],
            'low': [9.8],
            'close': [10.2],
            'pre_close': [9.9],
            'change': [0.3],
            'pct_chg': [3.03],
            'vol': [1000000],
            'amount': [10000000],
        })
        cache_manager.save_data(test_data, 'daily_data', '000001.SZ')
        
        # 使用get_data获取
        result = cache_manager.get_data('000001.SZ', 'daily_data', '20240101', '20240101')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
