"""
增强缓存管理器测试
"""
import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.data.storage.backup_cache_manager import BackupCacheManager


@pytest.mark.slow
class TestBackupCacheManager:
    """BackupCacheManager测试类"""
    
    @pytest.fixture
    def cache_manager(self, temp_dir):
        """创建BackupCacheManager实例"""
        db_path = temp_dir / 'test_backup_cache.db'
        backup_dir = temp_dir / 'test_backup'
        if db_path.exists():
            db_path.unlink()
        
        return BackupCacheManager(db_path=str(db_path), backup_dir=str(backup_dir), enable_backup=True)
    
    def test_init(self, temp_dir):
        """测试初始化"""
        db_path = temp_dir / 'test_backup_init.db'
        backup_dir = temp_dir / 'test_backup_init'
        cm = BackupCacheManager(db_path=str(db_path), backup_dir=str(backup_dir))
        
        assert cm.db_path == db_path
        assert cm.conn is not None
        assert cm.backup_dir == backup_dir
        assert cm.enable_backup == True
        assert db_path.exists()
        assert backup_dir.exists()
    
    def test_init_without_backup(self, temp_dir):
        """测试不启用备份的初始化"""
        db_path = temp_dir / 'test_no_backup.db'
        cm = BackupCacheManager(db_path=str(db_path), enable_backup=False)
        
        assert cm.enable_backup == False
    
    def test_save_data_with_backup(self, cache_manager):
        """测试带备份的数据保存"""
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
        
        cache_manager.save_data_with_backup(test_data, 'daily_data', '000001.SZ')
        
        # 验证SQLite中有数据
        result = cache_manager.get_data('000001.SZ', 'daily_data', '20240101', '20240101')
        assert len(result) == 1
        
        # 验证CSV备份中有数据
        csv_data = cache_manager.read_from_csv('000001.SZ', 'daily_data', '20240101', '20240101')
        assert csv_data is not None
        assert len(csv_data) == 1
    
    def test_get_data_with_backup(self, cache_manager):
        """测试带备份的数据获取"""
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
        cache_manager.save_data_with_backup(test_data, 'daily_data', '000001.SZ')
        
        # 从备份获取数据
        result = cache_manager.get_data_with_backup('000001.SZ', 'daily_data', '20240101', '20240101')
        assert result is not None
        assert len(result) == 1
    
    def test_save_to_csv(self, cache_manager):
        """测试保存到CSV"""
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240101'],
            'close': [10.2],
        })
        
        cache_manager.save_to_csv(test_data, '000001.SZ', 'daily_data')
        
        # 验证CSV文件存在
        csv_file = cache_manager.backup_dir / 'daily_data' / '000001.SZ.csv'
        assert csv_file.exists()
    
    def test_read_from_csv(self, cache_manager):
        """测试从CSV读取"""
        # 先保存到CSV
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240101'],
            'close': [10.2],
        })
        cache_manager.save_to_csv(test_data, '000001.SZ', 'daily_data')
        
        # 从CSV读取
        result = cache_manager.read_from_csv('000001.SZ', 'daily_data', '20240101', '20240101')
        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]['close'] == 10.2
    
    def test_read_from_csv_not_exists(self, cache_manager):
        """测试读取不存在的CSV"""
        result = cache_manager.read_from_csv('999999.SZ', 'daily_data', '20240101', '20240101')
        assert result is None
    
    def test_get_backup_stats(self, cache_manager):
        """测试获取备份统计"""
        # 先保存一些数据
        test_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20240101'],
            'close': [10.2],
        })
        cache_manager.save_to_csv(test_data, '000001.SZ', 'daily_data')
        
        stats = cache_manager.get_backup_stats()
        assert isinstance(stats, dict)
        assert 'sqlite' in stats
        assert 'csv' in stats
        assert 'daily_data' in stats['csv']
        assert stats['csv']['daily_data'] >= 1  # 至少1个文件

