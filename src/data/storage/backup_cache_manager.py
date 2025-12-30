"""
备份缓存管理器 - 支持CSV备份
实现双层存储：SQLite（快速查询） + CSV（可读备份）
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from .cache_manager import CacheManager, SUPPORTED_DATA_TYPES
from src.utils.logger import log


class BackupCacheManager(CacheManager):
    """备份缓存管理器 - 支持CSV备份"""
    
    def __init__(self, db_path: str = None, backup_dir: str = None, enable_backup: bool = True):
        """
        初始化备份缓存管理器
        
        Args:
            db_path: 数据库路径
            backup_dir: CSV备份目录
            enable_backup: 是否启用CSV备份
        """
        super().__init__(db_path)
        
        # CSV备份目录
        if backup_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            backup_dir = project_root / 'data' / 'backup'
        
        self.backup_dir = Path(backup_dir)
        self.enable_backup = enable_backup
        
        if self.enable_backup:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"CSV备份目录: {self.backup_dir}")
    
    # ============================================================================
    # 重写基类方法，自动使用备份功能
    # ============================================================================
    
    def get_data(
        self,
        ts_code: str,
        data_type: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取股票数据（自动使用CSV备份，如果启用）
        
        Args:
            ts_code: 股票代码
            data_type: 数据类型
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            
        Returns:
            数据DataFrame，如果没有数据返回空DataFrame
        """
        if not self.enable_backup:
            # 如果未启用备份，直接调用基类方法
            return super().get_data(ts_code, data_type, start_date, end_date)
        
        # 使用备份功能
        result = self.get_data_with_backup(ts_code, data_type, start_date, end_date)
        return result if result is not None else pd.DataFrame()
    
    def save_data(
        self,
        data: pd.DataFrame,
        data_type: str,
        ts_code: str
    ):
        """
        保存股票数据（自动使用CSV备份，如果启用）
        
        Args:
            data: 数据DataFrame（必须包含 trade_date 列）
            data_type: 数据类型
            ts_code: 股票代码
        """
        # 先调用基类方法保存到SQLite
        super().save_data(data, data_type, ts_code)
        
        # 如果启用备份，同时保存到CSV
        if self.enable_backup:
            try:
                self.save_to_csv(data, ts_code, data_type)
            except Exception as e:
                log.warning(f"保存CSV失败: {e}")
    
    # ============================================================================
    # 显式备份方法（保留用于向后兼容和特殊场景）
    # ============================================================================
    
    def get_data_with_backup(
        self,
        ts_code: str,
        data_type: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        优先从CSV备份读取数据
        
        Args:
            ts_code: 股票代码
            data_type: 数据类型
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            数据DataFrame，如果没有返回None
        """
        if not self.enable_backup:
            # 如果未启用备份，直接从SQLite读取
            df = self.get_data(ts_code, data_type, start_date, end_date)
            return df if not df.empty else None
        
        # 1. 优先从CSV读取
        csv_data = self.read_from_csv(ts_code, data_type, start_date, end_date)
        if csv_data is not None and not csv_data.empty:
            return csv_data
        
        # 2. 从SQLite读取
        db_data = self.get_data(ts_code, data_type, start_date, end_date)
        if not db_data.empty:
            # 同时保存到CSV（补充CSV备份）
            try:
                self.save_to_csv(db_data, ts_code, data_type)
            except Exception as e:
                log.warning(f"保存CSV失败: {e}")
            return db_data
        
        # 3. 没有数据，返回None（由fetcher从网络获取）
        return None
    
    def save_data_with_backup(
        self,
        df: pd.DataFrame,
        data_type: str,
        ts_code: str
    ):
        """
        同时保存到SQLite和CSV
        
        Args:
            df: 数据DataFrame
            data_type: 数据类型
            ts_code: 股票代码
        """
        if df.empty:
            return
        
        # 保存到SQLite
        self.save_data(df, data_type, ts_code)
        
        # 保存到CSV（如果启用）
        if self.enable_backup:
            try:
                self.save_to_csv(df, ts_code, data_type)
            except Exception as e:
                log.warning(f"保存CSV失败: {e}")
    
    def save_to_csv(
        self,
        df: pd.DataFrame,
        ts_code: str,
        data_type: str
    ):
        """
        保存数据到CSV文件（增量更新）
        
        Args:
            df: 数据DataFrame
            ts_code: 股票代码
            data_type: 数据类型
        """
        if df.empty:
            return
        
        # 创建目录
        csv_dir = self.backup_dir / data_type
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        csv_file = csv_dir / f"{ts_code}.csv"
        
        # 如果文件已存在，合并数据（避免重复）
        if csv_file.exists():
            try:
                existing_df = pd.read_csv(csv_file)
                
                # 合并数据，去重
                df_merged = pd.concat([existing_df, df])
                
                # 根据主键去重
                if 'trade_date' in df_merged.columns:
                    df_merged = df_merged.drop_duplicates(
                        subset=['ts_code', 'trade_date'], keep='last'
                    ).sort_values('trade_date')
                
                df_merged.to_csv(csv_file, index=False, encoding='utf-8-sig')
                log.debug(f"✓ CSV已更新: {ts_code} {data_type} ({len(df_merged)}条)")
            except Exception as e:
                log.warning(f"合并CSV失败: {csv_file}, {e}")
                # 如果合并失败，直接覆盖
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        else:
            # 新文件，直接保存
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            log.info(f"✓ CSV已创建: {ts_code} {data_type} ({len(df)}条)")
    
    def read_from_csv(
        self,
        ts_code: str,
        data_type: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        从CSV文件读取数据
        
        Args:
            ts_code: 股票代码
            data_type: 数据类型
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            数据DataFrame，如果没有返回None
        """
        csv_file = self.backup_dir / data_type / f"{ts_code}.csv"
        
        if not csv_file.exists():
            return None
        
        try:
            df = pd.read_csv(csv_file)
            
            if df.empty or 'trade_date' not in df.columns:
                return None
            
            # 过滤日期范围
            df_filtered = df[
                (df['trade_date'] >= start_date) &
                (df['trade_date'] <= end_date)
            ].sort_values('trade_date')
            
            if not df_filtered.empty:
                log.info(f"✓ 从CSV读取: {ts_code} {data_type} ({len(df_filtered)}条)")
                return df_filtered
            
        except Exception as e:
            log.warning(f"读取CSV失败: {csv_file}, {e}")
            return None
        
        return None
    
    def export_all_to_csv(self, data_types: list = None):
        """
        导出所有SQLite数据到CSV
        
        Args:
            data_types: 要导出的数据类型列表，None表示全部
        """
        if data_types is None:
            data_types = SUPPORTED_DATA_TYPES
        
        log.info("="*80)
        log.info("开始导出SQLite数据到CSV")
        log.info("="*80)
        
        import sqlite3
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 获取所有股票代码
                cursor.execute("SELECT DISTINCT ts_code FROM stock_data_cache")
                stocks = [row[0] for row in cursor.fetchall()]
                
                log.info(f"找到 {len(stocks)} 只股票")
                
                total_exported = 0
                
                for idx, ts_code in enumerate(stocks, 1):
                    if idx % 100 == 0:
                        log.info(f"进度: {idx}/{len(stocks)} ({idx/len(stocks)*100:.1f}%)")
                    
                    for data_type in data_types:
                        # 使用 get_data 方法读取数据
                        # 获取一个很大的日期范围来获取所有数据
                        df = self.get_data(ts_code, data_type, '19900101', '20991231')
                        
                        if not df.empty:
                            self.save_to_csv(df, ts_code, data_type)
                            total_exported += 1
                
                log.info(f"\n✓ 导出完成！共导出 {total_exported} 个文件")
                
        except Exception as e:
            log.error(f"导出失败: {e}")
        
        # 创建备份索引
        self.create_backup_index()
    
    def import_from_csv(self, data_types: list = None):
        """
        从CSV导入数据到SQLite
        
        Args:
            data_types: 要导入的数据类型列表，None表示全部
        """
        if data_types is None:
            data_types = SUPPORTED_DATA_TYPES
        
        log.info("="*80)
        log.info("开始从CSV导入数据到SQLite")
        log.info("="*80)
        
        total_imported = 0
        
        for data_type in data_types:
            csv_dir = self.backup_dir / data_type
            
            if not csv_dir.exists():
                log.warning(f"目录不存在: {csv_dir}")
                continue
            
            csv_files = list(csv_dir.glob('*.csv'))
            log.info(f"\n{data_type}: 找到 {len(csv_files)} 个文件")
            
            for idx, csv_file in enumerate(csv_files, 1):
                ts_code = csv_file.stem
                
                try:
                    df = pd.read_csv(csv_file)
                    
                    if not df.empty:
                        self.save_data(df, data_type, ts_code)
                        total_imported += 1
                    
                    if idx % 100 == 0:
                        log.info(f"  进度: {idx}/{len(csv_files)}")
                
                except Exception as e:
                    log.error(f"导入失败: {csv_file}, {e}")
        
        log.success(f"\n✓ 导入完成！共导入 {total_imported} 个文件")
    
    def create_backup_index(self):
        """创建备份索引文件"""
        metadata_dir = self.backup_dir / 'metadata'
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        index = {
            'backup_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'backup_directory': str(self.backup_dir),
            'data_types': {},
            'total_stocks': 0,
            'total_files': 0,
        }
        
        total_stocks = set()
        total_files = 0
        
        for data_type in SUPPORTED_DATA_TYPES:
            csv_dir = self.backup_dir / data_type
            
            if csv_dir.exists():
                files = list(csv_dir.glob('*.csv'))
                stocks = [f.stem for f in files]
                
                index['data_types'][data_type] = {
                    'file_count': len(files),
                    'stocks': stocks,
                }
                
                total_stocks.update(stocks)
                total_files += len(files)
        
        index['total_stocks'] = len(total_stocks)
        index['total_files'] = total_files
        
        index_file = metadata_dir / 'backup_index.json'
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        
        log.success(f"✓ 备份索引已创建: {index_file}")
        log.info(f"  - 股票数: {index['total_stocks']}")
        log.info(f"  - 文件数: {index['total_files']}")
        
        return index
    
    def get_backup_stats(self) -> dict:
        """获取备份统计信息"""
        stats = {
            'sqlite': self.get_stats(),
            'csv': {},
        }
        
        if self.enable_backup:
            for data_type in SUPPORTED_DATA_TYPES:
                csv_dir = self.backup_dir / data_type
                
                if csv_dir.exists():
                    file_count = len(list(csv_dir.glob('*.csv')))
                    stats['csv'][data_type] = file_count
        
        return stats
    
    def clear_csv_backup(self, ts_code: str = None, data_type: str = None):
        """
        清除CSV备份
        
        Args:
            ts_code: 股票代码（None表示全部）
            data_type: 数据类型（None表示全部）
        """
        if not self.enable_backup:
            log.warning("CSV备份未启用")
            return
        
        if ts_code and data_type:
            # 清除指定股票的指定类型
            csv_file = self.backup_dir / data_type / f"{ts_code}.csv"
            if csv_file.exists():
                csv_file.unlink()
                log.info(f"已删除CSV: {csv_file}")
        
        elif ts_code:
            # 清除指定股票的所有类型
            for data_type in SUPPORTED_DATA_TYPES:
                csv_file = self.backup_dir / data_type / f"{ts_code}.csv"
                if csv_file.exists():
                    csv_file.unlink()
            log.info(f"已删除{ts_code}的所有CSV备份")
        
        else:
            # 清除所有CSV
            import shutil
            if self.backup_dir.exists():
                shutil.rmtree(self.backup_dir)
                self.backup_dir.mkdir(parents=True, exist_ok=True)
                log.info("已清除所有CSV备份")


if __name__ == '__main__':
    # 测试功能
    print("="*80)
    print("备份缓存管理器测试")
    print("="*80)
    
    cache = BackupCacheManager()
    
    # 获取统计
    stats = cache.get_backup_stats()
    print("\n当前状态:")
    print(f"SQLite: {stats['sqlite']}")
    print(f"CSV: {stats['csv']}")

