"""
缓存管理器 - SQLite 本地缓存

使用 SQLite 数据库缓存 API 数据，避免重复请求，提高效率。
支持：数据存储、过期检查、自动清理等功能。
"""
import os
import sqlite3
import pandas as pd
import json
import hashlib
from io import StringIO
from datetime import datetime, timedelta
from typing import Optional, Any
from pathlib import Path

from src.utils.logger import log
from config.settings import settings

# 支持的数据类型常量
SUPPORTED_DATA_TYPES = ['daily_data', 'weekly_data', 'daily_basic', 'stk_factor']


class CacheManager:
    """SQLite 缓存管理器"""
    
    def __init__(self, db_path: str = None):
        """
        初始化缓存管理器
        
        Args:
            db_path: 数据库路径，如果为None则使用配置文件中的路径
        """
        if db_path is None:
            db_path = settings.get('data_storage.cache.database', 'data/cache/quant_data.db')
        
        self.db_path = db_path
        self.expire_days = settings.get('data_storage.cache.expire_days', 7)
        
        # 确保目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
        log.info(f"CacheManager 初始化成功: {db_path}")
    
    def _init_database(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建通用缓存表（键值对存储）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    data_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    hit_count INTEGER DEFAULT 0
                )
            ''')
            
            # 创建股票数据表（按股票代码和数据类型存储，支持日期范围查询）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_data_cache (
                    ts_code TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (ts_code, data_type, trade_date)
                )
            ''')
            
            # 创建索引
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_cache_expires 
                ON cache(expires_at)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_cache_type 
                ON cache(data_type)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_stock_data_code_type 
                ON stock_data_cache(ts_code, data_type)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_stock_data_date 
                ON stock_data_cache(trade_date)
            ''')
            
            conn.commit()
    
    def _generate_key(self, data_type: str, **params) -> str:
        """
        生成缓存键
        
        Args:
            data_type: 数据类型（如 'daily', 'weekly', 'stock_list'）
            **params: 查询参数
            
        Returns:
            缓存键字符串
        """
        # 对参数排序后生成哈希
        params_str = json.dumps(params, sort_keys=True)
        hash_str = hashlib.md5(params_str.encode()).hexdigest()[:12]
        return f"{data_type}_{hash_str}"
    
    def get(
        self,
        data_type: str,
        **params
    ) -> Optional[pd.DataFrame]:
        """
        从缓存获取数据
        
        Args:
            data_type: 数据类型
            **params: 查询参数
            
        Returns:
            缓存的DataFrame，如果不存在或已过期返回None
        """
        cache_key = self._generate_key(data_type, **params)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 查询缓存
                cursor.execute('''
                    SELECT data, expires_at FROM cache 
                    WHERE cache_key = ? AND (expires_at IS NULL OR expires_at > ?)
                ''', (cache_key, datetime.now().isoformat()))
                
                row = cursor.fetchone()
                
                if row:
                    # 更新命中次数
                    cursor.execute('''
                        UPDATE cache SET hit_count = hit_count + 1 
                        WHERE cache_key = ?
                    ''', (cache_key,))
                    conn.commit()
                    
                    # 解析数据
                    data_json = row[0]
                    df = pd.read_json(StringIO(data_json), orient='records')
                    
                    # 恢复日期类型
                    for col in df.columns:
                        if 'date' in col.lower():
                            try:
                                # 如果是整数类型（如19910403），需要先转字符串再解析
                                if df[col].dtype in ['int64', 'float64']:
                                    df[col] = pd.to_datetime(df[col].astype(str), format='%Y%m%d', errors='coerce')
                                else:
                                    df[col] = pd.to_datetime(df[col], errors='coerce')
                            except:
                                pass
                    
                    return df
                
                return None
                
        except Exception as e:
            log.warning(f"缓存读取失败: {e}")
            return None
    
    def set(
        self,
        data_type: str,
        data: pd.DataFrame,
        expire_days: int = None,
        **params
    ):
        """
        存储数据到缓存
        
        Args:
            data_type: 数据类型
            data: 要缓存的DataFrame
            expire_days: 过期天数（None则使用默认值）
            **params: 查询参数
        """
        if data is None or data.empty:
            return
        
        cache_key = self._generate_key(data_type, **params)
        
        if expire_days is None:
            expire_days = self.expire_days
        
        expires_at = (datetime.now() + timedelta(days=expire_days)).isoformat()
        
        try:
            # 转换为JSON（处理日期类型）
            df_copy = data.copy()
            for col in df_copy.columns:
                if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            data_json = df_copy.to_json(orient='records')
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 插入或更新
                cursor.execute('''
                    INSERT OR REPLACE INTO cache 
                    (cache_key, data_type, data, created_at, expires_at, hit_count)
                    VALUES (?, ?, ?, ?, ?, 0)
                ''', (cache_key, data_type, data_json, 
                      datetime.now().isoformat(), expires_at))
                
                conn.commit()
                
        except Exception as e:
            log.warning(f"缓存写入失败: {e}")
    
    def invalidate(self, data_type: str = None, **params):
        """
        使缓存失效
        
        Args:
            data_type: 数据类型（如果为None则清除所有）
            **params: 查询参数（如果提供则只清除匹配的缓存）
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if data_type and params:
                    cache_key = self._generate_key(data_type, **params)
                    cursor.execute('DELETE FROM cache WHERE cache_key = ?', (cache_key,))
                elif data_type:
                    cursor.execute('DELETE FROM cache WHERE data_type = ?', (data_type,))
                else:
                    cursor.execute('DELETE FROM cache')
                
                deleted = cursor.rowcount
                conn.commit()
                
                log.info(f"清除缓存: {deleted} 条记录")
                
        except Exception as e:
            log.warning(f"清除缓存失败: {e}")
    
    def cleanup_expired(self):
        """清理过期缓存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    DELETE FROM cache WHERE expires_at < ?
                ''', (datetime.now().isoformat(),))
                
                deleted = cursor.rowcount
                conn.commit()
                
                if deleted > 0:
                    log.info(f"清理过期缓存: {deleted} 条记录")
                    
        except Exception as e:
            log.warning(f"清理过期缓存失败: {e}")
    
    def get_stats(self) -> dict:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 总记录数
                cursor.execute('SELECT COUNT(*) FROM cache')
                total_count = cursor.fetchone()[0]
                
                # 按类型统计
                cursor.execute('''
                    SELECT data_type, COUNT(*), SUM(hit_count) 
                    FROM cache GROUP BY data_type
                ''')
                type_stats = {}
                for row in cursor.fetchall():
                    type_stats[row[0]] = {
                        'count': row[1],
                        'hits': row[2] or 0
                    }
                
                # 数据库大小
                db_size = os.path.getsize(self.db_path) / (1024 * 1024)  # MB
                
                return {
                    'total_records': total_count,
                    'by_type': type_stats,
                    'database_size_mb': round(db_size, 2)
                }
                
        except Exception as e:
            log.warning(f"获取缓存统计失败: {e}")
            return {}
    
    # ============================================================================
    # 股票数据专用方法（按股票代码和数据类型存储，支持日期范围查询）
    # ============================================================================
    
    def get_data(
        self,
        ts_code: str,
        data_type: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取股票数据（按股票代码和日期范围）
        
        Args:
            ts_code: 股票代码
            data_type: 数据类型（如 'daily_data', 'weekly_data', 'daily_basic', 'stk_factor'）
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            
        Returns:
            数据DataFrame，如果没有数据返回空DataFrame
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 查询指定股票、类型和日期范围的数据
                query = '''
                    SELECT data_json FROM stock_data_cache
                    WHERE ts_code = ? AND data_type = ?
                    AND trade_date >= ? AND trade_date <= ?
                    ORDER BY trade_date
                '''
                
                cursor = conn.cursor()
                cursor.execute(query, (ts_code, data_type, start_date, end_date))
                rows = cursor.fetchall()
                
                if not rows:
                    return pd.DataFrame()
                
                # 合并所有数据
                all_data = []
                for row in rows:
                    data_json = row[0]
                    df_part = pd.read_json(StringIO(data_json), orient='records')
                    all_data.append(df_part)
                
                if all_data:
                    df = pd.concat(all_data, ignore_index=True)
                    # 去重（按trade_date）
                    if 'trade_date' in df.columns:
                        df = df.drop_duplicates(subset=['trade_date'], keep='last')
                        df = df.sort_values('trade_date').reset_index(drop=True)
                    
                    # 恢复日期类型
                    for col in df.columns:
                        if 'date' in col.lower():
                            try:
                                # 如果是整数类型（如19910403），需要先转字符串再解析
                                if df[col].dtype in ['int64', 'float64']:
                                    df[col] = pd.to_datetime(df[col].astype(str), format='%Y%m%d', errors='coerce')
                                else:
                                    df[col] = pd.to_datetime(df[col], errors='coerce')
                            except:
                                pass
                    
                    return df
                
                return pd.DataFrame()
                
        except Exception as e:
            log.warning(f"获取股票数据失败: {e}")
            return pd.DataFrame()
    
    def save_data(
        self,
        data: pd.DataFrame,
        data_type: str,
        ts_code: str
    ):
        """
        保存股票数据（按股票代码和数据类型）
        
        Args:
            data: 数据DataFrame（必须包含 trade_date 列）
            data_type: 数据类型
            ts_code: 股票代码
        """
        if data is None or data.empty:
            return
        
        if 'trade_date' not in data.columns:
            log.warning(f"数据缺少 trade_date 列，无法保存")
            return
        
        try:
            # 确保 ts_code 列存在
            if 'ts_code' not in data.columns:
                data = data.copy()
                data['ts_code'] = ts_code
            
            # 转换日期为字符串格式（YYYYMMDD）
            data_copy = data.copy()
            if pd.api.types.is_datetime64_any_dtype(data_copy['trade_date']):
                data_copy['trade_date'] = data_copy['trade_date'].dt.strftime('%Y%m%d')
            else:
                # 如果已经是字符串，确保格式正确
                data_copy['trade_date'] = pd.to_datetime(data_copy['trade_date']).dt.strftime('%Y%m%d')
            
            # 按日期分组，每个日期保存为一条记录
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 按日期分组
                grouped = data_copy.groupby('trade_date')
                
                for trade_date, group_df in grouped:
                    trade_date_str = str(trade_date)
                    
                    # 将分组数据转换为JSON
                    data_json = group_df.to_json(orient='records', date_format='iso')
                    
                    # 插入或更新
                    cursor.execute('''
                        INSERT OR REPLACE INTO stock_data_cache
                        (ts_code, data_type, trade_date, data_json, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (ts_code, data_type, trade_date_str, data_json, datetime.now().isoformat()))
                
                conn.commit()
                
        except Exception as e:
            log.warning(f"保存股票数据失败: {e}")
    
    def has_data(
        self,
        ts_code: str,
        data_type: str,
        start_date: str,
        end_date: str
    ) -> bool:
        """
        检查是否有指定日期范围的数据
        
        Args:
            ts_code: 股票代码
            data_type: 数据类型
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            
        Returns:
            如果缓存中有完整的数据范围返回True，否则返回False
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 查询缓存中的日期范围
                cursor.execute('''
                    SELECT MIN(trade_date) as min_date, MAX(trade_date) as max_date
                    FROM stock_data_cache
                    WHERE ts_code = ? AND data_type = ?
                ''', (ts_code, data_type))
                
                row = cursor.fetchone()
                
                if not row or row[0] is None:
                    return False
                
                cached_start = row[0]
                cached_end = row[1]
                
                # 检查缓存是否完全覆盖需要的日期范围
                return cached_start <= start_date and cached_end >= end_date
                
        except Exception as e:
            log.warning(f"检查数据存在性失败: {e}")
            return False
    
    def get_missing_dates(
        self,
        ts_code: str,
        data_type: str,
        start_date: str,
        end_date: str
    ) -> Optional[tuple]:
        """
        获取缺失的日期范围
        
        Args:
            ts_code: 股票代码
            data_type: 数据类型
            start_date: 需要的开始日期 (YYYYMMDD)
            end_date: 需要的结束日期 (YYYYMMDD)
            
        Returns:
            (missing_start, missing_end) 元组，如果没有缺失返回None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 查询缓存中的日期范围
                cursor.execute('''
                    SELECT MIN(trade_date) as min_date, MAX(trade_date) as max_date
                    FROM stock_data_cache
                    WHERE ts_code = ? AND data_type = ?
                ''', (ts_code, data_type))
                
                row = cursor.fetchone()
                
                if not row or row[0] is None:
                    # 完全没有数据，返回整个范围
                    return (start_date, end_date)
                
                cached_start = row[0]
                cached_end = row[1]
                
                # 检查缺失范围
                missing_start = None
                missing_end = None
                
                if cached_start > start_date:
                    # 前面有缺失
                    missing_start = start_date
                    missing_end = min(cached_start, end_date)
                elif cached_end < end_date:
                    # 后面有缺失
                    missing_start = max(cached_end, start_date)
                    missing_end = end_date
                
                if missing_start and missing_end:
                    return (missing_start, missing_end)
                
                return None
                
        except Exception as e:
            log.warning(f"获取缺失日期失败: {e}")
            return None
    
    def get_cache_stats(self) -> dict:
        """
        获取缓存统计信息（兼容方法，调用 get_stats）
        """
        return self.get_stats()
