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
from datetime import datetime, timedelta
from typing import Optional, Any
from pathlib import Path

from src.utils.logger import log
from config.settings import settings


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
            
            # 创建缓存表
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
            
            # 创建索引
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_cache_expires 
                ON cache(expires_at)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_cache_type 
                ON cache(data_type)
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
                    df = pd.read_json(data_json, orient='records')
                    
                    # 恢复日期类型
                    for col in df.columns:
                        if 'date' in col.lower():
                            try:
                                df[col] = pd.to_datetime(df[col])
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
