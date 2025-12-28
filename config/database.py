"""
数据库配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DB_DIR = PROJECT_ROOT / 'data' / 'database'
os.makedirs(DB_DIR, exist_ok=True)


class DatabaseConfig:
    """数据库配置"""
    
    # 数据库类型
    DB_TYPE = os.getenv('DB_TYPE', 'sqlite')  # sqlite, mysql, postgresql
    
    # SQLite配置
    SQLITE_DB_PATH = DB_DIR / 'aiquant.db'
    
    # MySQL配置
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', '3306'))
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'aiquant')
    
    # PostgreSQL配置
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', '5432'))
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', '')
    POSTGRES_DATABASE = os.getenv('POSTGRES_DATABASE', 'aiquant')
    
    # Redis配置（缓存）
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_DB = int(os.getenv('REDIS_DB', '0'))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
    
    @classmethod
    def get_database_url(cls):
        """获取数据库连接URL"""
        if cls.DB_TYPE == 'sqlite':
            return f'sqlite:///{cls.SQLITE_DB_PATH}'
        elif cls.DB_TYPE == 'mysql':
            return f'mysql+pymysql://{cls.MYSQL_USER}:{cls.MYSQL_PASSWORD}@{cls.MYSQL_HOST}:{cls.MYSQL_PORT}/{cls.MYSQL_DATABASE}'
        elif cls.DB_TYPE == 'postgresql':
            return f'postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DATABASE}'
        else:
            raise ValueError(f"不支持的数据库类型: {cls.DB_TYPE}")


# 数据库配置实例
db_config = DatabaseConfig()

