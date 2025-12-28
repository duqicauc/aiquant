"""
全局配置文件
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
LOG_DIR = PROJECT_ROOT / 'logs'

# 确保目录存在
os.makedirs(LOG_DIR, exist_ok=True)


class GlobalConfig:
    """全局配置"""
    
    # 项目信息
    PROJECT_NAME = 'AIQuant'
    VERSION = '2.0.0'
    
    # 数据目录
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODEL_DIR = DATA_DIR / 'models'
    BACKTEST_DIR = DATA_DIR / 'backtest'
    
    # 日志配置
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}'
    
    # 数据更新配置
    AUTO_UPDATE = os.getenv('AUTO_UPDATE', 'True').lower() == 'true'
    UPDATE_TIME = os.getenv('UPDATE_TIME', '17:00')  # 每日更新时间
    
    # 回测配置
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '1000000'))  # 初始资金100万
    COMMISSION = float(os.getenv('COMMISSION', '0.0003'))  # 手续费0.03%
    SLIPPAGE = float(os.getenv('SLIPPAGE', '0.001'))  # 滑点0.1%
    
    # 性能配置
    N_JOBS = int(os.getenv('N_JOBS', '-1'))  # 并行任务数，-1表示使用所有CPU
    CACHE_SIZE = int(os.getenv('CACHE_SIZE', '1000'))  # 缓存大小（MB）


# 全局配置实例
config = GlobalConfig()

