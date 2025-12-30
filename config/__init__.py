"""
AIQuant 配置模块

使用方法：
    # 全局配置
    from config import config, settings
    from config import PROJECT_ROOT, MODELS_DIR, LOG_DIR
    
    # 获取全局配置
    top_n = settings.get('prediction.scoring.top_n')
    
    # 获取模型配置
    from config import get_model_config
    model_config = get_model_config('breakout_launch_scorer')
    
    # 路径工具
    from config import get_model_path, get_training_path
    model_dir = get_model_path('breakout_launch_scorer', 'v1.4.0')
"""

# 路径常量
from .config import (
    PROJECT_ROOT,
    CONFIG_DIR,
    DATA_DIR,
    LOG_DIR,
    SCRIPTS_DIR,
    SRC_DIR,
    TESTS_DIR,
    DOCS_DIR,
    RAW_DATA_DIR,
    CACHE_DIR,
    MODELS_DIR,
    TRAINING_DIR,
    PREDICTION_DIR,
    BACKUP_DIR,
    TRAINING_SAMPLES_DIR,
    TRAINING_FEATURES_DIR,
    TRAINING_METRICS_DIR,
    TRAINING_CHARTS_DIR,
    PREDICTION_RESULTS_DIR,
    PREDICTION_METADATA_DIR,
    PREDICTION_ANALYSIS_DIR,
)

# 全局配置类
from .config import config, GlobalConfig

# 路径工具函数
from .config import get_model_path, get_training_path, get_prediction_path

# 统一配置管理
from .settings import settings, Settings, get_model_config, get_setting

__all__ = [
    # 路径常量
    'PROJECT_ROOT',
    'CONFIG_DIR',
    'DATA_DIR',
    'LOG_DIR',
    'SCRIPTS_DIR',
    'SRC_DIR',
    'TESTS_DIR',
    'DOCS_DIR',
    'RAW_DATA_DIR',
    'CACHE_DIR',
    'MODELS_DIR',
    'TRAINING_DIR',
    'PREDICTION_DIR',
    'BACKUP_DIR',
    'TRAINING_SAMPLES_DIR',
    'TRAINING_FEATURES_DIR',
    'TRAINING_METRICS_DIR',
    'TRAINING_CHARTS_DIR',
    'PREDICTION_RESULTS_DIR',
    'PREDICTION_METADATA_DIR',
    'PREDICTION_ANALYSIS_DIR',
    # 配置类
    'config',
    'GlobalConfig',
    'settings',
    'Settings',
    # 便捷函数
    'get_model_path',
    'get_training_path',
    'get_prediction_path',
    'get_model_config',
    'get_setting',
]

