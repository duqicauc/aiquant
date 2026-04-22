"""
全局配置文件

职责：
- 定义项目路径常量
- 加载环境变量
- 提供全局配置类

注意：模型相关配置请使用 config/settings.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# =========================================================================
# 路径常量
# =========================================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 主要目录
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"

# 数据子目录
RAW_DATA_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = DATA_DIR / "models"
TRAINING_DIR = DATA_DIR / "training"
PREDICTION_DIR = DATA_DIR / "prediction"
BACKUP_DIR = DATA_DIR / "backup"

# 训练相关子目录
TRAINING_SAMPLES_DIR = TRAINING_DIR / "samples"
TRAINING_FEATURES_DIR = TRAINING_DIR / "features"
TRAINING_METRICS_DIR = TRAINING_DIR / "metrics"
TRAINING_CHARTS_DIR = TRAINING_DIR / "charts"

# 预测相关子目录
PREDICTION_RESULTS_DIR = PREDICTION_DIR / "results"
PREDICTION_METADATA_DIR = PREDICTION_DIR / "metadata"
PREDICTION_ANALYSIS_DIR = PREDICTION_DIR / "analysis"

# 确保关键目录存在
for dir_path in [LOG_DIR, CACHE_DIR, MODELS_DIR, TRAINING_DIR, PREDICTION_DIR]:
    os.makedirs(dir_path, exist_ok=True)


# =========================================================================
# 全局配置类
# =========================================================================


class GlobalConfig:
    """全局配置"""

    # 项目信息
    PROJECT_NAME = "AIQuant"
    VERSION = "3.0.0"

    # 路径（为了兼容旧代码）
    PROJECT_ROOT = PROJECT_ROOT
    DATA_DIR = DATA_DIR
    LOG_DIR = LOG_DIR
    RAW_DATA_DIR = RAW_DATA_DIR
    PROCESSED_DATA_DIR = TRAINING_DIR  # 兼容旧代码
    MODEL_DIR = MODELS_DIR
    BACKTEST_DIR = DATA_DIR / "backtest"

    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}"

    # 数据更新配置
    AUTO_UPDATE = os.getenv("AUTO_UPDATE", "True").lower() == "true"
    UPDATE_TIME = os.getenv("UPDATE_TIME", "17:00")

    # 回测配置
    INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000000"))
    COMMISSION = float(os.getenv("COMMISSION", "0.0003"))
    SLIPPAGE = float(os.getenv("SLIPPAGE", "0.001"))

    # 性能配置
    N_JOBS = int(os.getenv("N_JOBS", "-1"))
    CACHE_SIZE = int(os.getenv("CACHE_SIZE", "1000"))


# 全局配置实例
config = GlobalConfig()


# =========================================================================
# 路径工具函数
# =========================================================================


def get_model_path(model_name: str, version: str = None) -> Path:
    """
    获取模型目录路径

    Args:
        model_name: 模型名称
        version: 版本号，None 表示获取模型根目录

    Returns:
        模型路径
    """
    if version:
        return MODELS_DIR / model_name / "versions" / version
    return MODELS_DIR / model_name


def get_training_path(subdir: str = None) -> Path:
    """
    获取训练数据目录路径

    Args:
        subdir: 子目录名（samples, features, metrics, charts）

    Returns:
        训练数据路径
    """
    if subdir:
        return TRAINING_DIR / subdir
    return TRAINING_DIR


def get_prediction_path(subdir: str = None) -> Path:
    """
    获取预测数据目录路径

    Args:
        subdir: 子目录名（results, metadata, analysis）

    Returns:
        预测数据路径
    """
    if subdir:
        return PREDICTION_DIR / subdir
    return PREDICTION_DIR


# =========================================================================
# 测试
# =========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("路径配置测试")
    print("=" * 60)

    print(f"\n📁 项目根目录: {PROJECT_ROOT}")
    print(f"📁 数据目录: {DATA_DIR}")
    print(f"📁 模型目录: {MODELS_DIR}")
    print(f"📁 日志目录: {LOG_DIR}")

    print("\n📦 模型路径示例:")
    print(f"  breakout_launch_scorer: {get_model_path('breakout_launch_scorer')}")
    print(f"  v1.4.0: {get_model_path('breakout_launch_scorer', 'v1.4.0')}")

    print("\n✅ 路径配置测试完成")
