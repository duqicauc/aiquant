"""
pytest配置文件
提供测试用的fixtures和工具函数
"""
import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================================
# 环境变量和dotenv Mock（必须在导入任何配置模块之前）
# ============================================================================
# 设置默认环境变量，避免读取.env文件
os.environ.setdefault('TUSHARE_TOKEN', 'test_token_for_testing')
os.environ.setdefault('LOG_LEVEL', 'INFO')
os.environ.setdefault('DEFAULT_DATA_SOURCE', 'tushare')

# Mock dotenv.load_dotenv，避免权限问题
# 必须在导入任何使用dotenv的模块之前执行
# 使用patch在模块级别mock，确保所有导入都使用mock版本
import dotenv
import dotenv.main

def mock_load_dotenv(*args, **kwargs):
    """Mock load_dotenv，不实际读取文件，直接返回True"""
    return True

# 替换dotenv模块的load_dotenv函数（多个位置）
dotenv.load_dotenv = mock_load_dotenv
if hasattr(dotenv, 'main'):
    dotenv.main.load_dotenv = mock_load_dotenv
if hasattr(dotenv, 'load_dotenv'):
    # 确保所有可能的引用都被替换
    import sys
    sys.modules['dotenv'].load_dotenv = mock_load_dotenv
    if 'dotenv.main' in sys.modules:
        sys.modules['dotenv.main'].load_dotenv = mock_load_dotenv

# Mock tushare导入，避免SSL权限问题
# 在导入任何使用tushare的模块之前执行
try:
    import sys
    # 创建一个mock的tushare模块
    mock_tushare = type(sys)('tushare')
    mock_tushare.pro = type(sys)('tushare.pro')
    mock_tushare.pro_api = Mock()
    # 添加set_token方法
    mock_tushare.set_token = Mock(return_value=None)
    sys.modules['tushare'] = mock_tushare
    sys.modules['tushare.pro'] = mock_tushare.pro
except Exception:
    pass


@pytest.fixture(scope="session")
def project_path():
    """项目根目录路径"""
    return project_root


@pytest.fixture(scope="session")
def test_data_dir():
    """测试数据目录"""
    test_dir = project_root / "tests" / "data"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture(scope="session")
def temp_dir():
    """临时文件目录"""
    temp_dir = project_root / "tests" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


@pytest.fixture
def mock_data_manager():
    """模拟DataManager"""
    from src.data.data_manager import DataManager
    
    mock_dm = Mock(spec=DataManager)
    mock_dm.source = 'tushare'
    mock_dm.fetcher = Mock()
    
    # 模拟股票列表
    mock_stock_list = pd.DataFrame({
        'ts_code': ['000001.SZ', '600000.SH', '000002.SZ'],
        'name': ['平安银行', '浦发银行', '万科A'],
        'list_date': ['19910403', '19991110', '19910129'],
        'market': ['主板', '主板', '主板']
    })
    mock_dm.get_stock_list.return_value = mock_stock_list
    
    # 模拟日线数据
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    mock_daily_data = pd.DataFrame({
        'trade_date': dates.strftime('%Y%m%d'),
        'open': np.random.uniform(10, 20, 30),
        'high': np.random.uniform(15, 25, 30),
        'low': np.random.uniform(8, 18, 30),
        'close': np.random.uniform(10, 20, 30),
        'vol': np.random.uniform(1000000, 10000000, 30),
        'amount': np.random.uniform(10000000, 100000000, 30),
    })
    mock_dm.get_daily_data.return_value = mock_daily_data
    
    # 模拟其他方法
    mock_dm.get_weekly_data.return_value = pd.DataFrame({
        'trade_date': dates.strftime('%Y%m%d'),
        'close': np.random.uniform(10, 20, 30),
    })
    mock_dm.get_daily_basic.return_value = pd.DataFrame({
        'trade_date': dates.strftime('%Y%m%d'),
        'volume_ratio': [1.2] * 30,
    })
    mock_dm.get_stk_factor.return_value = pd.DataFrame({
        'trade_date': dates.strftime('%Y%m%d'),
        'macd': [0.5] * 30,
    })
    mock_dm.get_trade_calendar.return_value = pd.DataFrame({
        'cal_date': dates.strftime('%Y%m%d'),
        'is_open': [1] * 30,
    })
    mock_dm.batch_get_daily_data.return_value = {
        '000001.SZ': mock_daily_data,
        '600000.SH': mock_daily_data,
    }
    mock_dm.batch_get_daily_basic.return_value = pd.DataFrame({
        'ts_code': ['000001.SZ', '600000.SH'],
        'trade_date': ['20240101', '20240101'],
        'volume_ratio': [1.2, 1.3],
    })
    
    return mock_dm


@pytest.fixture
def sample_stock_data():
    """示例股票数据"""
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    return pd.DataFrame({
        'trade_date': dates.strftime('%Y%m%d'),
        'ts_code': '000001.SZ',
        'open': np.random.uniform(10, 20, 60),
        'high': np.random.uniform(15, 25, 60),
        'low': np.random.uniform(8, 18, 60),
        'close': np.random.uniform(10, 20, 60),
        'vol': np.random.uniform(1000000, 10000000, 60),
        'amount': np.random.uniform(10000000, 100000000, 60),
    })


@pytest.fixture
def sample_stocks_df():
    """示例股票列表DataFrame"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '600000.SH', '000002.SZ'],
        '股票代码': ['000001.SZ', '600000.SH', '000002.SZ'],
        'name': ['平安银行', '浦发银行', '万科A'],
        '股票名称': ['平安银行', '浦发银行', '万科A'],
        'list_date': ['19910403', '19991110', '19910129'],
        '牛股概率': [0.85, 0.72, 0.68],
    })


@pytest.fixture
def mock_tushare_fetcher():
    """模拟TushareFetcher"""
    mock_fetcher = Mock()
    
    # 模拟股票列表
    mock_fetcher.get_stock_list.return_value = pd.DataFrame({
        'ts_code': ['000001.SZ', '600000.SH'],
        'name': ['平安银行', '浦发银行'],
        'list_date': ['19910403', '19991110'],
    })
    
    # 模拟日线数据
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    mock_fetcher.get_daily_data.return_value = pd.DataFrame({
        'trade_date': dates.strftime('%Y%m%d'),
        'open': np.random.uniform(10, 20, 30),
        'high': np.random.uniform(15, 25, 30),
        'low': np.random.uniform(8, 18, 30),
        'close': np.random.uniform(10, 20, 30),
        'vol': np.random.uniform(1000000, 10000000, 30),
    })
    
    return mock_fetcher


@pytest.fixture(autouse=True)
def reset_logger():
    """每个测试前重置logger配置"""
    from loguru import logger
    logger.remove()
    yield
    logger.remove()


@pytest.fixture
def mock_config():
    """模拟配置对象"""
    config = Mock()
    config.LOG_LEVEL = 'INFO'
    config.LOG_FORMAT = '{time} | {level} | {message}'
    return config


@pytest.fixture
def sample_model_data():
    """示例模型数据（用于模型测试）"""
    return {
        'features': ['feature1', 'feature2', 'feature3'],
        'target': [0, 1, 0, 1, 0],
        'weights': [1.0, 1.0, 1.0, 1.0, 1.0]
    }


@pytest.fixture
def mock_model():
    """模拟模型对象"""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([0.7, 0.8, 0.6])
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.2, 0.8], [0.4, 0.6]])
    return mock_model


@pytest.fixture
def sample_prediction_result():
    """示例预测结果"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '600000.SH', '000002.SZ'],
        'prediction': [0.85, 0.72, 0.68],
        'confidence': [0.9, 0.8, 0.75],
        'date': ['20240101', '20240101', '20240101']
    })


@pytest.fixture
def clean_temp_dir(temp_dir):
    """清理临时目录的fixture"""
    import shutil
    # 清理临时目录
    for item in temp_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    yield temp_dir
    # 测试后清理
    for item in temp_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


@pytest.fixture
def mock_xgboost_model():
    """模拟XGBoost模型"""
    import xgboost as xgb
    # 创建简单的XGBoost模型用于测试
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model = xgb.XGBClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def sample_time_series_data():
    """示例时间序列数据"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'date': dates,
        'value': np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000, 10000, 100)
    })


@pytest.fixture
def mock_cache_db(temp_dir):
    """模拟缓存数据库路径"""
    return temp_dir / "test_cache.db"


@pytest.fixture
def sample_technical_indicators():
    """示例技术指标数据"""
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    prices = np.random.uniform(10, 20, 60)
    return pd.DataFrame({
        'trade_date': dates.strftime('%Y%m%d'),
        'close': prices,
        'ma5': pd.Series(prices).rolling(5).mean(),
        'ma10': pd.Series(prices).rolling(10).mean(),
        'ma20': pd.Series(prices).rolling(20).mean(),
        'macd': np.random.uniform(-1, 1, 60),
        'rsi': np.random.uniform(30, 70, 60),
    })


# ============================================================================
# 测试辅助函数
# ============================================================================

def assert_dataframe_equal(df1, df2, check_dtype=False, check_index=False):
    """断言两个DataFrame相等（忽略顺序）"""
    import pandas.testing as pdt
    try:
        pdt.assert_frame_equal(
            df1.sort_values(by=df1.columns[0]).reset_index(drop=True),
            df2.sort_values(by=df2.columns[0]).reset_index(drop=True),
            check_dtype=check_dtype,
            check_index=check_index
        )
    except AssertionError as e:
        pytest.fail(f"DataFrames不相等: {e}")


def create_test_stock_data(ts_code='000001.SZ', days=60):
    """创建测试用的股票数据"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    return pd.DataFrame({
        'trade_date': dates.strftime('%Y%m%d'),
        'ts_code': ts_code,
        'open': np.random.uniform(10, 20, days),
        'high': np.random.uniform(15, 25, days),
        'low': np.random.uniform(8, 18, days),
        'close': np.random.uniform(10, 20, days),
        'vol': np.random.uniform(1000000, 10000000, days),
        'amount': np.random.uniform(10000000, 100000000, days),
    })


# 将辅助函数注册为fixture（可选）
@pytest.fixture
def test_helpers():
    """测试辅助函数集合"""
    return {
        'assert_dataframe_equal': assert_dataframe_equal,
        'create_test_stock_data': create_test_stock_data,
    }
