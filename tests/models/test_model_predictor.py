"""
测试模型预测器（ModelPredictor）

测试内容：
- 预测器初始化
- 模型加载
- 股票列表获取
- 特征提取
- 批量预测
- 结果保存
"""
import pytest
import pandas as pd
import numpy as np
import json
import yaml
import xgboost as xgb
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.models.lifecycle.predictor import ModelPredictor


@pytest.fixture
def test_model_name():
    """测试模型名称"""
    return "test_model"


@pytest.fixture
def sample_config_file(temp_dir, test_model_name):
    """创建示例模型配置文件"""
    config_dir = temp_dir / "config" / "models"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / f"{test_model_name}.yaml"
    config_content = """
model:
  name: test_model
  type: xgboost

data:
  feature_extraction:
    lookback_days: 34

prediction:
  top_n: 50
  min_probability: 0.0
"""
    config_file.write_text(config_content, encoding='utf-8')
    return config_file


@pytest.fixture
def sample_model_file(temp_dir, test_model_name):
    """创建示例模型文件"""
    # 创建模型目录结构
    model_dir = temp_dir / "data" / "models" / test_model_name / "versions" / "v1.0.0" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建简单的XGBoost模型
    X = np.random.randn(20, 3)
    y = np.random.randint(0, 2, 20)
    model = xgb.XGBClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # 保存模型
    model_file = model_dir / "model.json"
    model.get_booster().save_model(str(model_file))
    
    # 保存特征名称
    feature_names = ['close_mean', 'pct_chg_mean', 'close_std']
    feature_names_file = model_dir / "feature_names.json"
    with open(feature_names_file, 'w', encoding='utf-8') as f:
        json.dump(feature_names, f)
    
    return model_file


@pytest.fixture
def predictor(test_model_name, sample_config_file, temp_dir, mock_data_manager):
    """创建测试用的ModelPredictor"""
    with patch('src.models.lifecycle.predictor.Path') as mock_path:
        def path_mock(path_str):
            if path_str.startswith('data/'):
                return temp_dir / path_str
            return Path(path_str)
        
        mock_path.side_effect = path_mock
        
        with patch('builtins.open', create=True) as mock_open:
            with open(sample_config_file, 'r') as f:
                config_content = yaml.safe_load(f)
            
            def open_mock(file_path, mode='r', **kwargs):
                if 'config/models' in str(file_path):
                    from io import StringIO
                    return StringIO(yaml.dump(config_content))
                return open(file_path, mode, **kwargs)
            
            mock_open.side_effect = open_mock
            
            predictor = ModelPredictor(test_model_name, str(sample_config_file))
            predictor.base_path = temp_dir / "data" / "models" / test_model_name
            predictor.dm = mock_data_manager
            
            return predictor


class TestModelPredictor:
    """测试ModelPredictor类"""
    
    def test_init(self, predictor, test_model_name):
        """测试初始化"""
        assert predictor.model_name == test_model_name
        assert predictor.iterator is not None
        assert predictor.dm is not None
        assert predictor.config is not None
    
    def test_load_model(self, predictor, sample_model_file):
        """测试模型加载"""
        # 创建版本目录
        version = 'v1.0.0'
        version_path = predictor.iterator.versions_path / version
        version_path.mkdir(parents=True, exist_ok=True)
        
        # 复制模型文件到版本目录
        model_dir = version_path / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy(sample_model_file, model_dir / "model.json")
        
        # 加载特征名称
        feature_names_file = model_dir / "feature_names.json"
        with open(feature_names_file, 'w', encoding='utf-8') as f:
            json.dump(['close_mean', 'pct_chg_mean', 'close_std'], f)
        
        # 测试加载模型
        model = predictor._load_model(version)
        
        assert model is not None
        assert hasattr(model, 'booster')
        assert model.feature_names == ['close_mean', 'pct_chg_mean', 'close_std']
    
    def test_get_valid_stocks(self, predictor, mock_data_manager):
        """测试获取有效股票列表"""
        stocks = predictor._get_valid_stocks()
        
        assert len(stocks) > 0
        assert 'ts_code' in stocks.columns
        assert 'name' in stocks.columns
        
        # 验证排除规则（不应该有ST股票）
        for _, stock in stocks.iterrows():
            assert 'ST' not in stock['name']
            assert not stock['ts_code'].endswith('.BJ')
    
    def test_extract_stock_features(self, predictor, mock_data_manager):
        """测试提取股票特征"""
        ts_code = '000001.SZ'
        name = '测试股票'
        lookback_days = 34
        
        # Mock日线数据
        dates = pd.date_range(end=datetime.now(), periods=lookback_days, freq='D')
        mock_daily = pd.DataFrame({
            'trade_date': dates.strftime('%Y%m%d'),
            'close': 10 + np.random.randn(lookback_days),
            'pct_chg': np.random.randn(lookback_days),
            'vol': 1000000 + np.random.randn(lookback_days) * 100000,
        })
        mock_data_manager.get_daily_data.return_value = mock_daily
        
        # 测试特征提取
        features = predictor._extract_stock_features(ts_code, name, lookback_days)
        
        assert features is not None
        assert 'latest_close' in features
        assert 'close_mean' in features
        assert 'pct_chg_mean' in features
        assert 'close_trend' in features
    
    def test_extract_stock_features_insufficient_data(self, predictor, mock_data_manager):
        """测试数据不足时的特征提取"""
        ts_code = '000001.SZ'
        name = '测试股票'
        
        # Mock数据不足的情况
        mock_data_manager.get_daily_data.return_value = pd.DataFrame({
            'trade_date': ['20240101', '20240102'],
            'close': [10, 11],
            'pct_chg': [1, 2],
            'vol': [1000000, 1100000],
        })
        
        # 应该返回None
        features = predictor._extract_stock_features(ts_code, name, 34)
        assert features is None
    
    @patch('src.models.lifecycle.predictor.xgb.Booster')
    def test_predict_with_mock_model(self, mock_booster, predictor, sample_model_file):
        """测试预测（使用Mock模型）"""
        # 创建版本
        version = 'v1.0.0'
        version_path = predictor.iterator.versions_path / version
        version_path.mkdir(parents=True, exist_ok=True)
        
        # Mock模型
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.7, 0.8, 0.6])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.2, 0.8], [0.4, 0.6]])
        mock_model.feature_names = ['close_mean', 'pct_chg_mean', 'close_std']
        
        # Mock加载模型
        with patch.object(predictor, '_load_model', return_value=mock_model):
            # Mock股票列表
            stocks = pd.DataFrame({
                'ts_code': ['000001.SZ', '600000.SH', '000002.SZ'],
                'name': ['股票1', '股票2', '股票3'],
            })
            
            with patch.object(predictor, '_get_valid_stocks', return_value=stocks):
                # Mock特征提取
                def extract_side_effect(ts_code, name, lookback_days, target_date=None):
                    return {
                        'latest_close': 10.0,
                        'close_mean': 10.0,
                        'pct_chg_mean': 1.0,
                        'close_std': 0.5,
                    }
                
                with patch.object(predictor, '_extract_stock_features', side_effect=extract_side_effect):
                    # 执行预测
                    results = predictor.predict(version=version, top_n=3)
                    
                    assert len(results) > 0
                    assert 'ts_code' in results.columns
                    assert 'name' in results.columns
                    assert 'probability' in results.columns
    
    def test_save_predictions(self, predictor, temp_dir):
        """测试保存预测结果"""
        # 创建版本目录
        version = 'v1.0.0'
        version_path = predictor.iterator.versions_path / version
        version_path.mkdir(parents=True, exist_ok=True)
        
        # 创建预测结果
        df_predictions = pd.DataFrame({
            'ts_code': ['000001.SZ', '600000.SH'],
            'name': ['股票1', '股票2'],
            'probability': [0.8, 0.7],
            'latest_close': [10.0, 11.0],
        })
        
        prediction_date = '20240101'
        
        # 测试保存
        predictor._save_predictions(df_predictions, version, prediction_date)
        
        # 检查文件是否保存
        prediction_dir = version_path / "prediction" / "results"
        csv_file = prediction_dir / f"predictions_{prediction_date}.csv"
        assert csv_file.exists()
        
        metadata_file = prediction_dir / f"metadata_{prediction_date}.json"
        assert metadata_file.exists()
        
        # 检查保存的内容
        saved_df = pd.read_csv(csv_file)
        assert len(saved_df) == len(df_predictions)
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            assert metadata['model_name'] == predictor.model_name
            assert metadata['version'] == version
            assert metadata['num_predictions'] == len(df_predictions)


class TestModelPredictorIntegration:
    """ModelPredictor集成测试"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_prediction_workflow(self, predictor):
        """测试完整预测流程（需要真实模型）"""
        # 这个测试需要真实的模型文件
        pytest.skip("需要真实模型文件，跳过单元测试")
    
    @pytest.mark.integration
    def test_predict_with_latest_version(self, predictor):
        """测试使用最新版本预测"""
        # 创建多个版本
        predictor.iterator.create_version('v1.0.0')
        predictor.iterator.create_version('v1.1.0')
        
        # Mock模型加载
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_model.feature_names = ['close_mean', 'pct_chg_mean']
        
        with patch.object(predictor, '_load_model', return_value=mock_model), \
             patch.object(predictor, '_get_valid_stocks', return_value=pd.DataFrame({
                 'ts_code': ['000001.SZ'],
                 'name': ['股票1'],
             })), \
             patch.object(predictor, '_extract_stock_features', return_value={
                 'latest_close': 10.0,
                 'close_mean': 10.0,
                 'pct_chg_mean': 1.0,
             }):
            
            # 使用latest版本
            results = predictor.predict(version='latest', top_n=1)
            
            assert len(results) > 0

