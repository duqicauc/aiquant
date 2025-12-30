"""
模型预测流程集成测试

测试完整的模型预测流程：
1. 模型加载
2. 股票列表获取
3. 特征提取
4. 批量预测
5. 结果保存
"""
import pytest
import pandas as pd
import numpy as np
import json
import xgboost as xgb
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.models.lifecycle.predictor import ModelPredictor
from src.models.lifecycle.iterator import ModelIterator
from src.models.lifecycle.trainer import ModelTrainer


@pytest.fixture
def test_model_name():
    """测试模型名称"""
    return "test_model"


@pytest.fixture
def trained_model_setup(temp_dir, test_model_name):
    """设置已训练的模型环境"""
    # 创建模型目录结构
    model_dir = temp_dir / "data" / "models" / test_model_name
    version_dir = model_dir / "versions" / "v1.0.0" / "model"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建简单的XGBoost模型（特征数量需要匹配）
    # 使用27个特征（训练时实际保存的特征数量，不包括 latest_close）
    n_features = 27
    X = np.random.randn(50, n_features)
    y = np.random.randint(0, 2, 50)
    model = xgb.XGBClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # 保存模型
    model_file = version_dir / "model.json"
    model.get_booster().save_model(str(model_file))
    
    # 保存特征名称（与训练时实际保存的特征一致，不包括 latest_close）
    # 训练时提取的特征不包括 latest_close，只有预测时才会添加
    feature_names = [
        'close_mean', 'close_std', 'close_max', 'close_min', 'close_trend',
        'pct_chg_mean', 'pct_chg_std', 'pct_chg_sum', 'positive_days', 'negative_days',
        'max_gain', 'max_loss', 'volume_ratio_mean', 'volume_ratio_max',
        'volume_ratio_gt_2', 'volume_ratio_gt_4', 'macd_mean', 'macd_positive_days',
        'macd_max', 'ma5_mean', 'price_above_ma5', 'ma10_mean', 'price_above_ma10',
        'total_mv_mean', 'circ_mv_mean', 'return_1w', 'return_2w'
    ]
    feature_names_file = version_dir / "feature_names.json"
    with open(feature_names_file, 'w', encoding='utf-8') as f:
        json.dump(feature_names, f)
    
    # 创建版本元数据
    metadata_file = model_dir / "versions" / "v1.0.0" / "metadata.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            'version': 'v1.0.0',
            'model_name': test_model_name,
            'status': 'production',
        }, f)
    
    return {
        'model_file': model_file,
        'feature_names': feature_names,
        'version': 'v1.0.0',
    }


@pytest.fixture
def model_config_file(temp_dir, test_model_name):
    """创建模型配置文件"""
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


@pytest.mark.integration
class TestModelPredictionPipeline:
    """模型预测流程集成测试"""
    
    def test_prediction_pipeline_loads_model(self, test_model_name, model_config_file,
                                             trained_model_setup, mock_data_manager, temp_dir):
        """测试预测流程加载模型"""
        with patch('src.models.lifecycle.predictor.Path') as mock_path:
            def path_mock(path_str):
                if path_str.startswith('data/'):
                    return temp_dir / path_str
                return Path(path_str)
            mock_path.side_effect = path_mock
            
            predictor = ModelPredictor(test_model_name, str(model_config_file))
            predictor.base_path = temp_dir / "data" / "models" / test_model_name
            predictor.dm = mock_data_manager
            
            # 测试加载模型
            model = predictor._load_model(trained_model_setup['version'])
            
            assert model is not None
            assert hasattr(model, 'booster')
            assert model.feature_names == trained_model_setup['feature_names']
    
    def test_prediction_pipeline_extracts_features(self, test_model_name, model_config_file,
                                                   trained_model_setup, mock_data_manager, temp_dir):
        """测试预测流程特征提取"""
        with patch('src.models.lifecycle.predictor.Path') as mock_path:
            def path_mock(path_str):
                if path_str.startswith('data/'):
                    return temp_dir / path_str
                return Path(path_str)
            mock_path.side_effect = path_mock
            
            predictor = ModelPredictor(test_model_name, str(model_config_file))
            predictor.base_path = temp_dir / "data" / "models" / test_model_name
            predictor.dm = mock_data_manager
            
            # Mock日线数据
            dates = pd.date_range(end=datetime.now(), periods=34, freq='D')
            mock_daily = pd.DataFrame({
                'trade_date': dates.strftime('%Y%m%d'),
                'close': 10 + np.random.randn(34),
                'pct_chg': np.random.randn(34),
                'vol': 1000000 + np.random.randn(34) * 100000,
            })
            mock_data_manager.get_daily_data.return_value = mock_daily
            
            # 测试特征提取
            features = predictor._extract_stock_features('000001.SZ', '测试股票', 34)
            
            assert features is not None
            assert 'close_mean' in features
            assert 'pct_chg_mean' in features
            assert 'close_trend' in features
    
    def test_prediction_pipeline_full_workflow(self, test_model_name, model_config_file,
                                               trained_model_setup, mock_data_manager, temp_dir):
        """测试完整预测流程"""
        with patch('src.models.lifecycle.predictor.Path') as mock_path:
            def path_mock(path_str):
                if path_str.startswith('data/'):
                    return temp_dir / path_str
                return Path(path_str)
            mock_path.side_effect = path_mock
            
            predictor = ModelPredictor(test_model_name, str(model_config_file))
            predictor.base_path = temp_dir / "data" / "models" / test_model_name
            predictor.dm = mock_data_manager
            
            # Mock股票列表
            stocks = pd.DataFrame({
                'ts_code': ['000001.SZ', '600000.SH', '000002.SZ'],
                'name': ['股票1', '股票2', '股票3'],
                'list_date': ['19910403', '19991110', '19910129'],
            })
            mock_data_manager.get_stock_list.return_value = stocks
            
            # Mock日线数据
            dates = pd.date_range(end=datetime.now(), periods=34, freq='D')
            mock_daily = pd.DataFrame({
                'trade_date': dates.strftime('%Y%m%d'),
                'close': 10 + np.random.randn(34),
                'pct_chg': np.random.randn(34),
                'vol': 1000000 + np.random.randn(34) * 100000,
            })
            mock_data_manager.get_daily_data.return_value = mock_daily
            
            # 执行预测
            results = predictor.predict(
                version=trained_model_setup['version'],
                top_n=3
            )
            
            assert len(results) > 0
            assert 'ts_code' in results.columns
            assert 'name' in results.columns
            assert 'probability' in results.columns
            
            # 验证结果已保存
            version_path = predictor.iterator.versions_path / trained_model_setup['version']
            prediction_dir = version_path / "prediction" / "results"
            
            # 查找保存的文件
            csv_files = list(prediction_dir.glob("predictions_*.csv"))
            assert len(csv_files) > 0
    
    def test_prediction_pipeline_with_latest_version(self, test_model_name, model_config_file,
                                                     trained_model_setup, mock_data_manager, temp_dir):
        """测试使用最新版本预测"""
        with patch('src.models.lifecycle.predictor.Path') as mock_path:
            def path_mock(path_str):
                if path_str.startswith('data/'):
                    return temp_dir / path_str
                return Path(path_str)
            mock_path.side_effect = path_mock
            
            predictor = ModelPredictor(test_model_name, str(model_config_file))
            predictor.base_path = temp_dir / "data" / "models" / test_model_name
            predictor.dm = mock_data_manager
            
            # Mock股票和日线数据
            stocks = pd.DataFrame({
                'ts_code': ['000001.SZ'],
                'name': ['股票1'],
                'list_date': ['19910403'],
            })
            mock_data_manager.get_stock_list.return_value = stocks
            
            dates = pd.date_range(end=datetime.now(), periods=34, freq='D')
            mock_daily = pd.DataFrame({
                'trade_date': dates.strftime('%Y%m%d'),
                'close': 10 + np.random.randn(34),
                'pct_chg': np.random.randn(34),
                'vol': 1000000 + np.random.randn(34) * 100000,
            })
            mock_data_manager.get_daily_data.return_value = mock_daily
            
            # 使用latest版本
            results = predictor.predict(version='latest', top_n=1)
            
            assert len(results) > 0
    
    def test_prediction_pipeline_saves_metadata(self, test_model_name, model_config_file,
                                                trained_model_setup, mock_data_manager, temp_dir):
        """测试预测流程保存元数据"""
        with patch('src.models.lifecycle.predictor.Path') as mock_path:
            def path_mock(path_str):
                if path_str.startswith('data/'):
                    return temp_dir / path_str
                return Path(path_str)
            mock_path.side_effect = path_mock
            
            predictor = ModelPredictor(test_model_name, str(model_config_file))
            predictor.base_path = temp_dir / "data" / "models" / test_model_name
            predictor.dm = mock_data_manager
            
            # Mock数据
            stocks = pd.DataFrame({
                'ts_code': ['000001.SZ'],
                'name': ['股票1'],
                'list_date': ['19910403'],
            })
            mock_data_manager.get_stock_list.return_value = stocks
            
            dates = pd.date_range(end=datetime.now(), periods=34, freq='D')
            mock_daily = pd.DataFrame({
                'trade_date': dates.strftime('%Y%m%d'),
                'close': 10 + np.random.randn(34),
                'pct_chg': np.random.randn(34),
                'vol': 1000000 + np.random.randn(34) * 100000,
            })
            mock_data_manager.get_daily_data.return_value = mock_daily
            
            # 执行预测
            results = predictor.predict(version=trained_model_setup['version'], top_n=1)
            
            # 检查元数据文件
            version_path = predictor.iterator.versions_path / trained_model_setup['version']
            prediction_dir = version_path / "prediction" / "results"
            metadata_files = list(prediction_dir.glob("metadata_*.json"))
            
            assert len(metadata_files) > 0
            
            # 检查元数据内容
            with open(metadata_files[0], 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                assert metadata['model_name'] == test_model_name
                assert metadata['version'] == trained_model_setup['version']
                assert metadata['num_predictions'] > 0


@pytest.mark.integration
class TestTrainingAndPredictionIntegration:
    """训练和预测流程集成测试"""
    
    def test_train_then_predict_workflow(self, test_model_name, model_config_file,
                                         mock_data_manager, temp_dir):
        """测试训练后立即预测的完整流程"""
        # 准备训练数据
        training_dir = temp_dir / "data" / "training"
        features_dir = training_dir / "features"
        samples_dir = training_dir / "samples"
        
        features_dir.mkdir(parents=True, exist_ok=True)
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建简化的训练数据
        n_samples = 30
        pos_data = []
        for i in range(n_samples):
            for day in range(34):
                pos_data.append({
                    'ts_code': f'00000{i%5}.SZ',
                    'name': f'股票{i%5}',
                    'label': 1,
                    'days_to_t1': day - 17,
                    'trade_date': f'2024{(i%12+1):02d}{(day%28+1):02d}',
                    'close': 10 + np.random.randn(),
                    'pct_chg': np.random.randn() * 2 + 1,
                    'vol': 1000000,
                    'volume_ratio': 1.5,
                    'macd': 0.5,
                    'ma5': 10,
                    'ma10': 10,
                    'total_mv': 1000000000,
                    'circ_mv': 500000000,
                })
        
        df_pos = pd.DataFrame(pos_data)
        df_pos.to_csv(features_dir / "feature_data_34d.csv", index=False)
        
        neg_data = []
        for i in range(n_samples):
            for day in range(34):
                neg_data.append({
                    'ts_code': f'60000{i%5}.SH',
                    'name': f'股票{i%5}',
                    'label': 0,
                    'days_to_t1': day - 17,
                    'trade_date': f'2024{(i%12+1):02d}{(day%28+1):02d}',
                    'close': 10 + np.random.randn(),
                    'pct_chg': np.random.randn() * 0.5,
                    'vol': 1000000,
                    'volume_ratio': 1.0,
                    'macd': 0.1,
                    'ma5': 10,
                    'ma10': 10,
                    'total_mv': 1000000000,
                    'circ_mv': 500000000,
                })
        
        df_neg = pd.DataFrame(neg_data)
        df_neg.to_csv(features_dir / "negative_feature_data_v2_34d.csv", index=False)
        
        pos_samples = pd.DataFrame({
            'ts_code': [f'00000{i%5}.SZ' for i in range(n_samples)],
            'name': [f'股票{i%5}' for i in range(n_samples)],
            't1_date': [f'2024-{(i%12+1):02d}-17' for i in range(n_samples)],
            'total_return': [50 + np.random.randn() * 10 for _ in range(n_samples)],
        })
        pos_samples.to_csv(samples_dir / "positive_samples.csv", index=False)
        
        neg_samples = pd.DataFrame({
            'ts_code': [f'60000{i%5}.SH' for i in range(n_samples)],
            'name': [f'股票{i%5}' for i in range(n_samples)],
            't1_date': [f'2024-{(i%12+1):02d}-17' for i in range(n_samples)],
            'total_return': [5 + np.random.randn() * 5 for _ in range(n_samples)],
        })
        neg_samples.to_csv(samples_dir / "negative_samples_v2.csv", index=False)
        
        pos_file = features_dir / "feature_data_34d.csv"
        neg_file = features_dir / "negative_feature_data_v2_34d.csv"
        
        with patch('src.models.lifecycle.trainer.Path') as mock_path, \
             patch('src.models.lifecycle.predictor.Path') as mock_path2, \
             patch('src.models.lifecycle.trainer.pd.read_csv') as mock_read_csv:
            
            def path_mock(path_str):
                if path_str.startswith('data/'):
                    return temp_dir / path_str
                return Path(path_str)
            
            mock_path.side_effect = path_mock
            mock_path2.side_effect = path_mock
            
            # 在 patch 之前保存原始的 read_csv
            from pandas.io.parsers import read_csv as pandas_read_csv
            
            def read_csv_mock(filepath, **kwargs):
                filepath_str = str(filepath)
                if 'feature_data_34d.csv' in filepath_str:
                    return pandas_read_csv(pos_file) if pos_file.exists() else pd.DataFrame()
                elif 'negative_feature_data_v2_34d.csv' in filepath_str:
                    return pandas_read_csv(neg_file) if neg_file.exists() else pd.DataFrame()
                elif 'positive_samples.csv' in filepath_str:
                    samples_file = samples_dir / "positive_samples.csv"
                    return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                elif 'negative_samples' in filepath_str:
                    samples_file = samples_dir / "negative_samples_v2.csv"
                    return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                return pandas_read_csv(filepath, **kwargs)
            
            mock_read_csv.side_effect = read_csv_mock
            
            # 1. 训练模型
            trainer = ModelTrainer(test_model_name, str(model_config_file))
            trainer.base_path = temp_dir / "data" / "models" / test_model_name
            trainer.base_path.mkdir(parents=True, exist_ok=True)
            
            model, metrics = trainer.train_version(version='v1.0.0-test')
            version = 'v1.0.0-test'
            
            # 2. 使用训练的模型进行预测
            predictor = ModelPredictor(test_model_name, str(model_config_file))
            predictor.base_path = temp_dir / "data" / "models" / test_model_name
            predictor.dm = mock_data_manager
            
            # Mock股票和日线数据
            stocks = pd.DataFrame({
                'ts_code': ['000001.SZ'],
                'name': ['股票1'],
                'list_date': ['19910403'],
            })
            mock_data_manager.get_stock_list.return_value = stocks
            
            dates = pd.date_range(end=datetime.now(), periods=34, freq='D')
            mock_daily = pd.DataFrame({
                'trade_date': dates.strftime('%Y%m%d'),
                'close': 10 + np.random.randn(34),
                'pct_chg': np.random.randn(34),
                'vol': 1000000 + np.random.randn(34) * 100000,
            })
            mock_data_manager.get_daily_data.return_value = mock_daily
            
            # 执行预测
            results = predictor.predict(version='v1.0.0-test', top_n=1)
            
            # 验证流程完整性
            assert len(results) > 0
            assert version == 'v1.0.0-test'
            
            # 验证模型文件存在
            version_path = trainer.iterator.versions_path / version
            model_file = version_path / "model" / "model.json"
            assert model_file.exists()

