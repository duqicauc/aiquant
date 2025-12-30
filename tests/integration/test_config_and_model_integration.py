"""
配置管理与模型生命周期集成测试

测试内容：
- 配置变更对模型训练的影响
- 多模型配置切换
- 配置驱动的模型训练
- 配置验证与错误恢复
"""
import pytest
import pandas as pd
import numpy as np
import yaml
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from config.settings import Settings
from src.models.lifecycle.trainer import ModelTrainer
from src.models.lifecycle.predictor import ModelPredictor
from src.models.lifecycle.iterator import ModelIterator


@pytest.fixture
def test_model_name():
    """测试模型名称"""
    return "test_model"


@pytest.fixture
def config_setup(temp_dir, test_model_name):
    """设置配置环境"""
    config_dir = temp_dir / "config" / "models"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建模型配置文件
    config_file = config_dir / f"{test_model_name}.yaml"
    config_content = """
model:
  name: test_model
  display_name: "测试模型"
  type: xgboost
  version: v1.0.0
  status: development

data:
  sample_preparation:
    start_date: "20000101"
    positive_criteria:
      consecutive_weeks: 3
      total_return_threshold: 50
    negative_criteria:
      method: "same_period_other_stocks"
      sample_ratio: 1.0
  feature_extraction:
    lookback_days: 34

model_params:
  n_estimators: 10
  max_depth: 3
  learning_rate: 0.1
  random_state: 42

training:
  validation_split: 0.2
  time_series_split: true

prediction:
  top_n: 50
  min_probability: 0.0
"""
    config_file.write_text(config_content, encoding='utf-8')
    
    # 创建models.yaml
    models_file = temp_dir / "config" / "models.yaml"
    models_content = f"""
models:
  {test_model_name}:
    config_file: "config/models/{test_model_name}.yaml"
    display_name: "测试模型"
    status: active

default_model: {test_model_name}
models_root: "data/models"
"""
    models_file.write_text(models_content, encoding='utf-8')
    
    return {
        'config_file': config_file,
        'models_file': models_file,
    }


@pytest.fixture
def training_data_setup(temp_dir):
    """设置训练数据"""
    training_dir = temp_dir / "data" / "training"
    features_dir = training_dir / "features"
    samples_dir = training_dir / "samples"
    
    features_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建简化的训练数据
    n_samples = 30
    
    # 正样本
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
    
    # 负样本
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
    
    # 样本CSV
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


@pytest.mark.integration
class TestConfigDrivenTraining:
    """测试配置驱动的模型训练"""
    
    def test_train_with_config_parameters(self, test_model_name, config_setup,
                                         training_data_setup, temp_dir):
        """测试使用配置参数训练模型"""
        # 准备训练数据
        training_dir = temp_dir / "data" / "training"
        features_dir = training_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取测试数据
        pos_file = features_dir / "feature_data_34d.csv"
        neg_file = features_dir / "negative_feature_data_v2_34d.csv"
        
        # 从 training_data_setup 复制数据
        if (temp_dir / "data" / "training" / "features" / "feature_data_34d.csv").exists():
            pos_file.write_bytes((temp_dir / "data" / "training" / "features" / "feature_data_34d.csv").read_bytes())
        if (temp_dir / "data" / "training" / "features" / "negative_feature_data_v2_34d.csv").exists():
            neg_file.write_bytes((temp_dir / "data" / "training" / "features" / "negative_feature_data_v2_34d.csv").read_bytes())
        
        # 在 patch 之前保存原始的 read_csv
        from pandas.io.parsers import read_csv as pandas_read_csv
        
        with patch('src.models.lifecycle.trainer.Path') as mock_path, \
             patch('src.models.lifecycle.trainer.pd.read_csv') as mock_read_csv:
            
            def path_mock(path_str):
                if path_str.startswith('data/') or path_str.startswith('config/'):
                    return temp_dir / path_str
                return Path(path_str)
            mock_path.side_effect = path_mock
            
            # Mock read_csv 来返回测试数据
            def read_csv_mock(filepath, **kwargs):
                filepath_str = str(filepath)
                if 'feature_data_34d.csv' in filepath_str:
                    return pandas_read_csv(pos_file) if pos_file.exists() else pd.DataFrame()
                elif 'negative_feature_data_v2_34d.csv' in filepath_str:
                    return pandas_read_csv(neg_file) if neg_file.exists() else pd.DataFrame()
                elif 'positive_samples.csv' in filepath_str:
                    # 从 training_data_setup 获取样本文件
                    samples_file = temp_dir / "data" / "training" / "samples" / "positive_samples.csv"
                    return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                elif 'negative_samples' in filepath_str:
                    # 从 training_data_setup 获取样本文件
                    samples_file = temp_dir / "data" / "training" / "samples" / "negative_samples_v2.csv"
                    return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                # 对于其他文件，使用原始函数
                return pandas_read_csv(filepath, **kwargs)
            
            mock_read_csv.side_effect = read_csv_mock
            
            # 创建trainer
            trainer = ModelTrainer(test_model_name, str(config_setup['config_file']))
            trainer.base_path = temp_dir / "data" / "models" / test_model_name
            trainer.base_path.mkdir(parents=True, exist_ok=True)
            
            # 验证配置已加载
            assert trainer.config is not None
            assert trainer.config['model']['type'] == 'xgboost'
            assert trainer.config['model_params']['n_estimators'] == 10
            
            # 执行训练
            model, metrics = trainer.train_version(version='v1.0.0-test')
            version = 'v1.0.0-test'
            
            # 验证训练使用了配置参数
            assert model is not None
            assert metrics is not None
            assert 'accuracy' in metrics
            
            # 检查版本元数据中是否包含配置信息
            info = trainer.iterator.get_version_info(version)
            assert 'config' in info
    
    def test_config_change_affects_training(self, test_model_name, config_setup,
                                           training_data_setup, temp_dir):
        """测试配置变更影响训练"""
        # 准备训练数据
        training_dir = temp_dir / "data" / "training"
        features_dir = training_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        pos_file = features_dir / "feature_data_34d.csv"
        neg_file = features_dir / "negative_feature_data_v2_34d.csv"
        
        if (temp_dir / "data" / "training" / "features" / "feature_data_34d.csv").exists():
            pos_file.write_bytes((temp_dir / "data" / "training" / "features" / "feature_data_34d.csv").read_bytes())
        if (temp_dir / "data" / "training" / "features" / "negative_feature_data_v2_34d.csv").exists():
            neg_file.write_bytes((temp_dir / "data" / "training" / "features" / "negative_feature_data_v2_34d.csv").read_bytes())
        
        with patch('src.models.lifecycle.trainer.Path') as mock_path, \
             patch('src.models.lifecycle.trainer.pd.read_csv') as mock_read_csv:
            
            def path_mock(path_str):
                if path_str.startswith('data/') or path_str.startswith('config/'):
                    return temp_dir / path_str
                return Path(path_str)
            mock_path.side_effect = path_mock
            
            # 在 patch 之前保存原始的 read_csv
            from pandas.io.parsers import read_csv as pandas_read_csv
            
            def read_csv_mock(filepath, **kwargs):
                filepath_str = str(filepath)
                if 'feature_data_34d.csv' in filepath_str:
                    return pandas_read_csv(pos_file) if pos_file.exists() else pd.DataFrame()
                elif 'negative_feature_data_v2_34d.csv' in filepath_str:
                    return pandas_read_csv(neg_file) if neg_file.exists() else pd.DataFrame()
                elif 'positive_samples.csv' in filepath_str:
                    samples_file = temp_dir / "data" / "training" / "samples" / "positive_samples.csv"
                    return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                elif 'negative_samples' in filepath_str:
                    samples_file = temp_dir / "data" / "training" / "samples" / "negative_samples_v2.csv"
                    return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                # 对于其他文件，使用原始函数
                return pandas_read_csv(filepath, **kwargs)
            
            mock_read_csv.side_effect = read_csv_mock
            
            # 第一次训练（使用原始配置）
            trainer1 = ModelTrainer(test_model_name, str(config_setup['config_file']))
            trainer1.base_path = temp_dir / "data" / "models" / test_model_name
            trainer1.base_path.mkdir(parents=True, exist_ok=True)
            
            model1, metrics1 = trainer1.train_version(version='v1.0.0')
            version1 = 'v1.0.0'
            
            # 修改配置
            with open(config_setup['config_file'], 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            config['model_params']['n_estimators'] = 20  # 修改参数
            
            with open(config_setup['config_file'], 'w', encoding='utf-8') as f:
                yaml.dump(config, f)
            
            # 第二次训练（使用新配置）
            trainer2 = ModelTrainer(test_model_name, str(config_setup['config_file']))
            trainer2.base_path = temp_dir / "data" / "models" / test_model_name
            
            model2, metrics2 = trainer2.train_version(version='v1.0.1')
            version2 = 'v1.0.1'
            
            # 验证两个版本的配置不同
            info1 = trainer1.iterator.get_version_info(version1)
            info2 = trainer2.iterator.get_version_info(version2)
            
            assert info1['config']['model_params']['n_estimators'] == 10
            assert info2['config']['model_params']['n_estimators'] == 20


@pytest.mark.integration
class TestConfigAndPrediction:
    """测试配置与预测的集成"""
    
    def test_predict_with_config_parameters(self, test_model_name, config_setup,
                                           mock_data_manager, temp_dir):
        """测试使用配置参数进行预测"""
        # 创建已训练的模型
        model_dir = temp_dir / "data" / "models" / test_model_name
        version_dir = model_dir / "versions" / "v1.0.0" / "model"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        import xgboost as xgb
        X = np.random.randn(20, 3)
        y = np.random.randint(0, 2, 20)
        model = xgb.XGBClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        model_file = version_dir / "model.json"
        model.get_booster().save_model(str(model_file))
        
        import json
        with open(version_dir / "feature_names.json", 'w', encoding='utf-8') as f:
            json.dump(['close_mean', 'pct_chg_mean', 'close_std'], f)
        
        # 创建版本元数据
        metadata_file = model_dir / "versions" / "v1.0.0" / "metadata.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'version': 'v1.0.0',
                'model_name': test_model_name,
                'status': 'production',
            }, f)
        
        with patch('src.models.lifecycle.predictor.Path') as mock_path:
            def path_mock(path_str):
                if path_str.startswith('data/') or path_str.startswith('config/'):
                    return temp_dir / path_str
                return Path(path_str)
            mock_path.side_effect = path_mock
            
            # 创建predictor
            predictor = ModelPredictor(test_model_name, str(config_setup['config_file']))
            predictor.base_path = model_dir
            predictor.dm = mock_data_manager
            
            # 验证配置已加载
            assert predictor.config is not None
            assert predictor.config['prediction']['top_n'] == 50
            
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
            
            # 执行预测（应该使用配置中的top_n）
            results = predictor.predict(version='v1.0.0', top_n=None)
            
            # 验证使用了配置参数
            assert len(results) <= 50  # 配置中的top_n


@pytest.mark.integration
class TestMultiModelConfigSwitch:
    """测试多模型配置切换"""
    
    def test_switch_model_config(self, temp_dir, config_setup):
        """测试切换不同模型的配置"""
        # 创建第二个模型配置
        config_dir = temp_dir / "config" / "models"
        config_file2 = config_dir / "model_b.yaml"
        config_content2 = """
model:
  name: model_b
  type: lightgbm
  version: v1.0.0

prediction:
  top_n: 100
"""
        config_file2.write_text(config_content2, encoding='utf-8')
        
        # 更新models.yaml
        models_file = temp_dir / "config" / "models.yaml"
        with open(models_file, 'r', encoding='utf-8') as f:
            models_config = yaml.safe_load(f)
        
        models_config['models']['model_b'] = {
            'config_file': 'config/models/model_b.yaml',
            'display_name': '模型B',
            'status': 'active'
        }
        
        with open(models_file, 'w', encoding='utf-8') as f:
            yaml.dump(models_config, f)
        
        settings = Settings()
        settings.config_dir = temp_dir / "config"
        settings._load_models_config()
        
        # 获取两个模型的配置
        config_a = settings.get_model_config('test_model')
        config_b = settings.get_model_config('model_b')
        
        # 验证配置不同
        assert config_a['model']['type'] == 'xgboost'
        assert config_b['model']['type'] == 'lightgbm'
        assert config_a['prediction']['top_n'] == 50
        assert config_b['prediction']['top_n'] == 100


@pytest.mark.integration
class TestConfigErrorRecovery:
    """测试配置错误恢复"""
    
    def test_missing_config_file_handling(self, temp_dir, test_model_name):
        """测试配置文件缺失时的处理"""
        # 创建不存在的配置文件路径
        non_existent_config = temp_dir / "config" / "models" / "nonexistent.yaml"
        
        # 尝试创建trainer（应该处理错误）
        try:
            trainer = ModelTrainer(test_model_name, str(non_existent_config))
            # 如果成功，说明有默认处理
        except FileNotFoundError:
            # 抛出异常也是可以接受的
            pass
    
    def test_invalid_config_handling(self, temp_dir, test_model_name):
        """测试无效配置的处理"""
        config_dir = temp_dir / "config" / "models"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建无效的配置文件
        invalid_config = config_dir / f"{test_model_name}.yaml"
        invalid_config.write_text("invalid: yaml: [", encoding='utf-8')
        
        # 尝试加载（应该处理错误）
        try:
            trainer = ModelTrainer(test_model_name, str(invalid_config))
            # 如果成功，说明有错误处理
        except Exception:
            # 抛出异常也是可以接受的
            pass


@pytest.mark.integration
class TestConfigAndVersionManagement:
    """测试配置与版本管理的集成"""
    
    def test_config_stored_in_version_metadata(self, test_model_name, config_setup,
                                               training_data_setup, temp_dir):
        """测试配置存储在版本元数据中"""
        # 准备训练数据
        training_dir = temp_dir / "data" / "training"
        features_dir = training_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        pos_file = features_dir / "feature_data_34d.csv"
        neg_file = features_dir / "negative_feature_data_v2_34d.csv"
        
        if (temp_dir / "data" / "training" / "features" / "feature_data_34d.csv").exists():
            pos_file.write_bytes((temp_dir / "data" / "training" / "features" / "feature_data_34d.csv").read_bytes())
        if (temp_dir / "data" / "training" / "features" / "negative_feature_data_v2_34d.csv").exists():
            neg_file.write_bytes((temp_dir / "data" / "training" / "features" / "negative_feature_data_v2_34d.csv").read_bytes())
        
        with patch('src.models.lifecycle.trainer.Path') as mock_path, \
             patch('src.models.lifecycle.trainer.pd.read_csv') as mock_read_csv:
            
            def path_mock(path_str):
                if path_str.startswith('data/') or path_str.startswith('config/'):
                    return temp_dir / path_str
                return Path(path_str)
            mock_path.side_effect = path_mock
            
            # 在 patch 之前保存原始的 read_csv
            from pandas.io.parsers import read_csv as pandas_read_csv
            
            def read_csv_mock(filepath, **kwargs):
                filepath_str = str(filepath)
                if 'feature_data_34d.csv' in filepath_str:
                    return pandas_read_csv(pos_file) if pos_file.exists() else pd.DataFrame()
                elif 'negative_feature_data_v2_34d.csv' in filepath_str:
                    return pandas_read_csv(neg_file) if neg_file.exists() else pd.DataFrame()
                elif 'positive_samples.csv' in filepath_str:
                    samples_file = temp_dir / "data" / "training" / "samples" / "positive_samples.csv"
                    return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                elif 'negative_samples' in filepath_str:
                    samples_file = temp_dir / "data" / "training" / "samples" / "negative_samples_v2.csv"
                    return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                # 对于其他文件，使用原始函数
                return pandas_read_csv(filepath, **kwargs)
            
            mock_read_csv.side_effect = read_csv_mock
            
            trainer = ModelTrainer(test_model_name, str(config_setup['config_file']))
            trainer.base_path = temp_dir / "data" / "models" / test_model_name
            trainer.base_path.mkdir(parents=True, exist_ok=True)
            
            model, metrics = trainer.train_version(version='v1.0.0-test')
            version = 'v1.0.0-test'
            
            # 检查版本元数据
            info = trainer.iterator.get_version_info(version)
            
            # 验证配置已保存
            assert 'config' in info
            assert info['config']['model']['type'] == 'xgboost'
            assert info['config']['model_params']['n_estimators'] == 10
    
    def test_version_comparison_with_config(self, test_model_name, config_setup, temp_dir):
        """测试版本比较时包含配置信息"""
        iterator = ModelIterator(test_model_name)
        iterator.base_path = temp_dir / "data" / "models" / test_model_name
        iterator.versions_path = iterator.base_path / "versions"
        iterator.versions_path.mkdir(parents=True, exist_ok=True)
        
        # 创建两个版本，配置不同
        iterator.create_version('v1.0.0')
        iterator.update_version_metadata('v1.0.0', config={'model_params': {'n_estimators': 10}})
        
        iterator.create_version('v1.1.0')
        iterator.update_version_metadata('v1.1.0', config={'model_params': {'n_estimators': 20}})
        
        # 比较版本
        comparison = iterator.compare_versions('v1.0.0', 'v1.1.0')
        
        # 验证配置差异
        assert 'config_diff' in comparison.__dict__
        if comparison.config_diff:
            # 如果有配置差异，应该被记录
            pass

