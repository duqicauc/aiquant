"""
模型训练流程集成测试

测试完整的模型训练流程：
1. 数据准备
2. 特征提取
3. 模型训练
4. 模型保存
5. 版本管理
"""
import pytest
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.models.lifecycle.trainer import ModelTrainer
from src.models.lifecycle.iterator import ModelIterator


@pytest.fixture
def test_model_name():
    """测试模型名称"""
    return "test_model"


@pytest.fixture
def training_data_setup(temp_dir, test_model_name):
    """设置训练数据环境"""
    # 创建目录结构
    training_dir = temp_dir / "data" / "training"
    features_dir = training_dir / "features"
    samples_dir = training_dir / "samples"
    
    features_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建简化的训练数据（用于快速测试）
    n_samples = 50
    
    # 正样本特征数据
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
                'pct_chg': np.random.randn() * 2 + 1,  # 正样本涨幅更大
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
    
    # 负样本特征数据
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
                'pct_chg': np.random.randn() * 0.5,  # 负样本涨幅小
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
    
    return {
        'training_dir': training_dir,
        'features_dir': features_dir,
        'samples_dir': samples_dir,
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
"""
    config_file.write_text(config_content, encoding='utf-8')
    return config_file


@pytest.mark.integration
class TestModelTrainingPipeline:
    """模型训练流程集成测试"""
    
    def test_training_pipeline_creates_version(self, test_model_name, model_config_file, 
                                               training_data_setup, temp_dir):
        """测试训练流程创建版本"""
        # 准备训练数据路径
        features_dir = training_data_setup['features_dir']
        samples_dir = training_data_setup['samples_dir']
        pos_file = features_dir / "feature_data_34d.csv"
        neg_file = features_dir / "negative_feature_data_v2_34d.csv"
        
        # 设置路径
        with patch('src.models.lifecycle.trainer.Path') as mock_path, \
             patch('src.models.lifecycle.trainer.pd.read_csv') as mock_read_csv:
            
            def path_mock(path_str):
                if path_str.startswith('data/'):
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
                    samples_file = samples_dir / "positive_samples.csv"
                    return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                elif 'negative_samples' in filepath_str:
                    samples_file = samples_dir / "negative_samples_v2.csv"
                    return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                return pandas_read_csv(filepath, **kwargs)
            
            mock_read_csv.side_effect = read_csv_mock
            
            # 创建trainer
            trainer = ModelTrainer(test_model_name, str(model_config_file))
            trainer.base_path = temp_dir / "data" / "models" / test_model_name
            trainer.base_path.mkdir(parents=True, exist_ok=True)
            
            # 执行训练（使用小参数快速测试）
            model, metrics = trainer.train_version(version='v1.0.0-test')
            version = 'v1.0.0-test'
            
            # 验证版本已创建
            assert model is not None
            assert metrics is not None
            
            # 验证版本目录存在
            version_path = trainer.iterator.versions_path / version
            assert version_path.exists()
            
            # 验证模型文件已保存
            model_file = version_path / "model" / "model.json"
            assert model_file.exists()
            
            # 验证指标文件已保存
            metrics_file = version_path / "training" / "metrics.json"
            assert metrics_file.exists()
    
    def test_training_pipeline_version_metadata(self, test_model_name, model_config_file,
                                               training_data_setup, temp_dir):
        """测试训练流程更新版本元数据"""
        # 准备训练数据路径
        features_dir = training_data_setup['features_dir']
        samples_dir = training_data_setup['samples_dir']
        pos_file = features_dir / "feature_data_34d.csv"
        neg_file = features_dir / "negative_feature_data_v2_34d.csv"
        
        with patch('src.models.lifecycle.trainer.Path') as mock_path, \
             patch('src.models.lifecycle.trainer.pd.read_csv') as mock_read_csv:
            
            def path_mock(path_str):
                if path_str.startswith('data/'):
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
                    samples_file = samples_dir / "positive_samples.csv"
                    return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                elif 'negative_samples' in filepath_str:
                    samples_file = samples_dir / "negative_samples_v2.csv"
                    return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                return pandas_read_csv(filepath, **kwargs)
            
            mock_read_csv.side_effect = read_csv_mock
            
            trainer = ModelTrainer(test_model_name, str(model_config_file))
            trainer.base_path = temp_dir / "data" / "models" / test_model_name
            trainer.base_path.mkdir(parents=True, exist_ok=True)
            
            model, metrics = trainer.train_version(version='v1.0.0-test')
            version = 'v1.0.0-test'
            
            # 检查版本元数据
            info = trainer.iterator.get_version_info(version)
            
            assert 'metrics' in info
            assert 'training' in info
            assert info['training']['samples']['train'] > 0
            assert info['training']['samples']['test'] > 0
    
    @pytest.mark.slow
    def test_training_pipeline_with_real_data(self):
        """测试使用真实数据的训练流程"""
        # 这个测试需要真实的训练数据文件
        pytest.skip("需要真实数据文件，跳过快速测试")
    
    def test_training_pipeline_increments_version(self, test_model_name, model_config_file,
                                                  training_data_setup, temp_dir):
        """测试训练流程自动递增版本号"""
        # 准备训练数据路径
        features_dir = training_data_setup['features_dir']
        samples_dir = training_data_setup['samples_dir']
        pos_file = features_dir / "feature_data_34d.csv"
        neg_file = features_dir / "negative_feature_data_v2_34d.csv"
        
        with patch('src.models.lifecycle.trainer.Path') as mock_path, \
             patch('src.models.lifecycle.trainer.pd.read_csv') as mock_read_csv:
            
            def path_mock(path_str):
                if path_str.startswith('data/'):
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
                    samples_file = samples_dir / "positive_samples.csv"
                    return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                elif 'negative_samples' in filepath_str:
                    samples_file = samples_dir / "negative_samples_v2.csv"
                    return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                return pandas_read_csv(filepath, **kwargs)
            
            mock_read_csv.side_effect = read_csv_mock
            
            trainer = ModelTrainer(test_model_name, str(model_config_file))
            trainer.base_path = temp_dir / "data" / "models" / test_model_name
            trainer.base_path.mkdir(parents=True, exist_ok=True)
            
            # 第一次训练
            model1, metrics1 = trainer.train_version(version='v1.0.0')
            version1 = 'v1.0.0'
            assert model1 is not None
            
            # 第二次训练（不指定版本，应该自动递增）
            # 获取当前所有版本，找到最新的
            existing_versions = trainer.iterator.list_versions()
            latest_before = existing_versions[-1] if existing_versions else None
            
            model2, metrics2 = trainer.train_version()
            assert model2 is not None
            
            # 获取训练后的最新版本
            version2 = trainer.iterator.get_latest_version()
            # 验证版本已递增（应该比之前的版本号大）
            assert version2 is not None
            if latest_before:
                # 验证版本号确实递增了
                version_key_before = trainer.iterator._version_key(latest_before)
                version_key_after = trainer.iterator._version_key(version2)
                assert version_key_after > version_key_before


@pytest.mark.integration
class TestModelTrainingAndVersionManagement:
    """模型训练与版本管理集成测试"""
    
    def test_training_and_version_promotion(self, test_model_name, model_config_file,
                                            training_data_setup, temp_dir):
        """测试训练后版本提升"""
        # 准备训练数据路径
        features_dir = training_data_setup['features_dir']
        samples_dir = training_data_setup['samples_dir']
        pos_file = features_dir / "feature_data_34d.csv"
        neg_file = features_dir / "negative_feature_data_v2_34d.csv"
        
        with patch('src.models.lifecycle.trainer.Path') as mock_path, \
             patch('src.models.lifecycle.trainer.pd.read_csv') as mock_read_csv:
            
            def path_mock(path_str):
                if path_str.startswith('data/'):
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
                    samples_file = samples_dir / "positive_samples.csv"
                    return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                elif 'negative_samples' in filepath_str:
                    samples_file = samples_dir / "negative_samples_v2.csv"
                    return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                return pandas_read_csv(filepath, **kwargs)
            
            mock_read_csv.side_effect = read_csv_mock
            
            trainer = ModelTrainer(test_model_name, str(model_config_file))
            trainer.base_path = temp_dir / "data" / "models" / test_model_name
            trainer.base_path.mkdir(parents=True, exist_ok=True)
            
            # 训练模型
            model, metrics = trainer.train_version(version='v1.0.0-test')
            version = 'v1.0.0-test'
            
            # 提升版本到testing
            iterator = ModelIterator(test_model_name)
            iterator.base_path = trainer.base_path
            iterator.versions_path = trainer.iterator.versions_path
            iterator.current_file = trainer.base_path / "current.json"
            iterator._ensure_current_file()
            
            # 先设置为 development，然后提升到 testing
            iterator.set_current_version(version, 'development')
            iterator.promote_version(version, 'testing')
            
            # 验证版本状态
            info = iterator.get_version_info(version)
            assert info['status'] == 'testing'
            
            # 验证当前版本指针
            current = iterator.get_current_version('testing')
            assert current == version

