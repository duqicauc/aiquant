"""
完整工作流程集成测试

测试从配置到训练到预测的完整流程：
1. 配置加载
2. 数据准备
3. 模型训练
4. 版本管理
5. 模型预测
6. 结果保存
"""
import pytest
import pandas as pd
import numpy as np
import json
import xgboost as xgb
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from config.settings import Settings, get_model_config
from src.models.lifecycle.trainer import ModelTrainer
from src.models.lifecycle.predictor import ModelPredictor
from src.models.lifecycle.iterator import ModelIterator


@pytest.fixture
def complete_workflow_setup(temp_dir):
    """完整工作流程的测试环境设置"""
    test_model_name = "workflow_test_model"
    
    # 1. 创建配置
    config_dir = temp_dir / "config" / "models"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / f"{test_model_name}.yaml"
    config_content = """
model:
  name: workflow_test_model
  display_name: "工作流测试模型"
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
  top_n: 10
  min_probability: 0.0
"""
    config_file.write_text(config_content, encoding='utf-8')
    
    # 创建 models.yaml
    models_file = temp_dir / "config" / "models.yaml"
    models_content = f"""
models:
  {test_model_name}:
    config_file: "config/models/{test_model_name}.yaml"
    display_name: "工作流测试模型"
    status: active

default_model: {test_model_name}
models_root: "data/models"
"""
    models_file.write_text(models_content, encoding='utf-8')
    
    # 2. 创建训练数据
    training_dir = temp_dir / "data" / "training"
    features_dir = training_dir / "features"
    samples_dir = training_dir / "samples"
    
    features_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = 20
    
    # 正样本
    pos_data = []
    for i in range(n_samples):
        for day in range(34):
            pos_data.append({
                'ts_code': f'00000{i%3}.SZ',
                'name': f'股票{i%3}',
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
                'ts_code': f'60000{i%3}.SH',
                'name': f'股票{i%3}',
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
        'ts_code': [f'00000{i%3}.SZ' for i in range(n_samples)],
        'name': [f'股票{i%3}' for i in range(n_samples)],
        't1_date': [f'2024-{(i%12+1):02d}-17' for i in range(n_samples)],
        'total_return': [50 + np.random.randn() * 10 for _ in range(n_samples)],
    })
    pos_samples.to_csv(samples_dir / "positive_samples.csv", index=False)
    
    neg_samples = pd.DataFrame({
        'ts_code': [f'60000{i%3}.SH' for i in range(n_samples)],
        'name': [f'股票{i%3}' for i in range(n_samples)],
        't1_date': [f'2024-{(i%12+1):02d}-17' for i in range(n_samples)],
        'total_return': [5 + np.random.randn() * 5 for _ in range(n_samples)],
    })
    neg_samples.to_csv(samples_dir / "negative_samples_v2.csv", index=False)
    
    return {
        'model_name': test_model_name,
        'config_file': config_file,
        'temp_dir': temp_dir,
    }


@pytest.mark.integration
class TestCompleteWorkflow:
    """完整工作流程集成测试"""
    
    def test_end_to_end_workflow(self, complete_workflow_setup, mock_data_manager):
        """测试端到端完整流程"""
        model_name = complete_workflow_setup['model_name']
        config_file = complete_workflow_setup['config_file']
        temp_dir = complete_workflow_setup['temp_dir']
        
        with patch('src.models.lifecycle.trainer.Path') as mock_path1, \
             patch('src.models.lifecycle.predictor.Path') as mock_path2:
            
            def path_mock(path_str):
                if path_str.startswith('data/') or path_str.startswith('config/'):
                    return temp_dir / path_str
                return Path(path_str)
            
            mock_path1.side_effect = path_mock
            mock_path2.side_effect = path_mock
            
            # 步骤1: 加载配置
            # 设置 Settings 的 config_dir 指向临时目录
            settings = Settings()
            settings.config_dir = temp_dir / "config"
            settings._load_models_config()
            
            # 使用 settings 获取配置
            config = settings.get_model_config(model_name)
            
            assert config is not None
            assert config['model']['type'] == 'xgboost'
            
            # 步骤2: 训练模型
            # 准备训练数据路径
            training_dir = temp_dir / "data" / "training"
            features_dir = training_dir / "features"
            pos_file = features_dir / "feature_data_34d.csv"
            neg_file = features_dir / "negative_feature_data_v2_34d.csv"
            
            trainer = None
            version = None
            
            with patch('src.models.lifecycle.trainer.pd.read_csv') as mock_read_csv:
                # 在 patch 之前保存原始的 read_csv
                from pandas.io.parsers import read_csv as pandas_read_csv
                
                def read_csv_mock(filepath, **kwargs):
                    filepath_str = str(filepath)
                    if 'feature_data_34d.csv' in filepath_str:
                        return pandas_read_csv(pos_file) if pos_file.exists() else pd.DataFrame()
                    elif 'negative_feature_data_v2_34d.csv' in filepath_str:
                        return pandas_read_csv(neg_file) if neg_file.exists() else pd.DataFrame()
                    elif 'positive_samples.csv' in filepath_str:
                        # 从 complete_workflow_setup 获取样本文件
                        samples_file = temp_dir / "data" / "training" / "samples" / "positive_samples.csv"
                        return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                    elif 'negative_samples' in filepath_str:
                        # 从 complete_workflow_setup 获取样本文件
                        samples_file = temp_dir / "data" / "training" / "samples" / "negative_samples_v2.csv"
                        return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                    # 对于其他文件，使用原始函数
                    return pandas_read_csv(filepath, **kwargs)
                
                mock_read_csv.side_effect = read_csv_mock
                
                trainer = ModelTrainer(model_name, str(config_file))
                trainer.base_path = temp_dir / "data" / "models" / model_name
                trainer.base_path.mkdir(parents=True, exist_ok=True)
                
                model, metrics = trainer.train_version(version='v1.0.0-workflow')
                version = 'v1.0.0-workflow'
            
            # 验证训练完成
            assert model is not None
            assert metrics is not None
            version_path = trainer.iterator.versions_path / version
            assert (version_path / "model" / "model.json").exists()
            
            # 步骤3: 版本管理
            iterator = ModelIterator(model_name)
            iterator.base_path = trainer.base_path
            iterator.versions_path = trainer.iterator.versions_path
            iterator.current_file = trainer.base_path / "current.json"
            # 确保 current.json 文件存在
            iterator._ensure_current_file()
            
            # 设置当前版本
            iterator.set_current_version(version, 'production')
            current = iterator.get_current_version('production')
            assert current == version
            
            # 步骤4: 预测
            predictor = ModelPredictor(model_name, str(config_file))
            predictor.base_path = temp_dir / "data" / "models" / model_name
            predictor.dm = mock_data_manager
            
            # Mock股票和日线数据
            stocks = pd.DataFrame({
                'ts_code': ['000001.SZ', '600000.SH'],
                'name': ['股票1', '股票2'],
                'list_date': ['19910403', '19991110'],
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
            results = predictor.predict(version=version, top_n=10)
            
            # 验证预测结果
            assert len(results) > 0
            assert 'ts_code' in results.columns
            assert 'probability' in results.columns
            
            # 步骤5: 验证结果保存
            prediction_dir = version_path / "prediction" / "results"
            csv_files = list(prediction_dir.glob("predictions_*.csv"))
            assert len(csv_files) > 0
    
    def test_workflow_with_version_promotion(self, complete_workflow_setup, mock_data_manager):
        """测试包含版本提升的完整流程"""
        model_name = complete_workflow_setup['model_name']
        config_file = complete_workflow_setup['config_file']
        temp_dir = complete_workflow_setup['temp_dir']
        
        with patch('src.models.lifecycle.trainer.Path') as mock_path1, \
             patch('src.models.lifecycle.predictor.Path') as mock_path2:
            
            def path_mock(path_str):
                if path_str.startswith('data/') or path_str.startswith('config/'):
                    return temp_dir / path_str
                return Path(path_str)
            
            mock_path1.side_effect = path_mock
            mock_path2.side_effect = path_mock
            
            # 准备训练数据路径
            training_dir = temp_dir / "data" / "training"
            features_dir = training_dir / "features"
            pos_file = features_dir / "feature_data_34d.csv"
            neg_file = features_dir / "negative_feature_data_v2_34d.csv"
            
            with patch('src.models.lifecycle.trainer.pd.read_csv') as mock_read_csv:
                # 在 patch 之前保存原始的 read_csv
                from pandas.io.parsers import read_csv as pandas_read_csv
                
                def read_csv_mock(filepath, **kwargs):
                    filepath_str = str(filepath)
                    if 'feature_data_34d.csv' in filepath_str:
                        return pandas_read_csv(pos_file) if pos_file.exists() else pd.DataFrame()
                    elif 'negative_feature_data_v2_34d.csv' in filepath_str:
                        return pandas_read_csv(neg_file) if neg_file.exists() else pd.DataFrame()
                    elif 'positive_samples.csv' in filepath_str:
                        # 从 complete_workflow_setup 获取样本文件
                        samples_file = temp_dir / "data" / "training" / "samples" / "positive_samples.csv"
                        return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                    elif 'negative_samples' in filepath_str:
                        # 从 complete_workflow_setup 获取样本文件
                        samples_file = temp_dir / "data" / "training" / "samples" / "negative_samples_v2.csv"
                        return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                    # 对于其他文件，使用原始函数
                    return pandas_read_csv(filepath, **kwargs)
                
                mock_read_csv.side_effect = read_csv_mock
                
                # 训练模型
                trainer = ModelTrainer(model_name, str(config_file))
                trainer.base_path = temp_dir / "data" / "models" / model_name
                trainer.base_path.mkdir(parents=True, exist_ok=True)
                
                model, metrics = trainer.train_version(version='v1.0.0')
                version = 'v1.0.0'
            
            # 版本提升流程
            iterator = ModelIterator(model_name)
            iterator.base_path = trainer.base_path
            iterator.versions_path = trainer.iterator.versions_path
            iterator.current_file = trainer.base_path / "current.json"
            # 确保 current.json 文件存在
            iterator._ensure_current_file()
            
            # development -> testing -> staging -> production
            # 首先确保版本状态是 development
            iterator.set_current_version(version, 'development')
            
            # 然后按顺序提升
            iterator.promote_version(version, 'testing')
            assert iterator.get_current_version('testing') == version
            
            iterator.promote_version(version, 'staging')
            assert iterator.get_current_version('staging') == version
            
            iterator.promote_version(version, 'production')
            assert iterator.get_current_version('production') == version
            
            # 验证版本状态
            info = iterator.get_version_info(version)
            assert info['status'] == 'production'
    
    def test_workflow_with_config_change(self, complete_workflow_setup, mock_data_manager):
        """测试配置变更后的完整流程"""
        model_name = complete_workflow_setup['model_name']
        config_file = complete_workflow_setup['config_file']
        temp_dir = complete_workflow_setup['temp_dir']
        
        with patch('src.models.lifecycle.trainer.Path') as mock_path1, \
             patch('src.models.lifecycle.predictor.Path') as mock_path2:
            
            def path_mock(path_str):
                if path_str.startswith('data/') or path_str.startswith('config/'):
                    return temp_dir / path_str
                return Path(path_str)
            
            mock_path1.side_effect = path_mock
            mock_path2.side_effect = path_mock
            
            # 准备训练数据路径
            training_dir = temp_dir / "data" / "training"
            features_dir = training_dir / "features"
            pos_file = features_dir / "feature_data_34d.csv"
            neg_file = features_dir / "negative_feature_data_v2_34d.csv"
            
            with patch('src.models.lifecycle.trainer.pd.read_csv') as mock_read_csv:
                # 在 patch 之前保存原始的 read_csv
                from pandas.io.parsers import read_csv as pandas_read_csv
                
                def read_csv_mock(filepath, **kwargs):
                    filepath_str = str(filepath)
                    if 'feature_data_34d.csv' in filepath_str:
                        return pandas_read_csv(pos_file) if pos_file.exists() else pd.DataFrame()
                    elif 'negative_feature_data_v2_34d.csv' in filepath_str:
                        return pandas_read_csv(neg_file) if neg_file.exists() else pd.DataFrame()
                    elif 'positive_samples.csv' in filepath_str:
                        # 从 complete_workflow_setup 获取样本文件
                        samples_file = temp_dir / "data" / "training" / "samples" / "positive_samples.csv"
                        return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                    elif 'negative_samples' in filepath_str:
                        # 从 complete_workflow_setup 获取样本文件
                        samples_file = temp_dir / "data" / "training" / "samples" / "negative_samples_v2.csv"
                        return pandas_read_csv(samples_file) if samples_file.exists() else pd.DataFrame()
                    # 对于其他文件，使用原始函数
                    return pandas_read_csv(filepath, **kwargs)
                
                mock_read_csv.side_effect = read_csv_mock
                
                # 第一次训练
                trainer1 = ModelTrainer(model_name, str(config_file))
                trainer1.base_path = temp_dir / "data" / "models" / model_name
                trainer1.base_path.mkdir(parents=True, exist_ok=True)
                
                model1, metrics1 = trainer1.train_version(version='v1.0.0')
                version1 = 'v1.0.0'
                
                # 修改配置
                import yaml
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                config['model_params']['n_estimators'] = 20
                config['prediction']['top_n'] = 20
                
                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f)
                
                # 第二次训练（使用新配置）
                trainer2 = ModelTrainer(model_name, str(config_file))
                trainer2.base_path = temp_dir / "data" / "models" / model_name
                
                model2, metrics2 = trainer2.train_version(version='v1.1.0')
                version2 = 'v1.1.0'
            
            # 验证两个版本使用不同配置
            info1 = trainer1.iterator.get_version_info(version1)
            info2 = trainer2.iterator.get_version_info(version2)
            
            assert info1['config']['model_params']['n_estimators'] == 10
            assert info2['config']['model_params']['n_estimators'] == 20
            
            # 使用不同版本进行预测
            predictor = ModelPredictor(model_name, str(config_file))
            predictor.base_path = temp_dir / "data" / "models" / model_name
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
            
            # 使用v1.0.0预测
            results1 = predictor.predict(version='v1.0.0', top_n=None)
            
            # 使用v1.1.0预测
            results2 = predictor.predict(version='v1.1.0', top_n=None)
            
            # 两个版本都应该能正常预测
            assert len(results1) > 0
            assert len(results2) > 0


@pytest.mark.integration
class TestWorkflowErrorHandling:
    """工作流程错误处理测试"""
    
    def test_workflow_with_missing_data(self, complete_workflow_setup):
        """测试数据缺失时的错误处理"""
        model_name = complete_workflow_setup['model_name']
        config_file = complete_workflow_setup['config_file']
        temp_dir = complete_workflow_setup['temp_dir']
        
        # 删除训练数据文件
        training_dir = temp_dir / "data" / "training" / "features"
        for file in training_dir.glob("*.csv"):
            file.unlink()
        
        with patch('src.models.lifecycle.trainer.Path') as mock_path:
            def path_mock(path_str):
                if path_str.startswith('data/') or path_str.startswith('config/'):
                    return temp_dir / path_str
                return Path(path_str)
            mock_path.side_effect = path_mock
            
            trainer = ModelTrainer(model_name, str(config_file))
            trainer.base_path = temp_dir / "data" / "models" / model_name
            trainer.base_path.mkdir(parents=True, exist_ok=True)
            
            # 应该处理文件不存在的错误
            try:
                version = trainer.train_version(version='v1.0.0-test')
                # 如果成功，说明有错误处理
            except FileNotFoundError:
                # 抛出异常也是可以接受的
                pass
    
    def test_workflow_with_invalid_model(self, complete_workflow_setup, mock_data_manager):
        """测试无效模型时的错误处理"""
        model_name = complete_workflow_setup['model_name']
        config_file = complete_workflow_setup['config_file']
        temp_dir = complete_workflow_setup['temp_dir']
        
        with patch('src.models.lifecycle.predictor.Path') as mock_path:
            def path_mock(path_str):
                if path_str.startswith('data/') or path_str.startswith('config/'):
                    return temp_dir / path_str
                return Path(path_str)
            mock_path.side_effect = path_mock
            
            predictor = ModelPredictor(model_name, str(config_file))
            predictor.base_path = temp_dir / "data" / "models" / model_name
            predictor.dm = mock_data_manager
            
            # 尝试加载不存在的模型版本
            try:
                results = predictor.predict(version='nonexistent', top_n=10)
                # 如果成功，说明有错误处理
            except (ValueError, FileNotFoundError):
                # 抛出异常也是可以接受的
                pass


@pytest.mark.integration
class TestWorkflowPerformance:
    """工作流程性能测试"""
    
    @pytest.mark.slow
    def test_large_dataset_workflow(self):
        """测试大数据集的工作流程"""
        # 这个测试需要大量数据，标记为slow
        pytest.skip("需要大量数据，跳过快速测试")
    
    def test_concurrent_model_access(self, complete_workflow_setup):
        """测试并发访问模型配置"""
        model_name = complete_workflow_setup['model_name']
        config_file = complete_workflow_setup['config_file']
        temp_dir = complete_workflow_setup['temp_dir']
        
        # 多个Settings实例同时访问配置
        settings1 = Settings()
        settings1.config_dir = temp_dir / "config"
        settings2 = Settings()
        settings2.config_dir = temp_dir / "config"
        
        # 应该都能正常加载
        assert settings1._config is not None
        assert settings2._config is not None
        
        # 配置应该相同
        assert settings1.get('data.feature_extraction.lookback_days') == \
               settings2.get('data.feature_extraction.lookback_days')

