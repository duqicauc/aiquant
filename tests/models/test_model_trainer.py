"""
测试模型训练器（ModelTrainer）

测试内容：
- 训练器初始化
- 版本创建和管理
- 数据加载和准备
- 特征提取
- 时间序列划分
- 模型训练
- 模型保存
"""
import pytest
import pandas as pd
import numpy as np
import json
import yaml
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.models.lifecycle.trainer import ModelTrainer


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


@pytest.fixture
def sample_training_data(temp_dir):
    """创建示例训练数据"""
    # 创建训练数据目录
    training_dir = temp_dir / "data" / "training"
    features_dir = training_dir / "features"
    samples_dir = training_dir / "samples"
    
    features_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建正样本特征数据（模拟34天时序数据）
    n_samples = 100
    feature_data = []
    for i in range(n_samples):
        for day in range(34):
            feature_data.append({
                'ts_code': f'00000{i%10}.SZ',
                'name': f'股票{i%10}',
                'label': 1,
                'days_to_t1': day - 17,  # 以第17天为T1
                'trade_date': f'2024{(i%12+1):02d}{(day%28+1):02d}',
                'close': 10 + np.random.randn(),
                'pct_chg': np.random.randn() * 2,
                'vol': 1000000 + np.random.randn() * 100000,
                'volume_ratio': 1.5 + np.random.randn() * 0.5,
                'macd': np.random.randn() * 0.5,
                'ma5': 10 + np.random.randn() * 0.5,
                'ma10': 10 + np.random.randn() * 0.5,
                'total_mv': 1000000000,
                'circ_mv': 500000000,
            })
    
    df_features = pd.DataFrame(feature_data)
    features_file = features_dir / "feature_data_34d.csv"
    df_features.to_csv(features_file, index=False)
    
    # 创建负样本特征数据
    neg_feature_data = []
    for i in range(n_samples):
        for day in range(34):
            neg_feature_data.append({
                'ts_code': f'60000{i%10}.SH',
                'name': f'股票{i%10}',
                'label': 0,
                'days_to_t1': day - 17,
                'trade_date': f'2024{(i%12+1):02d}{(day%28+1):02d}',
                'close': 10 + np.random.randn(),
                'pct_chg': np.random.randn() * 0.5,
                'vol': 1000000 + np.random.randn() * 100000,
                'volume_ratio': 1.0 + np.random.randn() * 0.3,
                'macd': np.random.randn() * 0.2,
                'ma5': 10 + np.random.randn() * 0.3,
                'ma10': 10 + np.random.randn() * 0.3,
                'total_mv': 1000000000,
                'circ_mv': 500000000,
            })
    
    df_neg_features = pd.DataFrame(neg_feature_data)
    neg_features_file = features_dir / "negative_feature_data_v2_34d.csv"
    df_neg_features.to_csv(neg_features_file, index=False)
    
    # 创建正样本CSV（用于T1日期映射）
    positive_samples = []
    for i in range(n_samples):
        positive_samples.append({
            'ts_code': f'00000{i%10}.SZ',
            'name': f'股票{i%10}',
            't1_date': f'2024-{(i%12+1):02d}-17',
            'total_return': 50 + np.random.randn() * 10,
        })
    
    df_pos_samples = pd.DataFrame(positive_samples)
    pos_samples_file = samples_dir / "positive_samples.csv"
    df_pos_samples.to_csv(pos_samples_file, index=False)
    
    # 创建负样本CSV
    negative_samples = []
    for i in range(n_samples):
        negative_samples.append({
            'ts_code': f'60000{i%10}.SH',
            'name': f'股票{i%10}',
            't1_date': f'2024-{(i%12+1):02d}-17',
            'total_return': 5 + np.random.randn() * 5,
        })
    
    df_neg_samples = pd.DataFrame(negative_samples)
    neg_samples_file = samples_dir / "negative_samples_v2.csv"
    df_neg_samples.to_csv(neg_samples_file, index=False)
    
    return {
        'features_file': features_file,
        'neg_features_file': neg_features_file,
        'pos_samples_file': pos_samples_file,
        'neg_samples_file': neg_samples_file,
    }


@pytest.fixture
def trainer(test_model_name, sample_config_file, temp_dir):
    """创建测试用的ModelTrainer"""
    # 临时修改路径
    with patch('src.models.lifecycle.trainer.Path') as mock_path:
        # Mock路径指向临时目录
        def path_mock(path_str):
            if path_str.startswith('data/'):
                return temp_dir / path_str
            return Path(path_str)
        
        mock_path.side_effect = path_mock
        
        # Mock配置文件路径
        with patch('builtins.open', create=True) as mock_open:
            with open(sample_config_file, 'r') as f:
                config_content = yaml.safe_load(f)
            
            def open_mock(file_path, mode='r', **kwargs):
                if 'config/models' in str(file_path):
                    from io import StringIO
                    return StringIO(yaml.dump(config_content))
                return open(file_path, mode, **kwargs)
            
            mock_open.side_effect = open_mock
            
            trainer = ModelTrainer(test_model_name, str(sample_config_file))
            trainer.base_path = temp_dir / "data" / "models" / test_model_name
            trainer.base_path.mkdir(parents=True, exist_ok=True)
            
            return trainer


class TestModelTrainer:
    """测试ModelTrainer类"""
    
    def test_init(self, trainer, test_model_name):
        """测试初始化"""
        assert trainer.model_name == test_model_name
        assert trainer.iterator is not None
        assert trainer.config is not None
    
    def test_increment_version(self, trainer):
        """测试版本号递增"""
        # 测试版本号递增逻辑
        assert trainer._increment_version('v1.0.0') == 'v1.0.1'
        assert trainer._increment_version('v1.2.3') == 'v1.2.4'
        assert trainer._increment_version('v2.0.0') == 'v2.0.1'
    
    @patch('src.models.lifecycle.trainer.pd.read_csv')
    def test_load_and_prepare_data(self, mock_read_csv, trainer, sample_training_data):
        """测试数据加载和准备"""
        # Mock CSV读取
        pos_df = pd.read_csv(sample_training_data['features_file'])
        neg_df = pd.read_csv(sample_training_data['neg_features_file'])
        
        def read_csv_side_effect(file_path, **kwargs):
            if 'feature_data_34d.csv' in str(file_path):
                return pos_df.copy()
            elif 'negative_feature_data_v2_34d.csv' in str(file_path):
                return neg_df.copy()
            return pd.DataFrame()
        
        mock_read_csv.side_effect = read_csv_side_effect
        
        # 测试数据加载
        df = trainer._load_and_prepare_data(neg_version='v2')
        
        assert len(df) > 0
        assert 'label' in df.columns
        assert (df['label'] == 1).sum() > 0  # 有正样本
        assert (df['label'] == 0).sum() > 0  # 有负样本
    
    @patch('src.models.lifecycle.trainer.pd.read_csv')
    def test_extract_features(self, mock_read_csv, trainer, sample_training_data):
        """测试特征提取"""
        # 准备数据
        pos_df = pd.read_csv(sample_training_data['features_file'])
        neg_df = pd.read_csv(sample_training_data['neg_features_file'])
        df = pd.concat([pos_df, neg_df])
        df['label'] = df['label'].astype(int)
        
        # Mock样本CSV读取
        pos_samples = pd.read_csv(sample_training_data['pos_samples_file'])
        neg_samples = pd.read_csv(sample_training_data['neg_samples_file'])
        
        def read_csv_side_effect(file_path, **kwargs):
            if 'feature_data_34d.csv' in str(file_path):
                return pos_df.copy()
            elif 'negative_feature_data_v2_34d.csv' in str(file_path):
                return neg_df.copy()
            elif 'positive_samples.csv' in str(file_path):
                return pos_samples.copy()
            elif 'negative_samples_v2.csv' in str(file_path):
                return neg_samples.copy()
            return pd.DataFrame()
        
        mock_read_csv.side_effect = read_csv_side_effect
        
        # 测试特征提取
        df_features = trainer._extract_features(df)
        
        assert len(df_features) > 0
        assert 'sample_id' in df_features.columns
        assert 'label' in df_features.columns
        assert 't1_date' in df_features.columns
        # 检查特征列
        assert 'close_mean' in df_features.columns
        assert 'pct_chg_mean' in df_features.columns
    
    def test_timeseries_split(self, trainer):
        """测试时间序列划分"""
        # 创建示例特征数据
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        df_features = pd.DataFrame({
            'sample_id': range(100),
            'ts_code': [f'00000{i%10}.SZ' for i in range(100)],
            'name': [f'股票{i%10}' for i in range(100)],
            'label': [i % 2 for i in range(100)],
            't1_date': dates,
            'close_mean': np.random.randn(100),
            'pct_chg_mean': np.random.randn(100),
        })
        
        # 测试时间序列划分
        X_train, X_test, y_train, y_test, train_dates, test_dates = trainer._timeseries_split(df_features)
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        
        # 检查时间顺序（训练集应该早于测试集）
        assert train_dates.max() < test_dates.min()
    
    def test_train_model(self, trainer):
        """测试模型训练"""
        # 创建示例训练数据
        n_train = 50
        n_test = 20
        
        X_train = pd.DataFrame({
            'close_mean': np.random.randn(n_train),
            'pct_chg_mean': np.random.randn(n_train),
            'close_std': np.random.randn(n_train),
        })
        y_train = np.random.randint(0, 2, n_train)
        
        X_test = pd.DataFrame({
            'close_mean': np.random.randn(n_test),
            'pct_chg_mean': np.random.randn(n_test),
            'close_std': np.random.randn(n_test),
        })
        y_test = np.random.randint(0, 2, n_test)
        
        # 测试模型训练
        model, metrics = trainer._train_model(X_train, y_train, X_test, y_test)
        
        assert model is not None
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'auc' in metrics
        
        # 检查指标值在合理范围内
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['auc'] <= 1
    
    def test_save_model(self, trainer, temp_dir):
        """测试模型保存"""
        # 创建模拟模型
        import xgboost as xgb
        X = np.random.randn(20, 3)
        y = np.random.randint(0, 2, 20)
        model = xgb.XGBClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # 创建版本目录
        version = 'v1.0.0'
        version_path = trainer.iterator.versions_path / version
        version_path.mkdir(parents=True, exist_ok=True)
        
        metrics = {'accuracy': 0.8, 'precision': 0.75, 'recall': 0.8}
        train_dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
        test_dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        feature_cols = ['close_mean', 'pct_chg_mean', 'close_std']
        
        # 测试保存模型
        trainer._save_model(model, metrics, version, train_dates, test_dates, feature_cols)
        
        # 检查文件是否保存
        model_file = version_path / "model" / "model.json"
        assert model_file.exists()
        
        feature_names_file = version_path / "model" / "feature_names.json"
        assert feature_names_file.exists()
        
        metrics_file = version_path / "training" / "metrics.json"
        assert metrics_file.exists()
        
        # 检查保存的内容
        with open(feature_names_file, 'r', encoding='utf-8') as f:
            saved_features = json.load(f)
            assert saved_features == feature_cols


class TestModelTrainerIntegration:
    """ModelTrainer集成测试"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_training_workflow(self, trainer, sample_training_data):
        """测试完整训练流程（需要真实数据）"""
        # 这个测试需要真实的数据文件
        # 标记为slow和integration，避免在快速测试中运行
        pytest.skip("需要真实数据文件，跳过单元测试")
    
    @pytest.mark.integration
    def test_train_version_creates_new_version(self, trainer):
        """测试训练创建新版本"""
        # Mock所有依赖
        with patch.object(trainer, '_load_and_prepare_data') as mock_load, \
             patch.object(trainer, '_extract_features') as mock_extract, \
             patch.object(trainer, '_timeseries_split') as mock_split, \
             patch.object(trainer, '_train_model') as mock_train, \
             patch.object(trainer, '_save_model') as mock_save, \
             patch.object(trainer, '_generate_visualizations') as mock_viz:
            
            # Mock返回值
            mock_df = pd.DataFrame({'label': [0, 1]})
            mock_load.return_value = mock_df
            
            mock_features = pd.DataFrame({
                'sample_id': [1, 2],
                'label': [0, 1],
                't1_date': pd.date_range(start='2020-01-01', periods=2),
                'close_mean': [10, 11],
            })
            mock_extract.return_value = mock_features
            
            X_train = pd.DataFrame({'close_mean': [10]})
            X_test = pd.DataFrame({'close_mean': [11]})
            y_train = pd.Series([0])
            y_test = pd.Series([1])
            train_dates = pd.date_range(start='2020-01-01', periods=1)
            test_dates = pd.date_range(start='2023-01-01', periods=1)
            
            mock_split.return_value = (X_train, X_test, y_train, y_test, train_dates, test_dates)
            
            import xgboost as xgb
            model = xgb.XGBClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            metrics = {'accuracy': 0.8}
            mock_train.return_value = (model, metrics)
            
            # 执行训练
            version = trainer.train_version(version='v1.0.0-test')
            
            # 验证
            assert version == 'v1.0.0-test'
            mock_load.assert_called_once()
            mock_extract.assert_called_once()
            mock_split.assert_called_once()
            mock_train.assert_called_once()
            mock_save.assert_called_once()

