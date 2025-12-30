"""
测试路径配置（config/config.py）

测试内容：
- 路径常量
- 路径工具函数
"""
import pytest
from pathlib import Path
from config.config import (
    PROJECT_ROOT,
    MODELS_DIR,
    TRAINING_DIR,
    PREDICTION_DIR,
    get_model_path,
    get_training_path,
    get_prediction_path,
)


class TestPathConstants:
    """测试路径常量"""
    
    def test_project_root(self):
        """测试项目根目录"""
        assert isinstance(PROJECT_ROOT, Path)
        assert PROJECT_ROOT.exists()
        assert (PROJECT_ROOT / 'config').exists()
    
    def test_models_dir(self):
        """测试模型目录"""
        assert isinstance(MODELS_DIR, Path)
        assert MODELS_DIR == PROJECT_ROOT / 'data' / 'models'
    
    def test_training_dir(self):
        """测试训练目录"""
        assert isinstance(TRAINING_DIR, Path)
        assert TRAINING_DIR == PROJECT_ROOT / 'data' / 'training'
    
    def test_prediction_dir(self):
        """测试预测目录"""
        assert isinstance(PREDICTION_DIR, Path)
        assert PREDICTION_DIR == PROJECT_ROOT / 'data' / 'prediction'


class TestPathFunctions:
    """测试路径工具函数"""
    
    def test_get_model_path(self):
        """测试获取模型路径"""
        # 不带版本
        path = get_model_path('test_model')
        assert path == MODELS_DIR / 'test_model'
        
        # 带版本
        path = get_model_path('test_model', 'v1.0.0')
        assert path == MODELS_DIR / 'test_model' / 'versions' / 'v1.0.0'
    
    def test_get_training_path(self):
        """测试获取训练路径"""
        # 不带子目录
        path = get_training_path()
        assert path == TRAINING_DIR
        
        # 带子目录
        path = get_training_path('samples')
        assert path == TRAINING_DIR / 'samples'
        
        path = get_training_path('features')
        assert path == TRAINING_DIR / 'features'
    
    def test_get_prediction_path(self):
        """测试获取预测路径"""
        # 不带子目录
        path = get_prediction_path()
        assert path == PREDICTION_DIR
        
        # 带子目录
        path = get_prediction_path('results')
        assert path == PREDICTION_DIR / 'results'
        
        path = get_prediction_path('metadata')
        assert path == PREDICTION_DIR / 'metadata'

