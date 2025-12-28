"""
模型注册表测试
"""
import pytest
from pathlib import Path
from src.models.model_registry import ModelRegistry, ModelConfig


class TestModelConfig:
    """ModelConfig测试类"""
    
    def test_init(self):
        """测试初始化"""
        config = ModelConfig(
            name='test_model',
            display_name='测试模型',
            description='这是一个测试模型',
            data_dir='test',
            model_dir='models',
            sample_dir='samples',
            metrics_dir='metrics',
            prediction_dir='predictions'
        )
        assert config.name == 'test_model'
        assert config.display_name == '测试模型'
    
    def test_all_fields_required(self):
        """测试所有字段都是必需的"""
        with pytest.raises(TypeError):
            ModelConfig(name='test')  # 缺少其他字段


class TestModelRegistry:
    """ModelRegistry测试类"""
    
    def test_register_model(self):
        """测试注册模型"""
        config = ModelConfig(
            name='test_model',
            display_name='测试模型',
            description='测试',
            data_dir='test',
            model_dir='models',
            sample_dir='samples',
            metrics_dir='metrics',
            prediction_dir='predictions'
        )
        
        ModelRegistry.register(config)
        assert 'test_model' in ModelRegistry.list_all()
    
    def test_get_model(self):
        """测试获取模型配置"""
        config = ModelRegistry.get('momentum')
        assert config is not None
        assert config.name == 'momentum'
        assert config.display_name == '动量模型'
    
    def test_get_nonexistent_model(self):
        """测试获取不存在的模型"""
        config = ModelRegistry.get('nonexistent_model')
        assert config is None
    
    def test_list_all_models(self):
        """测试列出所有模型"""
        models = ModelRegistry.list_all()
        assert isinstance(models, dict)
        assert len(models) > 0
        assert 'momentum' in models
        assert 'breakout' in models
    
    def test_get_model_paths(self):
        """测试获取模型路径"""
        paths = ModelRegistry.get_model_paths('momentum')
        assert isinstance(paths, dict)
        assert 'samples' in paths
        assert 'models' in paths
        assert 'metrics' in paths
        assert 'predictions' in paths
        assert 'training_base' in paths
        
        # 验证路径存在
        for key, path in paths.items():
            assert isinstance(path, Path)
    
    def test_get_model_paths_nonexistent(self):
        """测试获取不存在模型的路径"""
        with pytest.raises(ValueError, match="未注册"):
            ModelRegistry.get_model_paths('nonexistent_model')
    
    def test_save_and_load_metadata(self, temp_dir):
        """测试保存和加载元数据"""
        # 先注册一个测试模型
        config = ModelConfig(
            name='test_metadata',
            display_name='测试元数据模型',
            description='测试',
            data_dir='test_metadata',
            model_dir='models',
            sample_dir='samples',
            metrics_dir='metrics',
            prediction_dir='predictions'
        )
        ModelRegistry.register(config)
        
        # 保存元数据
        metadata = {
            'version': '1.0.0',
            'accuracy': 0.85,
            'training_date': '20240101'
        }
        ModelRegistry.save_metadata('test_metadata', metadata)
        
        # 加载元数据
        loaded = ModelRegistry.load_metadata('test_metadata')
        assert loaded is not None
        assert loaded['model_name'] == 'test_metadata'
        assert loaded['display_name'] == '测试元数据模型'
        assert loaded['version'] == '1.0.0'
    
    def test_ensure_directories(self):
        """测试目录创建"""
        config = ModelConfig(
            name='test_dirs',
            display_name='测试目录',
            description='测试',
            data_dir='test_dirs',
            model_dir='models',
            sample_dir='samples',
            metrics_dir='metrics',
            prediction_dir='predictions'
        )
        
        ModelRegistry.register(config)
        
        # 验证目录已创建
        paths = ModelRegistry.get_model_paths('test_dirs')
        assert paths['samples'].exists()
        assert paths['models'].exists()
        assert paths['metrics'].exists()

