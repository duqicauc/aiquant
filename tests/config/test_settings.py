"""
测试配置管理系统（Settings）

测试内容：
- 全局配置加载
- 多模型配置
- 模型配置合并
- 配置路径工具
"""
import pytest
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from config.settings import Settings, get_model_config, get_setting


@pytest.fixture
def temp_config_dir():
    """临时配置目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_settings_yaml(temp_config_dir):
    """示例settings.yaml内容"""
    content = """
data:
  source: tushare
  sample_preparation:
    start_date: "20000101"
    end_date: null
    positive_criteria:
      consecutive_weeks: 3
      total_return_threshold: 50
    negative_criteria:
      method: "same_period_other_stocks"
      sample_ratio: 1.0
  feature_extraction:
    lookback_days: 34

model:
  type: xgboost
  version: v3

prediction:
  scoring:
    top_n: 50
    min_probability: 0.0
  exclusion_rules:
    exclude_st: true
    exclude_new_listed: true
    min_listing_days: 180

logging:
  level: INFO
  app_log: "logs/aiquant.log"
"""
    config_file = temp_config_dir / "settings.yaml"
    config_file.write_text(content, encoding='utf-8')
    return config_file


@pytest.fixture
def sample_models_yaml(temp_config_dir):
    """示例models.yaml内容"""
    content = """
models:
  test_model:
    config_file: "config/models/test_model.yaml"
    display_name: "测试模型"
    description: "用于测试的模型"
    status: active
    overrides:
      prediction:
        top_n: 30

default_model: test_model
models_root: "data/models"

shared:
  prediction:
    scoring:
      top_n: 50
      min_probability: 0.0
    exclusion_rules:
      exclude_st: true
      exclude_new_listed: true
      min_listing_days: 180
"""
    config_file = temp_config_dir / "models.yaml"
    config_file.write_text(content, encoding='utf-8')
    return config_file


@pytest.fixture
def sample_model_yaml(temp_config_dir):
    """示例模型独立配置文件"""
    content = """
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

model_params:
  n_estimators: 100
  max_depth: 5
  learning_rate: 0.1

prediction:
  top_n: 50
  min_probability: 0.0
"""
    # 创建模型配置目录
    model_config_dir = temp_config_dir / "models"
    model_config_dir.mkdir(exist_ok=True)
    
    config_file = model_config_dir / "test_model.yaml"
    config_file.write_text(content, encoding='utf-8')
    return config_file


class TestSettings:
    """测试Settings类"""
    
    def test_load_settings(self, sample_settings_yaml):
        """测试加载settings.yaml"""
        settings = Settings(str(sample_settings_yaml))
        
        assert settings.get('data.source') == 'tushare'
        assert settings.get('data.sample_preparation.start_date') == '20000101'
        assert settings.get('prediction.scoring.top_n') == 50
    
    def test_get_set_methods(self, sample_settings_yaml):
        """测试get和set方法"""
        settings = Settings(str(sample_settings_yaml))
        
        # 测试get
        top_n = settings.get('prediction.scoring.top_n')
        assert top_n == 50
        
        # 测试set
        settings.set('prediction.scoring.top_n', 100)
        assert settings.get('prediction.scoring.top_n') == 100
    
    def test_properties(self, sample_settings_yaml):
        """测试配置属性"""
        settings = Settings(str(sample_settings_yaml))
        
        assert isinstance(settings.data, dict)
        assert isinstance(settings.model, dict)
        assert isinstance(settings.prediction, dict)
        assert isinstance(settings.logging, dict)
    
    def test_load_models_config(self, sample_models_yaml):
        """测试加载models.yaml"""
        settings = Settings()
        settings.config_dir = Path(sample_models_yaml).parent
        settings._load_models_config()
        
        assert len(settings.models) > 0
        assert 'test_model' in settings.models
        assert settings.default_model == 'test_model'
    
    def test_list_models(self, sample_models_yaml):
        """测试列出所有模型"""
        settings = Settings()
        settings.config_dir = Path(sample_models_yaml).parent
        settings._load_models_config()
        
        models = settings.list_models()
        assert 'test_model' in models
    
    def test_get_model_info(self, sample_models_yaml):
        """测试获取模型信息"""
        settings = Settings()
        settings.config_dir = Path(sample_models_yaml).parent
        settings._load_models_config()
        
        info = settings.get_model_info('test_model')
        assert info is not None
        assert info['display_name'] == '测试模型'
        assert info['status'] == 'active'
    
    def test_get_model_config(self, sample_models_yaml, sample_model_yaml):
        """测试获取完整模型配置"""
        config_dir = Path(sample_models_yaml).parent
        
        settings = Settings()
        settings.config_dir = config_dir
        settings._load_models_config()
        
        # 需要设置正确的模型配置文件路径
        model_info = settings.get_model_info('test_model')
        model_info['config_file'] = str(sample_model_yaml)
        
        config = settings.get_model_config('test_model')
        
        assert config['_model_name'] == 'test_model'
        assert config['_display_name'] == '测试模型'
        assert config['model']['type'] == 'xgboost'
        assert config['prediction']['top_n'] == 30  # 被overrides覆盖
    
    def test_deep_merge(self, sample_models_yaml):
        """测试深度合并配置"""
        settings = Settings()
        settings.config_dir = Path(sample_models_yaml).parent
        settings._load_models_config()
        
        base = {'a': {'b': 1, 'c': 2}}
        override = {'a': {'b': 3}, 'd': 4}
        
        merged = settings._deep_merge(base, override)
        
        assert merged['a']['b'] == 3  # 被覆盖
        assert merged['a']['c'] == 2  # 保留
        assert merged['d'] == 4  # 新增
    
    def test_get_model_path(self, sample_models_yaml):
        """测试获取模型路径"""
        settings = Settings()
        settings.config_dir = Path(sample_models_yaml).parent
        settings._load_models_config()
        
        path = settings.get_model_path('test_model')
        assert path == Path('data/models/test_model')
    
    def test_get_default_value(self, sample_settings_yaml):
        """测试获取默认值"""
        settings = Settings(str(sample_settings_yaml))
        
        # 不存在的key应该返回默认值
        value = settings.get('nonexistent.key', 'default')
        assert value == 'default'


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    def test_get_setting(self, sample_settings_yaml):
        """测试get_setting函数"""
        with patch('config.settings.settings', Settings(str(sample_settings_yaml))):
            from config.settings import get_setting
            
            top_n = get_setting('prediction.scoring.top_n')
            assert top_n == 50
    
    def test_get_model_config_function(self, sample_models_yaml, sample_model_yaml):
        """测试get_model_config函数"""
        config_dir = Path(sample_models_yaml).parent
        
        with patch('config.settings.settings') as mock_settings:
            mock_settings._load_models_config = lambda: None
            mock_settings.config_dir = config_dir
            mock_settings._models_config = yaml.safe_load(sample_models_yaml.read_text())
            mock_settings._model_configs_cache = {}
            
            # Mock get_model_config方法
            def mock_get_model_config(model_name):
                if model_name == 'test_model':
                    return {
                        '_model_name': 'test_model',
                        '_display_name': '测试模型',
                        'model': {'type': 'xgboost'},
                        'prediction': {'top_n': 30}
                    }
                raise ValueError(f"模型 {model_name} 未注册")
            
            mock_settings.get_model_config = mock_get_model_config
            mock_settings.default_model = 'test_model'
            
            from config.settings import get_model_config
            
            config = get_model_config('test_model')
            assert config['_model_name'] == 'test_model'
            assert config['prediction']['top_n'] == 30


class TestConfigIntegration:
    """测试配置系统集成"""
    
    def test_settings_and_models_integration(self, sample_settings_yaml, sample_models_yaml):
        """测试settings和models配置的集成"""
        config_dir = Path(sample_models_yaml).parent
        
        settings = Settings(str(sample_settings_yaml))
        settings.config_dir = config_dir
        settings._load_models_config()
        
        # 全局配置
        assert settings.get('data.source') == 'tushare'
        
        # 模型配置
        assert len(settings.models) > 0
        assert settings.default_model == 'test_model'
        
        # 共享配置
        shared = settings.shared_config
        assert 'prediction' in shared

