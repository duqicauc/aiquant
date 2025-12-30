"""
统一配置管理模块

功能：
- 加载全局配置（settings.yaml）
- 加载多模型配置（models.yaml）
- 加载单个模型配置（config/models/{model_name}.yaml）
- 提供便捷的配置访问接口
"""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List


class Settings:
    """统一配置管理类"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_file: 配置文件路径，默认使用 settings.yaml
        """
        self.config_dir = Path(__file__).parent
        
        if config_file is None:
            config_file = self.config_dir / 'settings.yaml'
        
        self.config_file = Path(config_file)
        self._config = {}
        self._models_config = {}
        self._model_configs_cache = {}  # 缓存已加载的模型配置
        
        if self.config_file.exists():
            self.load()
        
        # 加载多模型配置
        self._load_models_config()
    
    def load(self):
        """加载配置文件"""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f) or {}
    
    def _load_models_config(self):
        """加载多模型配置"""
        models_file = self.config_dir / 'models.yaml'
        if models_file.exists():
            with open(models_file, 'r', encoding='utf-8') as f:
                self._models_config = yaml.safe_load(f) or {}
    
    def save(self):
        """保存配置文件"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, allow_unicode=True, default_flow_style=False)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置项
        
        Args:
            key_path: 配置路径，如 'data.sample_preparation.start_date'
            default: 默认值
        
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """
        设置配置项
        
        Args:
            key_path: 配置路径，如 'data.sample_preparation.start_date'
            value: 配置值
        """
        keys = key_path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    # =========================================================================
    # 全局配置属性
    # =========================================================================
    
    @property
    def data(self) -> Dict:
        """数据配置"""
        return self._config.get('data', {})
    
    @property
    def model(self) -> Dict:
        """模型配置（全局默认）"""
        return self._config.get('model', {})
    
    @property
    def prediction(self) -> Dict:
        """预测配置"""
        return self._config.get('prediction', {})
    
    @property
    def review(self) -> Dict:
        """回顾配置"""
        return self._config.get('review', {})
    
    @property
    def automation(self) -> Dict:
        """自动化配置"""
        return self._config.get('automation', {})
    
    @property
    def logging(self) -> Dict:
        """日志配置"""
        return self._config.get('logging', {})
    
    @property
    def data_storage(self) -> Dict:
        """数据存储配置"""
        return self._config.get('data_storage', {})
    
    # =========================================================================
    # 多模型配置
    # =========================================================================
    
    @property
    def models(self) -> Dict:
        """获取所有注册的模型"""
        return self._models_config.get('models', {})
    
    @property
    def default_model(self) -> str:
        """获取默认模型名称"""
        return self._models_config.get('default_model', 'breakout_launch_scorer')
    
    @property
    def models_root(self) -> str:
        """模型存储根目录"""
        return self._models_config.get('models_root', 'data/models')
    
    @property
    def version_management(self) -> Dict:
        """版本管理配置"""
        return self._models_config.get('version_management', {})
    
    @property
    def shared_config(self) -> Dict:
        """共享配置（所有模型共用）"""
        return self._models_config.get('shared', {})
    
    def list_models(self) -> List[str]:
        """列出所有注册的模型名称"""
        return list(self.models.keys())
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """获取模型基本信息（从 models.yaml）"""
        return self.models.get(model_name)
    
    def get_model_config(self, model_name: str) -> Dict:
        """
        获取完整的模型配置（加载模型独立配置文件）
        
        Args:
            model_name: 模型名称
        
        Returns:
            完整的模型配置（合并共享配置和模型独立配置）
        """
        # 检查缓存
        if model_name in self._model_configs_cache:
            return self._model_configs_cache[model_name]
        
        # 获取模型基本信息
        model_info = self.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"模型 {model_name} 未注册，请在 config/models.yaml 中添加")
        
        # 加载模型独立配置文件
        config_file = model_info.get('config_file')
        if config_file:
            config_path = Path(config_file)
            if not config_path.is_absolute():
                # 相对于项目根目录
                config_path = self.config_dir.parent / config_file
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    model_config = yaml.safe_load(f) or {}
            else:
                model_config = {}
        else:
            model_config = {}
        
        # 合并配置：共享配置 < 模型配置 < 覆盖配置
        merged = self._deep_merge(
            self.shared_config.copy(),
            model_config
        )
        
        # 应用覆盖配置
        overrides = model_info.get('overrides', {})
        if overrides:
            merged = self._deep_merge(merged, overrides)
        
        # 添加元信息
        merged['_model_name'] = model_name
        merged['_display_name'] = model_info.get('display_name', model_name)
        merged['_status'] = model_info.get('status', 'active')
        
        # 缓存
        self._model_configs_cache[model_name] = merged
        
        return merged
    
    def get_model_path(self, model_name: str) -> Path:
        """获取模型存储路径"""
        return Path(self.models_root) / model_name
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """深度合并两个字典"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def __repr__(self):
        return f"Settings(config_file={self.config_file}, models={len(self.models)})"


# =========================================================================
# 全局配置实例
# =========================================================================

try:
    settings = Settings()
except Exception as e:
    # 如果配置文件不存在，使用空配置
    print(f"Warning: Failed to load settings: {e}")
    settings = Settings.__new__(Settings)
    settings._config = {}
    settings._models_config = {}
    settings._model_configs_cache = {}


# =========================================================================
# 便捷函数
# =========================================================================

def get_model_config(model_name: str = None) -> Dict:
    """
    获取模型配置的便捷函数
    
    Args:
        model_name: 模型名称，None 表示使用默认模型
    
    Returns:
        模型配置字典
    """
    if model_name is None:
        model_name = settings.default_model
    return settings.get_model_config(model_name)


def get_setting(key_path: str, default: Any = None) -> Any:
    """
    获取全局配置的便捷函数
    
    Args:
        key_path: 配置路径
        default: 默认值
    
    Returns:
        配置值
    """
    return settings.get(key_path, default)


# =========================================================================
# 测试
# =========================================================================

if __name__ == '__main__':
    print("="*80)
    print("配置系统测试")
    print("="*80)
    
    s = Settings()
    
    # 全局配置
    print(f"\n📋 全局配置:")
    print(f"  数据起始日期: {s.get('data.sample_preparation.start_date')}")
    print(f"  推荐股票数: {s.get('prediction.scoring.top_n')}")
    
    # 模型配置
    print(f"\n📦 已注册模型:")
    for name in s.list_models():
        info = s.get_model_info(name)
        print(f"  - {name}: {info.get('display_name', '-')} [{info.get('status', '-')}]")
    
    print(f"\n🔧 默认模型: {s.default_model}")
    
    # 加载模型完整配置
    if s.list_models():
        model_name = s.default_model
        try:
            config = s.get_model_config(model_name)
            print(f"\n📊 {model_name} 配置:")
            print(f"  类型: {config.get('model', {}).get('type', '-')}")
            print(f"  Top N: {config.get('prediction', {}).get('top_n', '-')}")
        except Exception as e:
            print(f"  加载失败: {e}")
    
    print("\n✅ 配置系统测试完成")
