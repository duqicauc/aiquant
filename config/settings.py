"""
统一配置管理模块
"""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Settings:
    """统一配置管理类"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_file: 配置文件路径，默认使用 settings.yaml
        """
        if config_file is None:
            config_file = Path(__file__).parent / 'settings.yaml'
        
        self.config_file = Path(config_file)
        self._config = {}
        
        if self.config_file.exists():
            self.load()
    
    def load(self):
        """加载配置文件"""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f) or {}
    
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
    
    @property
    def data(self) -> Dict:
        """数据配置"""
        return self._config.get('data', {})
    
    @property
    def model(self) -> Dict:
        """模型配置"""
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
    
    def __repr__(self):
        return f"Settings(config_file={self.config_file})"


# 全局配置实例
try:
    settings = Settings()
except Exception as e:
    # 如果配置文件不存在，使用空配置
    print(f"Warning: Failed to load settings: {e}")
    settings = Settings.__new__(Settings)
    settings._config = {}


if __name__ == '__main__':
    # 测试配置加载
    print("="*80)
    print("配置文件测试")
    print("="*80)
    
    s = Settings()
    
    print(f"\n数据起始日期: {s.get('data.sample_preparation.start_date')}")
    print(f"模型版本: {s.get('model.version')}")
    print(f"推荐股票数: {s.get('prediction.scoring.top_n')}")
    print(f"回顾周期: {s.get('review.periods')}")
    
    print("\n✓ 配置加载成功")

