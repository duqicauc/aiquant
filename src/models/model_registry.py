"""
模型注册表 - 管理多个模型

每个模型都有独立的：
- 数据样本目录
- 训练流程
- 评测流程
- 模型文件存储
"""
import os
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
import json


@dataclass
class ModelConfig:
    """模型配置"""
    name: str  # 模型名称（如 'test_model'）
    display_name: str  # 显示名称（如 '测试模型'）
    description: str  # 模型描述
    data_dir: str  # 数据目录（相对于 data/training/models/）
    model_dir: str  # 模型文件目录（相对于 data/training/models/）
    sample_dir: str  # 样本目录（相对于 data/training/models/）
    metrics_dir: str  # 评测结果目录（相对于 data/training/models/）
    prediction_dir: str  # 预测结果目录（相对于 data/prediction/models/）


class ModelRegistry:
    """模型注册表"""
    
    _models: Dict[str, ModelConfig] = {}
    
    @classmethod
    def register(cls, config: ModelConfig):
        """注册模型"""
        cls._models[config.name] = config
        # 确保目录存在
        cls._ensure_directories(config)
    
    @classmethod
    def get(cls, model_name: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        return cls._models.get(model_name)
    
    @classmethod
    def list_all(cls) -> Dict[str, ModelConfig]:
        """列出所有注册的模型"""
        return cls._models.copy()
    
    @classmethod
    def _ensure_directories(cls, config: ModelConfig):
        """确保模型相关目录存在"""
        project_root = Path(__file__).parent.parent.parent
        
        # 训练相关目录
        training_base = project_root / 'data' / 'training' / 'models' / config.data_dir
        (training_base / config.sample_dir).mkdir(parents=True, exist_ok=True)
        (training_base / config.model_dir).mkdir(parents=True, exist_ok=True)
        (training_base / config.metrics_dir).mkdir(parents=True, exist_ok=True)
        
        # 预测相关目录
        prediction_base = project_root / 'data' / 'prediction' / 'models' / config.name
        prediction_base.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_paths(cls, model_name: str) -> Dict[str, Path]:
        """获取模型的所有路径"""
        config = cls.get(model_name)
        if not config:
            raise ValueError(f"模型 {model_name} 未注册")
        
        project_root = Path(__file__).parent.parent.parent
        training_base = project_root / 'data' / 'training' / 'models' / config.data_dir
        prediction_base = project_root / 'data' / 'prediction' / 'models' / config.name
        
        return {
            'samples': training_base / config.sample_dir,
            'models': training_base / config.model_dir,
            'metrics': training_base / config.metrics_dir,
            'predictions': prediction_base,
            'training_base': training_base,
        }
    
    @classmethod
    def save_metadata(cls, model_name: str, metadata: dict):
        """保存模型元数据"""
        config = cls.get(model_name)
        if not config:
            raise ValueError(f"模型 {model_name} 未注册")
        
        paths = cls.get_model_paths(model_name)
        metadata_file = paths['models'] / 'model_metadata.json'
        
        metadata['model_name'] = model_name
        metadata['display_name'] = config.display_name
        metadata['description'] = config.description
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_metadata(cls, model_name: str) -> Optional[dict]:
        """加载模型元数据"""
        paths = cls.get_model_paths(model_name)
        metadata_file = paths['models'] / 'model_metadata.json'
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)


# 注意：实际使用的模型：
# 1. xgboost_timeseries - 由 scripts/score_current_stocks.py 使用（效果最好）
# 2. left_breakout - 左侧起爆点模型，完整实现（不在 ModelRegistry 中注册）

