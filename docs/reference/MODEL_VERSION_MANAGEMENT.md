# 模型版本管理指南

## 📋 概述

本文档详细说明如何管理同一个模型的不同版本，包括版本创建、存储、对比、升级和回滚等完整流程。

---

## 🏗️ 版本管理架构

### 1. 目录结构

```
data/models/{model_name}/
├── config.yaml                    # 模型基础配置
├── current_version.txt            # 当前使用的版本（符号链接或文本）
│
├── versions/                      # 版本目录
│   ├── v1.0/                      # 版本1.0
│   │   ├── metadata.json          # 版本元数据
│   │   ├── config.yaml            # 版本特定配置
│   │   ├── model/                 # 模型文件
│   │   │   ├── model.joblib       # 模型文件
│   │   │   ├── feature_names.json # 特征名称
│   │   │   └── scaler.pkl         # 特征缩放器（如有）
│   │   ├── training/              # 训练相关
│   │   │   ├── metrics.json       # 训练指标
│   │   │   ├── logs/              # 训练日志
│   │   │   ├── checkpoints/       # 检查点
│   │   │   └── training_config.json
│   │   ├── evaluation/            # 评测相关
│   │   │   ├── validation_metrics.json
│   │   │   ├── backtest_results.json
│   │   │   └── reports/           # 评测报告
│   │   └── experiments/           # 实验记录
│   │       └── experiment_001.json
│   │
│   ├── v1.1/                      # 版本1.1
│   │   └── ...                    # 同上结构
│   │
│   ├── v2.0/                      # 版本2.0（主版本升级）
│   │   └── ...                    # 同上结构
│   │
│   └── v2.0.1/                    # 版本2.0.1（补丁版本）
│       └── ...                    # 同上结构
│
├── staging/                        # 测试版本（可选）
│   └── v2.1-beta/                 # 测试版本
│       └── ...
│
└── production/                    # 生产版本（符号链接）
    └── -> versions/v2.0/          # 指向当前生产版本
```

### 2. 版本命名规则

#### 语义化版本（Semantic Versioning）

```
主版本号.次版本号.补丁版本号[-标识符]

示例：
- v1.0.0      # 初始版本
- v1.0.1      # 补丁版本（bug修复）
- v1.1.0      # 次版本（新功能）
- v2.0.0      # 主版本（重大变更）
- v2.1.0-beta # 测试版本
- v2.1.0-rc1  # 候选版本
```

#### 版本号含义

| 版本类型 | 说明 | 示例 |
|---------|------|------|
| **主版本号** | 不兼容的API变更或重大架构调整 | v1.0.0 → v2.0.0 |
| **次版本号** | 向后兼容的功能新增 | v1.0.0 → v1.1.0 |
| **补丁版本号** | 向后兼容的问题修复 | v1.0.0 → v1.0.1 |
| **标识符** | 预发布版本标识 | v1.1.0-alpha, v1.1.0-beta, v1.1.0-rc1 |

#### 特殊版本

- `latest` - 最新版本（自动指向最新版本号）
- `production` - 生产版本（指向当前生产环境使用的版本）
- `staging` - 测试版本（指向当前测试环境使用的版本）

---

## 📝 版本元数据

### metadata.json 结构

```json
{
  "version": "v1.0.0",
  "model_name": "xgboost_timeseries",
  "display_name": "XGBoost时间序列模型 v1.0.0",
  "description": "基于XGBoost的时间序列选股模型，使用34天回看窗口",

  "created_at": "2025-12-28T10:00:00Z",
  "created_by": "user@example.com",
  "parent_version": null,  // 如果是基于某个版本创建，记录父版本

  "status": "production",  // development, testing, staging, production, archived
  "tags": ["stable", "high-accuracy"],

  "config": {
    "data": {
      "sample_preparation": {
        "positive_criteria": {
          "consecutive_weeks": 3,
          "total_return_threshold": 50
        }
      },
      "feature_extraction": {
        "lookback_days": 34,
        "extractors": ["technical.ma", "technical.macd", "technical.rsi"]
      }
    },
    "model_params": {
      "objective": "binary:logistic",
      "n_estimators": 100,
      "learning_rate": 0.1,
      "max_depth": 5
    }
  },

  "training": {
    "started_at": "2025-12-28T10:00:00Z",
    "completed_at": "2025-12-28T11:30:00Z",
    "duration_seconds": 5400,
    "samples": {
      "train": 5000,
      "validation": 1000,
      "test": 500
    },
    "hyperparameters": {
      "n_estimators": 100,
      "learning_rate": 0.1,
      "max_depth": 5
    }
  },

  "metrics": {
    "training": {
      "accuracy": 0.89,
      "precision": 0.85,
      "recall": 0.88,
      "f1": 0.86,
      "auc": 0.92
    },
    "validation": {
      "accuracy": 0.77,
      "precision": 0.72,
      "recall": 0.75,
      "f1": 0.73,
      "auc": 0.81
    },
    "test": {
      "accuracy": 0.76,
      "precision": 0.71,
      "recall": 0.74,
      "f1": 0.72,
      "auc": 0.80
    }
  },

  "backtest": {
    "period": {
      "start": "2023-01-01",
      "end": "2024-12-31"
    },
    "metrics": {
      "annual_return": 0.18,
      "sharpe_ratio": 1.65,
      "max_drawdown": -0.12,
      "win_rate": 0.68
    }
  },

  "changes": [
    {
      "type": "feature",  // feature, parameter, bugfix, performance
      "description": "新增OBV和KDJ指标",
      "impact": "medium"  // low, medium, high
    },
    {
      "type": "parameter",
      "description": "调整n_estimators从80增加到100",
      "impact": "low"
    }
  ],

  "notes": "首次稳定版本，经过充分测试",
  "deprecated": false,
  "deprecation_date": null,
  "replacement_version": null
}
```

---

## 🔄 版本管理操作

### 1. 创建新版本

```python
from src.models.lifecycle.iterator import ModelIterator

iterator = ModelIterator('xgboost_timeseries')

# 创建新版本（基于当前版本）
new_version = iterator.create_version(
    base_version='v1.0.0',  # 基于哪个版本
    version='v1.1.0',      # 新版本号
    changes={
        'features': ['added_obv', 'added_kdj'],
        'parameters': {'n_estimators': 150},
        'description': '新增OBV和KDJ指标，优化模型参数'
    },
    created_by='user@example.com'
)

# 或者创建全新版本（不基于任何版本）
new_version = iterator.create_version(
    version='v2.0.0',
    changes={
        'architecture': 'major_refactor',
        'description': '重大架构调整，使用新的特征提取方法'
    }
)
```

### 2. 训练版本

```python
from src.models.lifecycle.trainer import ModelTrainer

trainer = ModelTrainer('xgboost_timeseries')

# 训练指定版本
result = trainer.train_version(
    version='v1.1.0',
    samples_path='data/models/xgboost_timeseries/data/samples/train.csv',
    validation_split=0.2
)

# 自动保存版本元数据
trainer.save_version_metadata(
    version='v1.1.0',
    metrics=result.metrics,
    training_info=result.training_info
)
```

### 3. 版本对比

```python
from src.models.lifecycle.iterator import ModelIterator

iterator = ModelIterator('xgboost_timeseries')

# 对比两个版本
comparison = iterator.compare_versions(
    version1='v1.0.0',
    version2='v1.1.0',
    metrics=['accuracy', 'precision', 'recall', 'f1', 'auc', 'sharpe_ratio']
)

print(comparison)
# {
#     'v1.0.0': {
#         'accuracy': 0.76,
#         'precision': 0.71,
#         'recall': 0.74,
#         'f1': 0.72,
#         'auc': 0.80,
#         'sharpe_ratio': 1.65
#     },
#     'v1.1.0': {
#         'accuracy': 0.78,
#         'precision': 0.73,
#         'recall': 0.76,
#         'f1': 0.74,
#         'auc': 0.82,
#         'sharpe_ratio': 1.72
#     },
#     'improvement': {
#         'accuracy': +0.02,
#         'precision': +0.02,
#         'recall': +0.02,
#         'f1': +0.02,
#         'auc': +0.02,
#         'sharpe_ratio': +0.07
#     },
#     'winner': 'v1.1.0'
# }
```

### 4. 版本升级

```python
from src.models.lifecycle.iterator import ModelIterator

iterator = ModelIterator('xgboost_timeseries')

# 将版本升级到生产环境
iterator.promote_version(
    version='v1.1.0',
    environment='production',  # production, staging
    reason='性能提升2%，Sharpe比率提升0.07'
)

# 检查是否可以升级
can_promote = iterator.can_promote(
    version='v1.1.0',
    environment='production'
)
# 返回: {'can_promote': True, 'reason': '所有测试通过'}
```

### 5. 版本回滚

```python
from src.models.lifecycle.iterator import ModelIterator

iterator = ModelIterator('xgboost_timeseries')

# 回滚到指定版本
iterator.rollback_version(
    from_version='v1.1.0',
    to_version='v1.0.0',
    environment='production',
    reason='v1.1.0在生产环境表现不佳'
)
```

### 6. 版本列表和查询

```python
from src.models.lifecycle.iterator import ModelIterator

iterator = ModelIterator('xgboost_timeseries')

# 列出所有版本
versions = iterator.list_versions()
# ['v1.0.0', 'v1.0.1', 'v1.1.0', 'v2.0.0']

# 获取版本信息
version_info = iterator.get_version_info('v1.1.0')

# 获取当前生产版本
production_version = iterator.get_production_version()
# 'v1.1.0'

# 获取最新版本
latest_version = iterator.get_latest_version()
# 'v2.0.0'

# 搜索版本
versions = iterator.search_versions(
    status='production',
    tags=['stable'],
    min_accuracy=0.75
)
```

---

## 📊 版本状态管理

### 状态流转

```
development → testing → staging → production
     ↓           ↓         ↓          ↓
   archived   archived  archived  archived
```

### 状态说明

| 状态 | 说明 | 使用场景 |
|------|------|---------|
| **development** | 开发中 | 正在开发或调试的版本 |
| **testing** | 测试中 | 完成开发，正在进行测试 |
| **staging** | 预发布 | 测试通过，准备发布到生产环境 |
| **production** | 生产环境 | 正在生产环境使用的版本 |
| **archived** | 已归档 | 不再使用的旧版本 |

### 状态管理操作

```python
# 更新版本状态
iterator.update_version_status(
    version='v1.1.0',
    status='production',
    reason='测试通过，性能提升'
)

# 归档版本
iterator.archive_version(
    version='v1.0.0',
    reason='已被v1.1.0替代'
)
```

---

## 🔍 版本选择策略

### 1. 自动选择最新版本

```python
from src.models.lifecycle.predictor import ModelPredictor

predictor = ModelPredictor('xgboost_timeseries')

# 自动使用最新生产版本
predictions = predictor.predict(
    version='production',  # 或 'latest'
    stock_data=data
)
```

### 2. 基于性能选择

```python
# 选择性能最好的版本
best_version = iterator.get_best_version(
    metric='sharpe_ratio',
    min_status='testing'  # 至少是测试状态
)
```

### 3. 基于标签选择

```python
# 选择带有特定标签的版本
stable_version = iterator.get_version_by_tag(
    tag='stable',
    status='production'
)
```

### 4. 基于日期选择

```python
# 选择特定日期之前的最新版本
version = iterator.get_version_by_date(
    before_date='2025-12-01',
    status='production'
)
```

---

## 📈 版本历史追踪

### 版本树

```
v1.0.0 (production)
  ├── v1.0.1 (archived) - bugfix
  └── v1.1.0 (production) - new features
      ├── v1.1.1 (testing) - bugfix
      └── v2.0.0 (staging) - major refactor
          └── v2.0.1 (development) - bugfix
```

### 查看版本历史

```python
# 获取版本树
version_tree = iterator.get_version_tree()

# 获取版本变更历史
history = iterator.get_version_history('v2.0.0')
# [
#     {'version': 'v1.0.0', 'action': 'created', 'date': '2025-12-01'},
#     {'version': 'v1.1.0', 'action': 'created', 'date': '2025-12-15', 'parent': 'v1.0.0'},
#     {'version': 'v2.0.0', 'action': 'created', 'date': '2025-12-28', 'parent': 'v1.1.0'}
# ]
```

---

## 🛠️ 实现示例

### ModelIterator 类实现

```python
# src/models/lifecycle/iterator.py

from pathlib import Path
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class VersionMetadata:
    """版本元数据"""
    version: str
    model_name: str
    status: str
    created_at: str
    created_by: str
    parent_version: Optional[str]
    metrics: Dict
    changes: List[Dict]
    # ... 其他字段

class ModelIterator:
    """模型迭代器 - 管理模型版本"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_path = Path(f"data/models/{model_name}")
        self.versions_path = self.base_path / "versions"
        self.versions_path.mkdir(parents=True, exist_ok=True)

    def create_version(
        self,
        version: str,
        base_version: Optional[str] = None,
        changes: Dict = None,
        created_by: str = None
    ) -> str:
        """创建新版本"""
        version_path = self.versions_path / version
        version_path.mkdir(parents=True, exist_ok=True)

        # 创建版本目录结构
        (version_path / "model").mkdir(exist_ok=True)
        (version_path / "training").mkdir(exist_ok=True)
        (version_path / "evaluation").mkdir(exist_ok=True)
        (version_path / "experiments").mkdir(exist_ok=True)

        # 创建元数据
        metadata = VersionMetadata(
            version=version,
            model_name=self.model_name,
            status='development',
            created_at=datetime.now().isoformat(),
            created_by=created_by or 'system',
            parent_version=base_version,
            metrics={},
            changes=changes or []
        )

        # 保存元数据
        self._save_metadata(version, metadata)

        # 如果基于某个版本，复制配置
        if base_version:
            self._copy_base_config(version, base_version)

        return version

    def get_version_info(self, version: str) -> Dict:
        """获取版本信息"""
        metadata_path = self.versions_path / version / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"版本 {version} 不存在")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def compare_versions(
        self,
        version1: str,
        version2: str,
        metrics: List[str] = None
    ) -> Dict:
        """对比两个版本"""
        info1 = self.get_version_info(version1)
        info2 = self.get_version_info(version2)

        metrics = metrics or ['accuracy', 'precision', 'recall', 'f1', 'auc']

        comparison = {
            version1: {},
            version2: {},
            'improvement': {},
            'winner': None
        }

        best_score = -1
        for metric in metrics:
            val1 = info1.get('metrics', {}).get('test', {}).get(metric, 0)
            val2 = info2.get('metrics', {}).get('test', {}).get(metric, 0)

            comparison[version1][metric] = val1
            comparison[version2][metric] = val2
            comparison['improvement'][metric] = val2 - val1

            # 判断哪个版本更好（使用加权平均）
            if metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                if val2 > best_score:
                    best_score = val2
                    comparison['winner'] = version2
                elif val1 > best_score:
                    best_score = val1
                    comparison['winner'] = version1

        return comparison

    def promote_version(
        self,
        version: str,
        environment: str = 'production',
        reason: str = None
    ):
        """升级版本到指定环境"""
        info = self.get_version_info(version)

        # 更新状态
        info['status'] = environment
        info['promoted_at'] = datetime.now().isoformat()
        info['promotion_reason'] = reason

        self._save_metadata(version, info)

        # 创建符号链接
        if environment == 'production':
            production_link = self.base_path / "production"
            if production_link.exists():
                production_link.unlink()
            production_link.symlink_to(f"versions/{version}")

    def list_versions(
        self,
        status: str = None,
        tags: List[str] = None
    ) -> List[str]:
        """列出所有版本"""
        versions = []
        for version_dir in self.versions_path.iterdir():
            if version_dir.is_dir():
                try:
                    info = self.get_version_info(version_dir.name)
                    if status and info.get('status') != status:
                        continue
                    if tags and not any(tag in info.get('tags', []) for tag in tags):
                        continue
                    versions.append(version_dir.name)
                except:
                    continue

        # 按版本号排序
        versions.sort(key=lambda v: self._version_key(v))
        return versions

    def _version_key(self, version: str) -> tuple:
        """将版本号转换为可排序的元组"""
        # 移除 'v' 前缀和标识符
        version = version.lstrip('v')
        if '-' in version:
            version = version.split('-')[0]

        parts = version.split('.')
        return tuple(int(p) if p.isdigit() else 0 for p in parts)

    def _save_metadata(self, version: str, metadata):
        """保存元数据"""
        metadata_path = self.versions_path / version / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(metadata) if hasattr(metadata, '__dict__') else metadata,
                     f, indent=2, ensure_ascii=False)
```

---

## 📋 使用示例

### 完整工作流

```python
from src.models.lifecycle.iterator import ModelIterator
from src.models.lifecycle.trainer import ModelTrainer
from src.models.lifecycle.evaluator import ModelEvaluator

# 1. 创建迭代器
iterator = ModelIterator('xgboost_timeseries')

# 2. 创建新版本
new_version = iterator.create_version(
    base_version='v1.0.0',
    version='v1.1.0',
    changes={
        'features': ['added_obv', 'added_kdj'],
        'parameters': {'n_estimators': 150}
    }
)

# 3. 训练新版本
trainer = ModelTrainer('xgboost_timeseries')
trainer.train_version('v1.1.0')

# 4. 评测新版本
evaluator = ModelEvaluator('xgboost_timeseries')
evaluator.evaluate_version('v1.1.0')

# 5. 对比版本
comparison = iterator.compare_versions('v1.0.0', 'v1.1.0')

# 6. 如果新版本更好，升级到生产环境
if comparison['winner'] == 'v1.1.0':
    iterator.promote_version(
        version='v1.1.0',
        environment='production',
        reason=f"性能提升: {comparison['improvement']}"
    )
```

---

## 🎯 最佳实践

1. **版本命名**：使用语义化版本号，清晰表达版本变更
2. **版本记录**：详细记录每个版本的变更和原因
3. **版本测试**：新版本必须经过充分测试才能升级
4. **版本对比**：升级前必须对比新旧版本性能
5. **版本回滚**：保留旧版本，支持快速回滚
6. **版本归档**：定期归档不再使用的旧版本

---

**文档版本**: v1.0
**创建日期**: 2025-12-28
**最后更新**: 2025-12-28
