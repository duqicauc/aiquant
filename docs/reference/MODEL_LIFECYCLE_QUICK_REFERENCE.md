# 模型生命周期快速参考

## 🚀 快速流程图

### 新建模型流程

```
┌─────────────┐
│ 需求分析    │ → 创建需求文档
└─────────────┘
      ↓
┌─────────────┐
│ 模型设计    │ → 创建设计文档
└─────────────┘
      ↓
┌─────────────┐
│ 注册模型    │ → config/models/{name}.yaml + ModelRegistry.register()
└─────────────┘
      ↓
┌─────────────┐
│ 准备数据    │ → prepare_positive_samples.py + prepare_negative_samples_v2.py
└─────────────┘
      ↓
┌─────────────┐
│ 训练v1.0.0  │ → ModelTrainer.train_version('v1.0.0')
└─────────────┘
      ↓
┌─────────────┐
│ 评估        │ → 生成评估报告
└─────────────┘
      ↓
┌─────────────┐
│ 部署        │ → status='production'
└─────────────┘
```

### 迭代模型流程

```
┌─────────────┐
│ 问题分析    │ → 创建变更日志
└─────────────┘
      ↓
┌─────────────┐
│ 版本规划    │ → 确定版本号（v1.1.0 / v2.0.0 / v1.0.1）
└─────────────┘
      ↓
┌─────────────┐
│ 创建版本    │ → ModelIterator.create_version('v1.1.0', base_version='v1.0.0')
└─────────────┘
      ↓
┌─────────────┐
│ 训练        │ → ModelTrainer.train_version('v1.1.0')
└─────────────┘
      ↓
┌─────────────┐
│ 对比评估    │ → 对比新旧版本指标
└─────────────┘
      ↓
┌─────────────┐
│ 决策        │ → 升级（production）或 回滚（deprecated）
└─────────────┘
```

---

## 📋 命令速查

### 新建模型

```bash
# 1. 创建配置文件
# 编辑 config/models/{model_name}.yaml

# 2. 注册模型（在代码中）
from src.models.model_registry import ModelRegistry, ModelConfig
config = ModelConfig(...)
ModelRegistry.register(config)

# 3. 准备数据
python scripts/prepare_positive_samples.py
python scripts/prepare_negative_samples_v2.py

# 4. 训练v1.0.0
python scripts/train_xgboost_timeseries.py --model {model_name} --version v1.0.0

# 5. 标记为生产版本
python -c "
from src.models.lifecycle.iterator import ModelIterator
iterator = ModelIterator('{model_name}')
iterator.update_version_metadata('v1.0.0', status='production')
"
```

### 迭代模型

```bash
# 1. 创建新版本
python -c "
from src.models.lifecycle.iterator import ModelIterator
iterator = ModelIterator('{model_name}')
iterator.create_version('v1.1.0', base_version='v1.0.0', changes={'description': '...'})
"

# 2. 训练新版本
python scripts/train_xgboost_timeseries.py --model {model_name} --version v1.1.0

# 3. 对比评估
python -c "
from src.models.lifecycle.iterator import ModelIterator
iterator = ModelIterator('{model_name}')
v1 = iterator.get_version_info('v1.0.0')
v2 = iterator.get_version_info('v1.1.0')
print(f'v1.0.0 AUC: {v1[\"metrics\"][\"test\"][\"auc\"]}')
print(f'v1.1.0 AUC: {v2[\"metrics\"][\"test\"][\"auc\"]}')
"

# 4. 升级（如果新版本更好）
python -c "
from src.models.lifecycle.iterator import ModelIterator
iterator = ModelIterator('{model_name}')
iterator.update_version_metadata('v1.1.0', status='production')
iterator.update_version_metadata('v1.0.0', status='deprecated')
"
```

---

## 🎯 版本号规范

| 变更类型 | 版本号 | 示例 | 说明 |
|---------|--------|------|------|
| 重大架构变更 | vX.0.0 | v2.0.0 | 算法更换、架构重构 |
| 新功能/重要特征 | v1.X.0 | v1.1.0 | 新增特征、重要改进 |
| 参数调优/bug修复 | v1.0.X | v1.0.1 | 超参数调整、bug修复 |

---

## 📁 目录结构

```
data/models/{model_name}/
├── config.yaml                    # 模型基础配置
├── versions/                      # 版本目录
│   ├── v1.0.0/                   # 版本1.0.0
│   │   ├── metadata.json         # 版本元数据
│   │   ├── model/                # 模型文件
│   │   │   ├── model.json
│   │   │   └── feature_names.json
│   │   ├── training/             # 训练相关
│   │   │   └── metrics.json
│   │   └── evaluation/           # 评估相关
│   └── v1.1.0/                   # 版本1.1.0
│       └── ...
└── ...

docs/models/{model_name}/
├── requirements.md               # 需求文档
├── design.md                     # 设计文档
├── evaluation/                   # 评估报告
│   ├── v1.0.0.md
│   └── v1.1.0.md
└── changelog/                     # 变更日志
    ├── v1.1.0.md
    └── v2.0.0.md
```

---

## ✅ 检查清单

### 新建模型检查清单

- [ ] 需求文档已创建 (`docs/models/{name}/requirements.md`)
- [ ] 设计文档已创建 (`docs/models/{name}/design.md`)
- [ ] 配置文件已创建 (`config/models/{name}.yaml`)
- [ ] 模型已注册 (`ModelRegistry.register()`)
- [ ] 数据已准备（正样本 + 负样本）
- [ ] 数据质量检查通过
- [ ] v1.0.0 训练完成
- [ ] 训练指标达到预期
- [ ] 评估报告已生成
- [ ] 版本状态已标记为 `production`

### 迭代模型检查清单

- [ ] 问题分析完成
- [ ] 变更日志已创建 (`docs/models/{name}/changelog/{version}.md`)
- [ ] 版本号已确定
- [ ] 新版本已创建
- [ ] 新版本训练完成
- [ ] 版本对比完成
- [ ] 决策已做出（升级/回滚）
- [ ] 版本状态已更新

---

## 🔧 常用代码片段

### 创建新版本

```python
from src.models.lifecycle.iterator import ModelIterator

iterator = ModelIterator('{model_name}')
new_version = iterator.create_version(
    version='v1.1.0',
    base_version='v1.0.0',
    changes={
        'type': 'feature',
        'description': '新增OBV和KDJ指标',
        'impact': 'medium'
    },
    created_by='your_name'
)
```

### 训练版本

```python
from src.models.lifecycle.trainer import ModelTrainer

trainer = ModelTrainer('{model_name}')
model, metrics = trainer.train_version(version='v1.1.0')
```

### 查看版本信息

```python
from src.models.lifecycle.iterator import ModelIterator

iterator = ModelIterator('{model_name}')
info = iterator.get_version_info('v1.0.0')
print(f"AUC: {info['metrics']['test']['auc']}")
print(f"状态: {info['status']}")
```

### 列出所有版本

```python
from src.models.lifecycle.iterator import ModelIterator

iterator = ModelIterator('{model_name}')
versions = iterator.list_versions()
print(f"所有版本: {versions}")

# 只列出生产版本
production_versions = iterator.list_versions(status='production')
print(f"生产版本: {production_versions}")
```

### 更新版本状态

```python
from src.models.lifecycle.iterator import ModelIterator

iterator = ModelIterator('{model_name}')

# 标记为生产版本
iterator.update_version_metadata('v1.1.0', status='production')

# 标记旧版本为已废弃
iterator.update_version_metadata('v1.0.0', status='deprecated')
```

---

## 📚 详细文档

完整流程请参考：[模型生命周期标准化流程](MODEL_LIFECYCLE_STANDARD.md)
