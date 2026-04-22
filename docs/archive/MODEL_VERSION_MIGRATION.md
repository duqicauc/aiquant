# 模型版本迁移和调整说明

## 📋 概述

本文档说明了对现有模型版本 `breakout_launch_scorer v1.0.0` 的调整，使其完全符合最新的模型训练框架要求，方便后续进行版本迭代。

---

## ✅ 已完成的工作

### 1. 检查现有模型版本

- ✅ 确认 `v1.0.0` 版本存在且文件完整
  - 模型文件: `data/models/breakout_launch_scorer/versions/v1.0.0/model/model.json`
  - 特征名称: `data/models/breakout_launch_scorer/versions/v1.0.0/model/feature_names.json`
  - 训练指标: `data/models/breakout_launch_scorer/versions/v1.0.0/training/metrics.json`
  - 版本元数据: `data/models/breakout_launch_scorer/versions/v1.0.0/metadata.json`

### 2. 更新版本元数据

更新了 `v1.0.0` 版本的元数据文件，补充了新框架要求的字段：

#### 新增字段：
- `display_name`: 模型显示名称
- `description`: 模型描述
- `config`: 完整的配置信息（包含数据配置、模型参数、训练配置、预测配置、评测配置）
- `tags`: 标签列表
- `deprecated`: 是否已弃用
- `deprecation_date`: 弃用日期
- `replacement_version`: 替代版本

#### 优化字段：
- `metrics`: 重新组织为 `training`、`validation`、`test` 三个层级
- `training`: 补充了 `completed_at`、`duration_seconds`、`hyperparameters`、日期范围等信息
- `changes`: 从字典改为列表格式，包含变更类型、描述和影响

### 3. 更新训练器

更新了 `src/models/lifecycle/trainer.py` 中的 `train_version` 方法，确保：

- ✅ 保存完整的配置信息到元数据
- ✅ 保存模型显示名称和描述
- ✅ 正确组织 metrics 结构（training/validation/test）
- ✅ 保存完整的训练信息（包括超参数、日期范围等）
- ✅ 记录训练开始和结束时间

### 4. 训练脚本兼容性

- ✅ `scripts/train_breakout_launch_scorer.py` 已使用新框架
- ✅ 自动版本号递增功能正常
- ✅ 自动创建版本目录结构

### 5. 创建验证工具

创建了 `scripts/check_model_versions.py` 脚本，用于：
- 列出所有模型版本
- 显示每个版本的详细信息
- 验证版本管理功能是否正常

---

## 📁 文件结构

```
data/models/breakout_launch_scorer/
├── versions/
│   └── v1.0.0/
│       ├── metadata.json          # ✅ 已更新，包含完整元数据
│       ├── model/
│       │   ├── model.json         # ✅ 模型文件
│       │   └── feature_names.json # ✅ 特征名称
│       ├── training/
│       │   └── metrics.json      # ✅ 训练指标
│       ├── evaluation/            # 评测目录（待使用）
│       └── experiments/           # 实验目录（待使用）
└── config.yaml                    # 模型配置文件（待创建）
```

---

## 🔄 版本迭代流程

### 创建新版本

```python
from src.models.lifecycle.trainer import ModelTrainer

trainer = ModelTrainer('breakout_launch_scorer')

# 自动创建新版本（基于最新版本递增）
model, metrics = trainer.train_version(neg_version='v2')

# 或指定版本号
model, metrics = trainer.train_version(version='v1.1.0', neg_version='v2')
```

### 查看版本信息

```python
from src.models.lifecycle.iterator import ModelIterator

iterator = ModelIterator('breakout_launch_scorer')

# 列出所有版本
versions = iterator.list_versions()
print(f"所有版本: {versions}")

# 获取最新版本
latest = iterator.get_latest_version()
print(f"最新版本: {latest}")

# 获取版本详细信息
info = iterator.get_version_info('v1.0.0')
print(f"版本信息: {info}")
```

### 使用检查脚本

```bash
python scripts/check_model_versions.py
```

---

## 📊 当前版本信息

### v1.0.0

- **状态**: development
- **创建时间**: 2025-12-28T22:13:47
- **训练样本**: 训练集 2442 条，测试集 613 条
- **训练日期范围**: 2002-02-07 至 2024-08-01
- **测试日期范围**: 2024-08-01 至 2025-12-04

#### 性能指标（测试集）

- **准确率**: 76.18%
- **精确率**: 73.71%
- **召回率**: 93.44%
- **F1分数**: 82.41%
- **AUC**: 0.7949

#### 模型参数

- **模型类型**: XGBoost
- **n_estimators**: 100
- **learning_rate**: 0.1
- **max_depth**: 5
- **subsample**: 0.8
- **colsample_bytree**: 0.8

---

## 🎯 后续版本迭代建议

### 版本命名规则

- **主版本号** (v2.0.0): 重大架构调整或不兼容变更
- **次版本号** (v1.1.0): 新功能添加（向后兼容）
- **补丁版本号** (v1.0.1): Bug修复或小优化

### 版本变更记录

每次创建新版本时，建议在 `changes` 字段中记录：

```python
changes = {
    'type': 'feature',  # feature, parameter, bugfix, performance
    'description': '新增OBV和KDJ指标',
    'impact': 'medium'  # low, medium, high
}
```

### 版本状态管理

- **development**: 开发中
- **testing**: 测试中
- **staging**: 预发布
- **production**: 生产环境
- **archived**: 已归档

---

## 🔍 验证检查清单

- [x] 现有版本文件完整
- [x] 元数据符合新框架要求
- [x] 训练器保存完整元数据
- [x] 训练脚本使用新框架
- [x] 版本管理功能正常
- [x] 创建验证工具脚本

---

## 📝 使用示例

### 训练新版本

```bash
# 自动创建新版本（v1.0.1）
python scripts/train_breakout_launch_scorer.py

# 或指定版本号
python -c "
from src.models.lifecycle.trainer import ModelTrainer
trainer = ModelTrainer('breakout_launch_scorer')
trainer.train_version(version='v1.1.0', neg_version='v2')
"
```

### 检查版本

```bash
python scripts/check_model_versions.py
```

### 使用模型进行预测

```bash
# 使用最新版本
python scripts/score_current_stocks.py

# 或指定版本（需要修改代码）
```

---

## 🎉 总结

✅ **模型版本已成功迁移到新框架**

- 现有版本 `v1.0.0` 的元数据已更新，符合新框架要求
- 训练器已更新，确保后续训练的新版本都会保存完整的元数据
- 版本管理功能完整，支持版本创建、查询、对比等操作
- 提供了验证工具，方便检查版本信息

现在可以正常进行版本迭代了！🚀

---

**文档版本**: v1.0
**创建日期**: 2025-12-28
**最后更新**: 2025-12-28
