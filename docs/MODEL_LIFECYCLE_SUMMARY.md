# 模型生命周期标准化流程 - 总结

## 📋 流程概述

我基于项目当前架构，设计了完整的模型迭代和新建标准化流程。以下是核心内容：

---

## 🎯 核心流程

### 1. 新建模型流程（7个阶段）

```
需求分析 → 模型设计 → 注册模型 → 准备数据 → 训练v1.0 → 评估 → 部署
```

**关键步骤**:
1. **需求分析**: 创建需求文档，明确目标和成功指标
2. **模型设计**: 创建设计文档，规划特征和算法
3. **注册模型**: 创建配置文件 + 代码注册
4. **准备数据**: 准备正负样本数据
5. **训练v1.0.0**: 使用ModelTrainer训练初始版本
6. **评估**: 生成评估报告，验证是否达到目标
7. **部署**: 标记为production状态

### 2. 迭代模型流程（6个阶段）

```
问题分析 → 版本规划 → 创建新版本 → 训练 → 对比评估 → 决策（升级/回滚）
```

**关键步骤**:
1. **问题分析**: 分析当前问题，确定改进方向
2. **版本规划**: 确定版本号（遵循语义化版本）
3. **创建新版本**: 使用ModelIterator创建新版本
4. **训练**: 训练新版本模型
5. **对比评估**: 对比新旧版本指标
6. **决策**: 升级（production）或回滚（deprecated）

---

## 📊 版本号规范

| 变更类型 | 版本号格式 | 示例 | 说明 |
|---------|-----------|------|------|
| 重大架构变更 | vX.0.0 | v2.0.0 | 算法更换、架构重构 |
| 新功能/重要特征 | v1.X.0 | v1.1.0 | 新增特征、重要改进 |
| 参数调优/bug修复 | v1.0.X | v1.0.1 | 超参数调整、bug修复 |

---

## 📁 目录结构

### 模型数据目录
```
data/models/{model_name}/
├── config.yaml                    # 模型基础配置
├── versions/                      # 版本目录
│   ├── v1.0.0/                   # 版本1.0.0
│   │   ├── metadata.json         # 版本元数据
│   │   ├── model/                # 模型文件
│   │   ├── training/             # 训练相关
│   │   └── evaluation/           # 评估相关
│   └── v1.1.0/                   # 版本1.1.0
└── ...
```

### 文档目录
```
docs/models/{model_name}/
├── requirements.md               # 需求文档（必须）
├── design.md                     # 设计文档（必须）
├── evaluation/                   # 评估报告（必须）
│   ├── v1.0.0.md
│   └── v1.1.0.md
└── changelog/                     # 变更日志（必须）
    ├── v1.1.0.md
    └── v2.0.0.md
```

---

## ✅ 检查点机制

### 新建模型检查点

每个阶段都有明确的检查清单和输出要求：

| 阶段 | 检查点 | 必须输出 |
|------|--------|---------|
| 需求分析 | 需求文档 | ✅ requirements.md |
| 模型设计 | 设计文档 | ✅ design.md |
| 模型注册 | 配置文件 | ✅ config/models/{name}.yaml |
| 数据准备 | 数据质量检查 | ✅ 数据文件 |
| 模型训练 | 训练完成 | ✅ 模型文件 + 指标 |
| 模型评估 | 评估报告 | ✅ evaluation/v1.0.0.md |
| 模型部署 | 生产标记 | ✅ status=production |

### 迭代模型检查点

| 阶段 | 检查点 | 必须输出 |
|------|--------|---------|
| 问题分析 | 变更日志 | ✅ changelog/{version}.md |
| 版本规划 | 版本号确定 | ✅ 版本元数据 |
| 创建版本 | 版本创建 | ✅ 版本目录 |
| 训练 | 训练完成 | ✅ 模型文件 + 指标 |
| 对比评估 | 对比报告 | ✅ 对比结果 |
| 决策 | 版本状态更新 | ✅ 状态更新 |

---

## 🛠️ 核心工具和API

### 1. ModelIterator - 版本管理

```python
from src.models.lifecycle.iterator import ModelIterator

iterator = ModelIterator('{model_name}')

# 创建新版本
iterator.create_version('v1.1.0', base_version='v1.0.0', changes={...})

# 获取版本信息
info = iterator.get_version_info('v1.0.0')

# 列出所有版本
versions = iterator.list_versions()

# 更新版本状态
iterator.update_version_metadata('v1.1.0', status='production')
```

### 2. ModelTrainer - 模型训练

```python
from src.models.lifecycle.trainer import ModelTrainer

trainer = ModelTrainer('{model_name}')
model, metrics = trainer.train_version(version='v1.1.0')
```

### 3. ModelRegistry - 模型注册

```python
from src.models.model_registry import ModelRegistry, ModelConfig

config = ModelConfig(...)
ModelRegistry.register(config)
```

---

## 📝 文档要求

### 必须文档

1. **需求文档** (`docs/models/{model_name}/requirements.md`)
   - 模型目标
   - 成功指标
   - 数据需求
   - 技术方案

2. **设计文档** (`docs/models/{model_name}/design.md`)
   - 特征工程方案
   - 算法选择
   - 训练策略

3. **评估报告** (`docs/models/{model_name}/evaluation/{version}.md`)
   - 训练指标
   - 测试集表现
   - 问题分析
   - 改进建议

4. **变更日志** (`docs/models/{model_name}/changelog/{version}.md`)
   - 变更类型
   - 变更描述
   - 影响评估

---

## 🎯 决策标准

### 版本升级决策

| 情况 | 决策 | 说明 |
|------|------|------|
| 新版本全面优于旧版本 | ✅ 升级 | 所有指标都有提升 |
| 新版本部分指标提升 | ⚠️ 评估后决定 | 权衡利弊 |
| 新版本不如旧版本 | ❌ 回滚 | 保持旧版本 |
| 新版本有严重问题 | ❌ 回滚 | 修复后重新训练 |

---

## 🚀 快速开始

### 新建模型

```bash
# 1. 创建需求文档
# 2. 创建设计文档
# 3. 创建配置文件 config/models/{name}.yaml
# 4. 注册模型（代码中）
# 5. 准备数据
python scripts/prepare_positive_samples.py
python scripts/prepare_negative_samples_v2.py
# 6. 训练v1.0.0
python scripts/train_xgboost_timeseries.py --model {name} --version v1.0.0
# 7. 评估
# 8. 标记为production
```

### 迭代模型

```bash
# 1. 分析问题，创建变更日志
# 2. 创建新版本（代码中）
# 3. 训练新版本
python scripts/train_xgboost_timeseries.py --model {name} --version v1.1.0
# 4. 对比评估
# 5. 决策（升级/回滚）
```

---

## 📚 相关文档

- **详细流程**: [模型生命周期标准化流程](MODEL_LIFECYCLE_STANDARD.md)
- **快速参考**: [模型生命周期快速参考](MODEL_LIFECYCLE_QUICK_REFERENCE.md)
- **版本管理**: [模型版本管理](MODEL_VERSION_MANAGEMENT.md)
- **训练指南**: [模型训练指南](MODEL_TRAINING_GUIDE.md)

---

## 👤 人工介入点

**关键人工决策环节**:

1. **正样本选择** ⭐⭐⭐⭐⭐
   - 筛选规则设定（连续周数、涨幅阈值等）
   - 配置位置: `config/settings.yaml`
   - 详见: [人工介入点文档](MODEL_LIFECYCLE_HUMAN_INTERVENTION.md#第一部分正样本选择)

2. **数据标注** ⭐⭐⭐
   - 标注规则验证和质量检查
   - 标注规则调整
   - 详见: [人工介入点文档](MODEL_LIFECYCLE_HUMAN_INTERVENTION.md#第二部分数据标注)

3. **因子/特征选择** ⭐⭐⭐⭐⭐
   - 决定使用哪些特征（价格、量比、MACD、MA、市值等）
   - 是否添加基本面特征
   - 代码位置: `scripts/train_xgboost_timeseries.py` - `extract_features_with_time()`
   - 详见: [人工介入点文档](MODEL_LIFECYCLE_HUMAN_INTERVENTION.md#第三部分因子特征选择)

4. **模型选择** ⭐⭐⭐⭐
   - 算法选择（XGBoost/LightGBM/CatBoost等）
   - 超参数调优
   - 训练策略设定
   - 配置位置: `config/models/{model_name}.yaml`
   - 详见: [人工介入点文档](MODEL_LIFECYCLE_HUMAN_INTERVENTION.md#第四部分模型选择)

5. **版本迭代决策** ⭐⭐⭐⭐⭐
   - 问题分析和改进方向
   - 版本规划和变更内容
   - 升级/回滚决策
   - 详见: [人工介入点文档](MODEL_LIFECYCLE_HUMAN_INTERVENTION.md#第五部分版本迭代决策)

**详细说明**: 请查看 [模型生命周期中的人工介入点](MODEL_LIFECYCLE_HUMAN_INTERVENTION.md)

---

## ❓ 待确认事项

请确认以下内容是否符合您的需求：

1. **流程阶段**: 新建模型7个阶段、迭代模型6个阶段是否合理？
2. **版本号规范**: 语义化版本（vX.Y.Z）是否符合项目需求？
3. **检查点机制**: 每个阶段的检查清单是否足够？
4. **文档要求**: 必须文档和可选文档的划分是否合适？
5. **决策标准**: 版本升级/回滚的决策标准是否清晰？
6. **人工介入点**: 正样本选择、因子选择、模型选择等人工决策环节是否明确？

如有需要调整的地方，请告知，我会相应修改流程文档。

---

## 🔄 后续计划

确认流程后，将：

1. **创建辅助脚本**:
   - `scripts/create_new_model.py` - 创建新模型模板
   - `scripts/compare_model_versions.py` - 版本对比工具
   - `scripts/update_model_status.py` - 状态管理工具

2. **完善文档模板**:
   - 需求文档模板
   - 设计文档模板
   - 评估报告模板
   - 变更日志模板

3. **集成到工作流**:
   - Git hooks检查
   - CI/CD集成
   - 自动化测试

