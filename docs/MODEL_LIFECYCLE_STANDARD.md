# 模型生命周期标准化流程

## 📋 概述

本文档定义了项目后续迭代模型、新建模型的标准化流程，确保模型开发、训练、评估、部署的规范化和可追溯性。

---

## 🎯 流程总览

```
新建模型流程：
需求分析 → 模型设计 → 注册模型 → 准备数据 → 训练v1.0 → 评估 → 部署

迭代模型流程：
问题分析 → 版本规划 → 创建新版本 → 训练 → 对比评估 → 决策（升级/回滚）
```

---

## 📝 第一部分：新建模型流程

### 阶段1: 需求分析与设计

#### 1.1 需求分析

**🤖 自动化程度**: 人工决策

**检查清单**:
- [ ] 明确模型目标（预测什么？解决什么问题？）
- [ ] 定义成功指标（准确率、召回率、AUC等）
- [ ] 确定数据需求（需要哪些数据？数据量要求？）
- [ ] 评估技术可行性

**输出文档**:
- `docs/models/{model_name}/requirements.md` - 需求文档

**模板**:
```markdown
# {模型名称} 需求文档

## 1. 模型目标
- 预测目标：...
- 业务场景：...
- 预期效果：...

## 2. 成功指标
- 准确率 >= X%
- 召回率 >= Y%
- AUC >= Z%

## 3. 数据需求
- 正样本：...
- 负样本：...
- 特征数据：...

## 4. 技术方案
- 算法选择：...
- 特征工程：...
- 训练策略：...
```

#### 1.2 模型设计

**检查清单**:
- [ ] 设计特征工程方案
- [ ] 选择算法和超参数范围
- [ ] 设计训练/验证/测试集划分策略
- [ ] 设计评估指标和验证方法

**输出文档**:
- `docs/models/{model_name}/design.md` - 设计文档

### 阶段2: 模型注册

#### 2.1 创建模型配置

**步骤**:

1. **创建模型配置文件** (`config/models/{model_name}.yaml`):

```yaml
model:
  name: {model_name}
  display_name: {显示名称}
  description: {模型描述}
  
data:
  positive_samples: data/training/samples/positive_samples.csv
  negative_samples: data/training/samples/negative_samples_v2.csv
  feature_data: data/training/features/feature_data_34d.csv
  
model_params:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  # ... 其他参数

training:
  train_test_split: 0.8
  validation_split: 0.2
  time_series_split: true
```

2. **注册模型** (在代码中注册):

```python
from src.models.model_registry import ModelRegistry, ModelConfig

config = ModelConfig(
    name='{model_name}',
    display_name='{显示名称}',
    description='{模型描述}',
    data_dir='{model_name}',
    model_dir='models',
    sample_dir='samples',
    metrics_dir='metrics',
    prediction_dir='predictions'
)

ModelRegistry.register(config)
```

**检查清单**:
- [ ] 配置文件创建完成
- [ ] 模型已注册到ModelRegistry
- [ ] 目录结构已自动创建

### 阶段3: 数据准备

#### 3.1 准备训练数据

**步骤**:

1. **准备正样本**:
```bash
python scripts/prepare_positive_samples.py
```

2. **准备负样本**:
```bash
python scripts/prepare_negative_samples_v2.py
```

3. **提取特征**:
```bash
# 特征提取通常在训练脚本中自动完成
```

**检查清单**:
- [ ] 正样本数据已准备（`data/training/samples/positive_samples.csv`）
- [ ] 负样本数据已准备（`data/training/samples/negative_samples_v2.csv`）
- [ ] 特征数据已准备（`data/training/features/feature_data_34d.csv`）
- [ ] 数据质量检查通过（运行 `python scripts/check_sample_quality.py`）

### 阶段4: 模型训练

#### 4.1 创建初始版本

**步骤**:

1. **使用ModelTrainer训练**:

```python
from src.models.lifecycle.trainer import ModelTrainer

trainer = ModelTrainer('{model_name}')
model, metrics = trainer.train_version(version='v1.0.0')
```

或使用训练脚本:

```bash
python scripts/train_xgboost_timeseries.py --model {model_name} --version v1.0.0
```

**输出**:
- 模型文件: `data/models/{model_name}/versions/v1.0.0/model/model.json`
- 特征名称: `data/models/{model_name}/versions/v1.0.0/model/feature_names.json`
- 训练指标: `data/models/{model_name}/versions/v1.0.0/training/metrics.json`
- 版本元数据: `data/models/{model_name}/versions/v1.0.0/metadata.json`

**检查清单**:
- [ ] 模型训练完成
- [ ] 训练指标达到预期
- [ ] 模型文件已保存
- [ ] 版本元数据已更新

### 阶段5: 模型评估

#### 5.1 性能评估

**步骤**:

1. **查看训练指标**:
```python
from src.models.lifecycle.iterator import ModelIterator

iterator = ModelIterator('{model_name}')
info = iterator.get_version_info('v1.0.0')
print(info['metrics'])
```

2. **运行回测** (如适用):
```bash
python scripts/backtest_example.py --model {model_name} --version v1.0.0
```

3. **分析预测准确率**:
```bash
python scripts/analyze_prediction_accuracy.py --model {model_name} --version v1.0.0
```

**检查清单**:
- [ ] 训练指标达到需求文档中的成功指标
- [ ] 回测结果符合预期（如适用）
- [ ] 预测准确率分析完成
- [ ] 评估报告已生成

#### 5.2 评估报告

**输出文档**:
- `docs/models/{model_name}/evaluation/v1.0.0.md` - 评估报告

**模板**:
```markdown
# {模型名称} v1.0.0 评估报告

## 1. 训练指标
- 准确率: X%
- 精确率: Y%
- 召回率: Z%
- F1分数: W%
- AUC: V%

## 2. 测试集表现
- ...

## 3. 回测结果（如适用）
- ...

## 4. 问题分析
- ...

## 5. 改进建议
- ...
```

### 阶段6: 模型部署

#### 6.1 标记为生产版本

**步骤**:

```python
from src.models.lifecycle.iterator import ModelIterator

iterator = ModelIterator('{model_name}')
iterator.update_version_metadata('v1.0.0', status='production')
```

**检查清单**:
- [ ] 版本状态已更新为 `production`
- [ ] 模型可用于预测（`scripts/score_current_stocks.py` 会自动使用最新生产版本）
- [ ] 部署文档已更新

---

## 🔄 第二部分：迭代模型流程

### 阶段1: 问题分析与版本规划

#### 1.1 问题分析

**触发条件**:
- 模型性能不达标
- 发现新的改进点
- 业务需求变化
- 数据质量提升

**分析步骤**:

1. **收集问题**:
   - 查看预测准确率分析报告
   - 分析错误案例
   - 收集用户反馈

2. **确定改进方向**:
   - 特征工程改进
   - 超参数调优
   - 算法改进
   - 数据质量提升

**输出文档**:
- `docs/models/{model_name}/changelog/{version}.md` - 变更日志

#### 1.2 版本规划

**版本号规范**:
- **主版本号 (vX.0.0)**: 重大架构变更、算法更换
- **次版本号 (v1.X.0)**: 新功能、重要特征添加
- **补丁版本 (v1.0.X)**: 参数调优、bug修复

**规划步骤**:

1. **确定版本号**:
```python
from src.models.lifecycle.iterator import ModelIterator

iterator = ModelIterator('{model_name}')
latest_version = iterator.get_latest_version()
# 根据变更类型确定新版本号
```

2. **记录变更内容**:
```python
changes = {
    'type': 'feature',  # feature, parameter, bugfix, performance
    'description': '新增OBV和KDJ指标',
    'impact': 'medium'  # low, medium, high
}
```

**检查清单**:
- [ ] 版本号已确定
- [ ] 变更内容已记录
- [ ] 变更日志已创建

### 阶段2: 创建新版本

#### 2.1 创建版本

**步骤**:

```python
from src.models.lifecycle.iterator import ModelIterator

iterator = ModelIterator('{model_name}')

# 创建新版本
new_version = iterator.create_version(
    version='v1.1.0',
    base_version='v1.0.0',  # 基于哪个版本
    changes={
        'features': ['added_obv', 'added_kdj'],
        'parameters': {'n_estimators': 150},
        'description': '新增OBV和KDJ指标，优化模型参数'
    },
    created_by='your_name'
)
```

**检查清单**:
- [ ] 新版本目录已创建
- [ ] 版本元数据已初始化
- [ ] 变更记录已保存

### 阶段3: 训练新版本

#### 3.1 训练

**步骤**:

```python
from src.models.lifecycle.trainer import ModelTrainer

trainer = ModelTrainer('{model_name}')
model, metrics = trainer.train_version(version='v1.1.0')
```

**检查清单**:
- [ ] 训练完成
- [ ] 训练指标已记录
- [ ] 模型文件已保存

### 阶段4: 对比评估

#### 4.1 版本对比

**步骤**:

```python
from src.models.lifecycle.iterator import ModelIterator

iterator = ModelIterator('{model_name}')

# 获取两个版本的指标
v1_info = iterator.get_version_info('v1.0.0')
v2_info = iterator.get_version_info('v1.1.0')

# 对比指标
print(f"v1.0.0 AUC: {v1_info['metrics']['test']['auc']}")
print(f"v1.1.0 AUC: {v2_info['metrics']['test']['auc']}")
```

**对比维度**:
- 训练指标（准确率、AUC、F1等）
- 测试集表现
- 回测结果（如适用）
- 预测准确率（实际应用表现）

**检查清单**:
- [ ] 版本对比完成
- [ ] 对比报告已生成
- [ ] 改进效果已量化

#### 4.2 决策

**决策标准**:

| 情况 | 决策 | 说明 |
|------|------|------|
| 新版本全面优于旧版本 | 升级 | 所有指标都有提升 |
| 新版本部分指标提升 | 评估后决定 | 权衡利弊 |
| 新版本不如旧版本 | 回滚 | 保持旧版本 |
| 新版本有严重问题 | 回滚 | 修复后重新训练 |

**决策步骤**:

1. **如果决定升级**:
```python
iterator.update_version_metadata('v1.1.0', status='production')
iterator.update_version_metadata('v1.0.0', status='deprecated')
```

2. **如果决定回滚**:
```python
# 保持旧版本为production
# 新版本标记为deprecated或删除
iterator.update_version_metadata('v1.1.0', status='deprecated')
```

**检查清单**:
- [ ] 决策已做出
- [ ] 版本状态已更新
- [ ] 决策记录已保存

---

## 📊 第三部分：流程检查点

### 新建模型检查点

| 阶段 | 检查点 | 必须项 | 输出 |
|------|--------|--------|------|
| 需求分析 | 需求文档 | ✅ | requirements.md |
| 模型设计 | 设计文档 | ✅ | design.md |
| 模型注册 | 配置文件 | ✅ | config/models/{name}.yaml |
| 数据准备 | 数据质量检查 | ✅ | 数据文件 |
| 模型训练 | 训练完成 | ✅ | 模型文件 + 指标 |
| 模型评估 | 评估报告 | ✅ | evaluation/v1.0.0.md |
| 模型部署 | 生产标记 | ✅ | status=production |

### 迭代模型检查点

| 阶段 | 检查点 | 必须项 | 输出 |
|------|--------|--------|------|
| 问题分析 | 变更日志 | ✅ | changelog/{version}.md |
| 版本规划 | 版本号确定 | ✅ | 版本元数据 |
| 创建版本 | 版本创建 | ✅ | 版本目录 |
| 训练 | 训练完成 | ✅ | 模型文件 + 指标 |
| 对比评估 | 对比报告 | ✅ | 对比结果 |
| 决策 | 版本状态更新 | ✅ | 状态更新 |

---

## 🛠️ 第四部分：工具和脚本

### 辅助脚本

#### 1. 创建新模型模板

```bash
# 创建新模型（待实现）
python scripts/create_new_model.py --name {model_name} --display-name "{显示名称}"
```

#### 2. 版本对比工具

```bash
# 对比两个版本（待实现）
python scripts/compare_model_versions.py --model {model_name} --v1 v1.0.0 --v2 v1.1.0
```

#### 3. 模型状态管理

```bash
# 列出所有版本
python scripts/list_model_versions.py --model {model_name}

# 查看版本详情
python scripts/show_model_version.py --model {model_name} --version v1.0.0

# 标记版本状态
python scripts/update_model_status.py --model {model_name} --version v1.1.0 --status production
```

---

## 📝 第五部分：文档规范

### 必须文档

1. **需求文档**: `docs/models/{model_name}/requirements.md`
2. **设计文档**: `docs/models/{model_name}/design.md`
3. **评估报告**: `docs/models/{model_name}/evaluation/{version}.md`
4. **变更日志**: `docs/models/{model_name}/changelog/{version}.md`

### 可选文档

1. **实验记录**: `docs/models/{model_name}/experiments/{experiment_id}.md`
2. **问题分析**: `docs/models/{model_name}/issues/{issue_id}.md`

---

## ✅ 第六部分：质量保证

### 测试要求

- [ ] 模型训练脚本有单元测试
- [ ] 特征工程有测试覆盖
- [ ] 模型预测有集成测试

### 代码审查

- [ ] 配置文件审查
- [ ] 训练脚本审查
- [ ] 评估结果审查

### 性能要求

- [ ] 训练指标达到需求文档要求
- [ ] 预测速度满足业务需求
- [ ] 模型文件大小合理

---

## 🎯 快速参考

### 新建模型快速流程

```bash
# 1. 创建需求文档
# 2. 创建设计文档
# 3. 创建配置文件
# 4. 注册模型
# 5. 准备数据
# 6. 训练v1.0.0
python scripts/train_xgboost_timeseries.py --model {model_name} --version v1.0.0
# 7. 评估
# 8. 标记为production
```

### 迭代模型快速流程

```bash
# 1. 分析问题，确定改进方向
# 2. 创建新版本
# 3. 训练新版本
python scripts/train_xgboost_timeseries.py --model {model_name} --version v1.1.0
# 4. 对比评估
# 5. 决策（升级/回滚）
```

---

## 📚 相关文档

- [模型版本管理](MODEL_VERSION_MANAGEMENT.md)
- [模型训练指南](MODEL_TRAINING_GUIDE.md)
- [模型注册表说明](../src/models/model_registry.py)

---

## ❓ 常见问题

### Q1: 如何确定版本号？

**A**: 
- 主版本号：重大架构变更
- 次版本号：新功能、重要特征
- 补丁版本：参数调优、bug修复

### Q2: 什么时候应该创建新模型而不是迭代？

**A**: 
- 预测目标完全不同
- 使用完全不同的算法
- 数据来源完全不同

### Q3: 如何回滚到旧版本？

**A**: 
```python
iterator.update_version_metadata('v1.0.0', status='production')
iterator.update_version_metadata('v1.1.0', status='deprecated')
```

---

## 🔄 流程改进

本流程会根据实际使用情况持续改进，如有建议请提交issue或PR。

