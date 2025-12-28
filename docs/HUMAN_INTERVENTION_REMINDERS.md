# 人工介入提醒机制

## 📋 概述

项目已集成**人工介入提醒机制**，在需要人工决策的关键环节会自动提醒，确保不会遗漏重要的决策点。

---

## 🎯 提醒机制总览

| 环节 | 提醒位置 | 提醒方式 | 检查工具 |
|------|---------|---------|---------|
| **正样本选择** | `prepare_positive_samples.py` | 运行前检查配置 | ✅ |
| **数据标注** | `prepare_positive_samples.py` | 运行后检查质量 | ✅ |
| **特征选择** | `train_xgboost_timeseries.py` | 特征提取后提醒 | ✅ |
| **模型配置** | 训练前 | 检查配置文件 | ✅ |
| **训练结果** | `train_xgboost_timeseries.py` | 训练后检查指标 | ✅ |
| **版本对比** | 版本迭代时 | 对比结果提醒 | ✅ |

---

## 🚀 自动提醒机制

### 1. 正样本筛选条件检查

**触发时机**: 运行 `prepare_positive_samples.py` 时

**检查内容**:
- 是否使用默认阈值
- 日期范围是否合理
- 筛选条件是否需要调整

**提醒示例**:
```
================================================================================
👤 人工介入检查: 正样本筛选条件
================================================================================
⚠️  需要人工介入！

警告:
  ⚠️  使用默认值: consecutive_weeks = 3。请确认是否符合当前需求。
  ⚠️  使用默认值: total_return_threshold = 50。请确认是否符合当前需求。

建议:
  💡 数据起始日期为2000-01-01，请确认是否需要调整。
================================================================================
```

### 2. 正样本质量检查

**触发时机**: 正样本筛选完成后

**提醒内容**:
- 样本数量是否合理
- 平均涨幅是否符合预期
- 是否需要调整筛选条件

**提醒示例**:
```
================================================================================
👤 人工介入提醒：请检查正样本质量
================================================================================
请确认：
  1. 样本数量是否合理（建议：1000-5000个）
  2. 平均涨幅是否符合预期
  3. 样本分布是否合理
  4. 是否需要调整筛选条件
================================================================================
```

### 3. 特征选择提醒

**触发时机**: 特征提取完成后

**提醒内容**:
- 当前特征数量
- 是否需要添加更多特征
- 是否避免未来函数

**提醒示例**:
```
================================================================================
👤 人工介入提醒：特征提取完成
================================================================================
当前特征数量: 22 个（不含sample_id, label, t1_date）
请确认：
  1. 特征是否足够？是否需要添加基本面特征或其他技术指标？
  2. 特征是否避免了未来函数？
  3. 特征重要性将在训练后显示，请关注
================================================================================
```

### 4. 训练结果检查

**触发时机**: 模型训练完成后

**检查内容**:
- AUC是否达标（< 0.7 会警告）
- 准确率是否达标（< 75% 会警告）
- F1分数是否达标（< 70% 会警告）
- 是否存在过拟合

**提醒示例**:
```
================================================================================
👤 人工介入检查：训练结果
================================================================================
⚠️  AUC = 0.65 < 0.7，模型性能可能不佳
⚠️  准确率 = 72.00% < 75%，模型性能可能不佳

建议：
  - 检查特征选择，考虑添加更多有效特征
  - 调整超参数（n_estimators, max_depth, learning_rate等）
  - 检查数据质量，确保正负样本质量
  - 考虑尝试其他算法（LightGBM, CatBoost等）
================================================================================
```

---

## 🛠️ 手动检查工具

### 使用检查脚本

```bash
# 检查所有环节
python scripts/check_human_intervention.py --stage all

# 检查正样本筛选条件
python scripts/check_human_intervention.py --stage positive_samples

# 检查特征选择
python scripts/check_human_intervention.py --stage features

# 检查模型配置
python scripts/check_human_intervention.py --stage model_config --model breakout_launch_scorer

# 检查训练结果
python scripts/check_human_intervention.py --stage training_results --model breakout_launch_scorer --version v1.0.0

# 检查版本对比
python scripts/check_human_intervention.py --stage version_comparison \
  --model breakout_launch_scorer \
  --old-version v1.0.0 \
  --new-version v1.1.0
```

### 使用Python API

```python
from src.utils.human_intervention import HumanInterventionChecker, require_human_confirmation

checker = HumanInterventionChecker()

# 检查正样本筛选条件
result = checker.check_positive_sample_criteria()
checker.print_intervention_reminder("正样本筛选条件", result)

# 检查模型配置
result = checker.check_model_config('breakout_launch_scorer')
checker.print_intervention_reminder("模型配置", result)

# 检查训练结果
result = checker.check_training_results('breakout_launch_scorer', 'v1.0.0')
checker.print_intervention_reminder("训练结果", result)

# 要求人工确认
confirmed = require_human_confirmation(
    "⚠️  检测到配置可能需要调整。是否继续？",
    default=False
)
```

---

## 📝 提醒机制集成位置

### 1. `scripts/prepare_positive_samples.py`

**集成点**:
- ✅ 运行前：检查正样本筛选条件
- ✅ 运行后：提醒检查正样本质量

**代码位置**:
```python
# 运行前检查
checker = HumanInterventionChecker()
criteria_check = checker.check_positive_sample_criteria()
needs_intervention = checker.print_intervention_reminder("正样本筛选条件", criteria_check)

# 运行后提醒
log.warning("👤 人工介入提醒：请检查正样本质量")
```

### 2. `scripts/train_xgboost_timeseries.py`

**集成点**:
- ✅ 训练前：检查特征选择
- ✅ 特征提取后：提醒检查特征
- ✅ 训练后：检查训练结果

**代码位置**:
```python
# 训练前检查
checker = HumanInterventionChecker()
feature_check = checker.check_feature_selection()
checker.print_intervention_reminder("特征选择", feature_check)

# 训练后检查
warnings = []
if metrics['auc'] < 0.7:
    warnings.append(f"⚠️  AUC = {metrics['auc']:.3f} < 0.7")
# ... 其他检查
```

---

## ✅ 检查清单

### 运行正样本准备时

- [ ] 是否看到正样本筛选条件检查提醒？
- [ ] 是否确认了筛选条件符合需求？
- [ ] 是否检查了正样本质量提醒？
- [ ] 是否验证了样本数量和质量？

### 运行模型训练时

- [ ] 是否看到特征选择提醒？
- [ ] 是否确认了特征是否足够？
- [ ] 是否看到了训练结果检查？
- [ ] 如果指标不达标，是否采取了改进措施？

### 版本迭代时

- [ ] 是否运行了版本对比检查？
- [ ] 是否分析了新旧版本差异？
- [ ] 是否做出了升级/回滚决策？

---

## 🎯 最佳实践

1. **不要忽略提醒**
   - 所有提醒都有其目的
   - 即使看起来正常，也建议检查一下

2. **及时响应**
   - 看到警告时，及时检查并调整
   - 不要等到问题积累后再处理

3. **记录决策**
   - 如果决定忽略某个提醒，记录原因
   - 便于后续回顾和优化

4. **定期检查**
   - 使用 `check_human_intervention.py` 定期检查
   - 确保配置和结果符合预期

---

## 📚 相关文档

- [模型生命周期中的人工介入点](MODEL_LIFECYCLE_HUMAN_INTERVENTION.md)
- [模型生命周期标准化流程](MODEL_LIFECYCLE_STANDARD.md)
- [人工介入检查工具](../src/utils/human_intervention.py)

---

## 🔧 自定义提醒

如果需要添加新的提醒点，可以：

1. **在脚本中添加检查**:
```python
from src.utils.human_intervention import HumanInterventionChecker

checker = HumanInterventionChecker()
# 执行检查
result = checker.check_xxx()
checker.print_intervention_reminder("检查标题", result)
```

2. **使用确认机制**:
```python
from src.utils.human_intervention import require_human_confirmation

confirmed = require_human_confirmation(
    "⚠️  需要人工确认的消息",
    default=False
)
if not confirmed:
    # 处理取消情况
    return
```

---

## ❓ 常见问题

### Q1: 如何关闭提醒？

**A**: 不建议关闭提醒，但可以通过修改代码跳过检查。建议保留提醒机制。

### Q2: 提醒太多怎么办？

**A**: 如果提醒过多，说明确实需要人工介入。建议：
- 逐一处理每个提醒
- 优化配置，减少不必要的提醒
- 记录决策，避免重复提醒

### Q3: 如何添加新的提醒点？

**A**: 
1. 在 `HumanInterventionChecker` 中添加新的检查方法
2. 在相应的脚本中调用检查方法
3. 使用 `print_intervention_reminder()` 显示提醒

---

## 🎉 总结

人工介入提醒机制确保在关键决策点不会遗漏，帮助：
- ✅ 及时发现配置问题
- ✅ 确保数据质量
- ✅ 优化模型性能
- ✅ 做出正确决策

**记住**: 看到提醒时，请及时检查和处理！

