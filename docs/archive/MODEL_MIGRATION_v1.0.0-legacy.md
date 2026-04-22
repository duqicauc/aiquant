# 模型迁移文档：xgboost_timeseries_v2_20251225_205905.json → v1.0.0-legacy

**迁移日期**: 2025-12-29
**迁移脚本**: `scripts/migrate_xgboost_timeseries_to_new_framework.py`

---

## 📋 迁移概述

将旧的 `xgboost_timeseries_v2_20251225_205905.json` 模型迁移到新框架，版本号为 `v1.0.0-legacy`。

---

## 📁 迁移内容

### 1. 模型文件
- **源文件**: `data/training/models/xgboost_timeseries_v2_20251225_205905.json`
- **目标文件**: `data/models/breakout_launch_scorer/versions/v1.0.0-legacy/model/model.json`
- **状态**: ✅ 已迁移

### 2. 指标文件
- **源文件**: `data/training/metrics/xgboost_timeseries_v2_metrics.json`
- **目标文件**: `data/models/breakout_launch_scorer/versions/v1.0.0-legacy/metadata.json`
- **状态**: ✅ 已转换为metadata格式

### 3. 可视化图表
- **源目录**: `data/training/charts/`
- **目标目录**: `data/models/breakout_launch_scorer/versions/v1.0.0-legacy/charts/`
- **状态**: ✅ 已迁移

### 4. 特征名称
- **目标文件**: `data/models/breakout_launch_scorer/versions/v1.0.0-legacy/model/feature_names.json`
- **特征数量**: 27个
- **状态**: ✅ 已提取并保存

---

## 📊 模型信息

### 性能指标
- **准确率**: 77.16%
- **精确率**: 75.80%
- **召回率**: 90.71%
- **F1分数**: 82.59%
- **AUC**: 0.8393

### 训练数据范围
- **训练集**: 2002-02-07 至 2024-08-01
- **测试集**: 2024-08-01 至 2025-12-04

### 特征列表（27个）
1. close_mean, close_std, close_max, close_min, close_trend
2. pct_chg_mean, pct_chg_std, pct_chg_sum
3. positive_days, negative_days, max_gain, max_loss
4. volume_ratio_mean, volume_ratio_max, volume_ratio_gt_2, volume_ratio_gt_4
5. macd_mean, macd_positive_days, macd_max
6. ma5_mean, price_above_ma5, ma10_mean, price_above_ma10
7. total_mv_mean, circ_mv_mean
8. return_1w, return_2w

---

## 🔧 使用方法

### 1. 使用迁移后的模型进行预测

```bash
# 使用指定版本
python scripts/score_current_stocks.py --date 20251225 --version v1.0.0-legacy
```

### 2. 在代码中使用

```python
from scripts.score_current_stocks import load_model

# 加载迁移后的模型
model = load_model(version='v1.0.0-legacy')

# 使用模型进行预测
# ... 预测代码 ...
```

### 3. 查看模型信息

```python
from src.models.lifecycle.iterator import ModelIterator

iterator = ModelIterator('breakout_launch_scorer')
info = iterator.get_version_info('v1.0.0-legacy')
print(info)
```

---

## ✅ 验证结果

### 1. 模型加载测试
- ✅ 模型文件可以正常加载
- ✅ 特征名称正确提取（27个特征）
- ✅ 模型可以正常进行预测

### 2. 功能测试
- ✅ 预测功能正常
- ✅ 特征计算兼容
- ✅ 结果输出正常

---

## 📝 迁移后的目录结构

```
data/models/breakout_launch_scorer/versions/v1.0.0-legacy/
├── model/
│   ├── model.json              # 模型文件
│   └── feature_names.json      # 特征名称
├── charts/                     # 可视化图表
│   ├── feature_distribution_comparison.png
│   ├── return_distribution.png
│   ├── sample_count_comparison.png
│   ├── sample_quality_comparison.html
│   └── time_distribution_comparison.png
├── metadata.json               # 版本元数据
├── evaluation/                 # 评估结果（空）
├── experiments/                # 实验记录（空）
└── training/                   # 训练记录（空）
```

---

## 🔄 兼容性说明

### 向后兼容
- ✅ 旧代码可以继续使用旧路径加载模型
- ✅ `load_model()` 函数自动兼容新旧框架
- ✅ 预测脚本无需修改即可使用

### 新框架优势
- ✅ 统一的版本管理
- ✅ 完整的元数据记录
- ✅ 便于后续迭代和对比

---

## 📌 注意事项

1. **版本标识**: 使用 `v1.0.0-legacy` 标识这是从旧框架迁移的模型
2. **状态标记**: 标记为 `production` 状态，表示这是生产可用版本
3. **原始文件**: 原始文件保留在 `data/training/` 目录下，未删除
4. **特征兼容**: 特征列表与旧模型完全一致，确保预测结果一致

---

## 🎯 后续建议

1. **测试验证**: 使用迁移后的模型进行实际预测，验证结果与旧模型一致
2. **文档更新**: 更新相关文档，说明新框架的使用方法
3. **逐步迁移**: 如果还有其他旧模型，可以按照相同方式迁移

---

**迁移完成时间**: 2025-12-29 23:53:10
**迁移脚本**: `scripts/migrate_xgboost_timeseries_to_new_framework.py`
**验证状态**: ✅ 通过
