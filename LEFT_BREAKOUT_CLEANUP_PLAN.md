# Left Breakout 文件清理计划

**日期**: 2025-12-28

## 📋 分析结果

### ✅ 核心三周连阳模型文件（必须保留）

这些是当前正在使用的核心模型文件：

1. **代码文件**
   - `src/strategy/screening/positive_sample_screener.py` - 核心正样本筛选器（实现三连阳逻辑）
   - `config/settings.yaml` - 配置文件（`consecutive_weeks: 3`）

2. **当前使用的数据文件**
   - `data/training/samples/positive_samples.csv` - 当前使用的正样本
   - `data/training/samples/negative_samples_v2.csv` - 当前使用的负样本
   - `data/training/features/feature_data_34d.csv` - 当前使用的特征数据

### ❌ Left Breakout 相关文件（可以删除）

这些是旧版本或测试版本的文件，代码中已不再引用：

1. **日志文件**（14个）
   - `logs/left_breakout_prepare_*.log` (9个)
   - `logs/left_breakout_training_*.log` (2个)
   - `logs/left_breakout_update_*.log` (2个)
   - `logs/left_breakout_samples_*.log` (1个)
   - `logs/train_left_breakout_*.log` (1个)

2. **数据文件**（4个）
   - `data/training/features/left_breakout_features.csv` - 旧版本特征数据（3802行）
   - `data/training/features/left_breakout_features_latest.csv` - 旧版本特征数据（3802行，可能是重复）
   - `data/training/samples/left_positive_samples.csv` - 旧版本正样本（2137行）
   - `data/training/samples/left_negative_samples.csv` - 旧版本负样本（2137行）

3. **验证结果**
   - ✅ 代码中无引用：`grep` 搜索 scripts 和 src 目录，未找到任何引用
   - ✅ 当前使用标准命名：`positive_samples.csv` 和 `negative_samples_v2.csv`

---

## 🎯 清理操作

### 步骤1：删除日志文件

```bash
# 删除所有 left_breakout 相关的日志文件
rm -f logs/left_breakout_*.log
rm -f logs/train_left_breakout_*.log
```

### 步骤2：删除旧数据文件

```bash
# 删除旧版本的特征数据
rm -f data/training/features/left_breakout_features.csv
rm -f data/training/features/left_breakout_features_latest.csv

# 删除旧版本的样本数据
rm -f data/training/samples/left_positive_samples.csv
rm -f data/training/samples/left_negative_samples.csv
```

---

## ⚠️ 注意事项

1. **核心模型文件不会受影响**
   - `src/strategy/screening/positive_sample_screener.py` - 保留
   - `config/settings.yaml` - 保留
   - `data/training/samples/positive_samples.csv` - 保留
   - `data/training/samples/negative_samples_v2.csv` - 保留

2. **文档中的引用**
   - 一些文档（如 `docs/OPTIMIZATION_COMPLETED.md`）中提到了 left_breakout
   - 这些是历史文档，不影响功能，可以保留作为历史记录

3. **如果误删了重要文件**
   - 可以从 Git 历史中恢复
   - 或者重新运行样本准备脚本生成

---

## 📊 清理前后对比

| 类型 | 清理前 | 清理后 | 说明 |
|------|--------|--------|------|
| 日志文件 | 14个 | 0个 | 历史运行日志 |
| 特征数据 | 2个 | 0个 | 旧版本数据 |
| 样本数据 | 2个 | 0个 | 旧版本数据 |
| **核心代码** | **1个** | **1个** | **保留** |
| **核心配置** | **1个** | **1个** | **保留** |
| **当前数据** | **3个** | **3个** | **保留** |

---

## ✅ 清理完成

**执行日期**: 2025-12-28

### 已删除的文件

1. **日志文件**（14个）
   - ✅ 所有 `logs/left_breakout_*.log` 文件
   - ✅ 所有 `logs/train_left_breakout_*.log` 文件

2. **数据文件**（4个）
   - ✅ `data/training/features/left_breakout_features.csv`
   - ✅ `data/training/features/left_breakout_features_latest.csv`
   - ✅ `data/training/samples/left_positive_samples.csv`
   - ✅ `data/training/samples/left_negative_samples.csv`

### 核心文件验证

✅ **所有核心文件完整保留**：
- `src/strategy/screening/positive_sample_screener.py` (15KB) - 核心筛选器
- `config/settings.yaml` (3.5KB) - 配置文件
- `data/training/samples/positive_samples.csv` (173KB) - 当前正样本
- `data/training/samples/negative_samples_v2.csv` (77KB) - 当前负样本

---

**状态**: ✅ 清理完成

