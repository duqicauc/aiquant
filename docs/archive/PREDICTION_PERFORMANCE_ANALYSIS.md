# 预测脚本性能分析与优化

## ⏱️ 当前执行时间估算

### 执行流程

```
步骤1：批量获取daily_basic
  - 时间：1-5秒（一次API调用）
  
步骤2：批量获取日线数据（串行）
  - 如果数据在缓存：~0.03秒/股票 × 5000只 = ~150秒（2.5分钟）
  - 如果数据不在缓存：~3秒/股票 × 5000只 = ~15000秒（4小时+）⚠️
  
步骤3：计算特征并评分（串行）
  - 时间：~0.01秒/股票 × 5000只 = ~50秒
```

### 总时间估算

| 场景 | 步骤1 | 步骤2 | 步骤3 | 总计 |
|------|-------|-------|-------|------|
| **数据已缓存** | 5秒 | 150秒 | 50秒 | **~3.5分钟** ✅ |
| **数据未缓存** | 5秒 | 15000秒 | 50秒 | **~4.2小时** ⚠️ |

---

## 🔍 性能瓶颈分析

### 瓶颈1：批量获取日线数据（串行）

**代码位置**: `src/data/data_manager.py:269-309`

```python
def batch_get_daily_data(self, stock_codes, start_date, end_date):
    # 串行获取（因为SQLite不支持多线程）
    for code in stock_codes:
        df = self.get_daily_data(code, start_date, end_date)  # 串行
```

**问题**:
- 串行处理，即使有缓存也需要逐个检查
- 如果数据不在缓存，需要逐个从API获取，非常慢

**优化空间**: ⚠️ 有限（SQLite限制）

### 瓶颈2：特征计算（串行）

**代码位置**: `scripts/score_current_stocks.py:415-509`

```python
for i, (_, stock) in enumerate(valid_stocks.iterrows()):
    # 串行计算特征
    features = _calculate_features_from_df(df, ts_code, name)
    # 串行预测
    prob = model.predict(dmatrix)[0]
```

**问题**:
- 特征计算是CPU密集型任务，可以并行
- 模型预测很快，但串行处理限制了速度

**优化空间**: ✅ **可以大幅优化**（使用多进程）

---

## 🚀 优化方案

### 优化1：特征计算并行化（推荐）

**优化效果**: 速度提升 **4-8倍**（取决于CPU核心数）

**实现方式**: 使用多进程并行计算特征

```python
from multiprocessing import Pool
import numpy as np

def _calculate_features_parallel(args):
    """并行计算特征的辅助函数"""
    df, ts_code, name, daily_basic_dict, feature_cols = args
    # ... 特征计算逻辑 ...
    return result

# 在score_all_stocks中使用
with Pool(processes=8) as pool:  # 8个进程
    args_list = [(df, ts_code, name, daily_basic_dict, feature_cols) 
                 for ts_code, df in daily_data_dict.items()]
    results = pool.map(_calculate_features_parallel, args_list)
```

### 优化2：批量预测（推荐）

**优化效果**: 速度提升 **10-50倍**

**实现方式**: 批量构建DMatrix，一次性预测

```python
# 当前方式（慢）
for features in features_list:
    dmatrix = xgb.DMatrix([feature_values], feature_names=feature_cols)
    prob = model.predict(dmatrix)[0]  # 逐个预测

# 优化方式（快）
all_feature_values = [features_to_values(f) for f in features_list]
dmatrix = xgb.DMatrix(all_feature_values, feature_names=feature_cols)
all_probs = model.predict(dmatrix)  # 批量预测
```

### 优化3：减少日志输出频率

**优化效果**: 减少I/O开销，速度提升 **5-10%**

**当前**: 每50只股票输出一次日志
**优化**: 每500只股票输出一次日志

---

## 📊 优化后时间估算

### 优化后执行时间

| 场景 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| **数据已缓存** | 3.5分钟 | **30-60秒** | **3-7倍** ⚡ |
| **数据未缓存** | 4.2小时 | **4.2小时** | 相同（API限制） |

**注意**: 如果数据不在缓存，主要瓶颈是API调用速度，无法通过代码优化解决。

---

## 🛠️ 立即优化建议

### 方案1：快速优化（5分钟实现）

1. **批量预测**：将逐个预测改为批量预测
2. **减少日志频率**：从每50只改为每500只

**预期提升**: 10-20倍（特征计算部分）

### 方案2：深度优化（30分钟实现）

1. **特征计算并行化**：使用多进程
2. **批量预测**：批量构建DMatrix
3. **向量化计算**：使用NumPy向量化操作

**预期提升**: 20-50倍（特征计算部分）

---

## 💡 当前状态建议

### 如果数据已在缓存中

- **预计时间**: 3-5分钟
- **优化空间**: 可以优化到30-60秒

### 如果数据不在缓存中

- **预计时间**: 4-6小时（首次运行）
- **建议**: 
  1. 先运行小范围测试（`--max-stocks 100`）
  2. 让数据进入缓存
  3. 再运行完整预测

---

## 🔧 快速优化实现

我可以立即实现以下优化：

1. ✅ **批量预测**：将逐个预测改为批量预测（提升10-20倍）
2. ✅ **减少日志频率**：减少I/O开销（提升5-10%）
3. ⚠️ **特征计算并行化**：需要更多测试（提升4-8倍）

**总预期提升**: **15-30倍**（如果数据已在缓存中）

---

**文档版本**: v1.0  
**创建日期**: 2025-12-28

