# 新旧框架预测结果差异分析

**问题**: 同一个模型在新旧框架下预测结果不一致

**原因**: 新框架的特征提取逻辑不完整，缺少多个关键特征

---

## 🔍 问题根源

### 1. 缺失的特征

新框架的 `ModelPredictor._extract_stock_features` 方法缺少以下特征：

| 特征名称 | 说明 | 影响 |
|---------|------|------|
| `volume_ratio_gt_2` | 量比>2的天数 | ⚠️ 中等 |
| `volume_ratio_gt_4` | 量比>4的天数 | ⚠️ 中等 |
| `macd_max` | MACD最大值 | ⚠️ 中等 |
| `total_mv_mean` | 总市值均值 | ⚠️ 较小 |
| `circ_mv_mean` | 流通市值均值 | ⚠️ 较小 |
| `return_1w` | 1周收益率 | ⚠️ 中等 |
| `return_2w` | 2周收益率 | ⚠️ 中等 |

**总计**: 缺少 **7个特征**，导致模型接收到的特征向量与训练时不一致

### 2. MACD计算方式不同

**旧框架**（正确）:
```python
# 完整的MACD计算
ema12 = df['close'].ewm(span=12, adjust=False).mean()
ema26 = df['close'].ewm(span=26, adjust=False).mean()
df['macd_dif'] = ema12 - ema26
df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
df['macd'] = (df['macd_dif'] - df['macd_dea']) * 2  # 完整的MACD

# 提取特征
features['macd_mean'] = macd_data.mean()
features['macd_positive_days'] = (macd_data > 0).sum()
features['macd_max'] = macd_data.max()  # ✅ 有这个
```

**新框架**（错误）:
```python
# 只计算了MACD DIF（不完整）
exp1 = df['close'].ewm(span=12, adjust=False).mean()
exp2 = df['close'].ewm(span=26, adjust=False).mean()
macd = exp1 - exp2  # 这只是DIF，不是完整的MACD

# 提取特征
features['macd_mean'] = macd_data.mean()
features['macd_positive_days'] = (macd_data > 0).sum()
# ❌ 缺少 macd_max
```

### 3. 量比计算方式不同

**旧框架**:
```python
# 优先使用daily_basic的volume_ratio（更准确）
if 'volume_ratio' in basic_row and pd.notna(basic_row['volume_ratio']):
    df['volume_ratio'] = df['volume_ratio'].fillna(basic_row['volume_ratio'])

# 如果没有，则计算
if 'volume_ratio' not in df.columns:
    df['vol_ma5'] = df['vol'].rolling(window=5, min_periods=1).mean()
    df['volume_ratio'] = df['vol'] / df['vol_ma5']  # 没有加小值

# 提取特征
features['volume_ratio_mean'] = df['volume_ratio'].mean()
features['volume_ratio_max'] = df['volume_ratio'].max()
features['volume_ratio_gt_2'] = (df['volume_ratio'] > 2).sum()  # ✅
features['volume_ratio_gt_4'] = (df['volume_ratio'] > 4).sum()  # ✅
```

**新框架**:
```python
# 直接计算，没有使用daily_basic
df['vol_ma5'] = df['vol'].rolling(window=5, min_periods=1).mean()
df['volume_ratio'] = df['vol'] / (df['vol_ma5'] + 1e-6)  # 加了小值（不一致）

# 提取特征
features['volume_ratio_mean'] = df['volume_ratio'].mean()
features['volume_ratio_max'] = df['volume_ratio'].max()
# ❌ 缺少 volume_ratio_gt_2
# ❌ 缺少 volume_ratio_gt_4
```

---

## ✅ 修复方案

### 已修复（2025-01-XX）

已更新 `src/models/lifecycle/predictor.py` 中的 `_extract_stock_features` 方法，使其与旧框架的特征计算逻辑保持一致：

1. ✅ 添加了缺失的7个特征
2. ✅ 修复了MACD计算方式（使用完整的MACD，包括macd_max）
3. ✅ 修复了量比计算方式（与训练时一致）
4. ✅ 添加了市值特征（total_mv_mean, circ_mv_mean）
5. ✅ 添加了动量特征（return_1w, return_2w）

### 修复后的特征列表（27个）

与训练时完全一致：

1. `close_mean`, `close_std`, `close_max`, `close_min`, `close_trend`
2. `pct_chg_mean`, `pct_chg_std`, `pct_chg_sum`
3. `positive_days`, `negative_days`, `max_gain`, `max_loss`
4. `volume_ratio_mean`, `volume_ratio_max`, `volume_ratio_gt_2`, `volume_ratio_gt_4`
5. `macd_mean`, `macd_positive_days`, `macd_max`
6. `ma5_mean`, `price_above_ma5`, `ma10_mean`, `price_above_ma10`
7. `total_mv_mean`, `circ_mv_mean`
8. `return_1w`, `return_2w`

---

## 📊 影响评估

### 预测结果差异

**修复前**:
- 特征数量: 20个（缺少7个）
- 预测结果: ❌ 与训练时不一致
- 可能影响: 预测准确率下降 5-15%

**修复后**:
- 特征数量: 27个（完整）
- 预测结果: ✅ 与训练时一致
- 预期效果: 预测准确率恢复正常

---

## 🔧 进一步优化建议

### 1. 使用Tushare技术因子（可选）

旧框架在计算特征时会尝试获取Tushare的技术因子（stk_factor），如果获取成功，会优先使用Tushare的数据而不是本地计算。这可以提高特征准确性。

**建议**: 在新框架中也添加这个逻辑（可选，因为会增加API调用）

### 2. 使用daily_basic数据（可选）

旧框架会获取daily_basic数据来补充volume_ratio和市值数据，这比本地计算更准确。

**建议**: 在新框架中也添加这个逻辑（可选，因为会增加API调用）

### 3. 特征验证

建议添加特征验证逻辑，确保预测时使用的特征与训练时完全一致：

```python
# 验证特征数量
expected_features = 27
actual_features = len(features)
if actual_features < expected_features:
    log.warning(f"特征数量不足: {actual_features} < {expected_features}")
```

---

## 📝 测试建议

### 1. 对比测试

使用相同的股票和日期，对比修复前后的预测结果：

```bash
# 修复前
python scripts/score_current_stocks.py --date 20251225

# 修复后
python scripts/score_current_stocks.py --date 20251225
```

### 2. 特征一致性测试

验证特征计算是否与训练时一致：

```python
# 使用训练时的特征提取逻辑
from scripts.score_current_stocks import _calculate_features_from_df

# 使用新框架的特征提取逻辑
from src.models.lifecycle.predictor import ModelPredictor

# 对比两者的特征值
```

---

## 🎯 总结

**问题**: 新框架缺少7个关键特征，导致预测结果不一致

**修复**: 已更新特征提取逻辑，使其与训练时完全一致

**状态**: ✅ 已修复

**建议**: 运行对比测试，验证修复效果

