# 模型特征对比说明

## 📊 特征对比

### 当前模型 v1.0.0 的特征（21个）

**已实现：**
- ✅ 价格特征（5个）：close_mean, close_std, close_max, close_min, close_trend
- ✅ 涨跌幅特征（7个）：pct_chg_mean, pct_chg_std, pct_chg_sum, positive_days, negative_days, max_gain, max_loss
- ✅ 量比特征（2个）：volume_ratio_mean, volume_ratio_max
- ✅ MACD特征（2个）：macd_mean, macd_positive_days
- ✅ MA特征（4个）：ma5_mean, price_above_ma5, ma10_mean, price_above_ma10

**缺失：**
- ❌ volume_ratio_gt_2 - 量比>2的天数
- ❌ volume_ratio_gt_4 - 量比>4的天数
- ❌ macd_max - 最大MACD值
- ❌ total_mv_mean - 平均总市值
- ❌ circ_mv_mean - 平均流通市值
- ❌ return_1w - 1周收益率
- ❌ return_2w - 2周收益率

### 目标特征（27个）

**完整特征列表：**

#### 1. 价格特征（5个）
1. close_mean - 平均收盘价
2. close_std - 收盘价标准差
3. close_max - 最高收盘价
4. close_min - 最低收盘价
5. close_trend - 价格趋势（34天累计涨跌幅）

#### 2. 涨跌幅特征（7个）
6. pct_chg_mean - 平均涨跌幅
7. pct_chg_std - 涨跌幅标准差
8. pct_chg_sum - 累计涨跌幅
9. positive_days - 上涨天数
10. negative_days - 下跌天数
11. max_gain - 最大单日涨幅
12. max_loss - 最大单日跌幅

#### 3. 量比特征（4个）
13. volume_ratio_mean - 平均量比
14. volume_ratio_max - 最大量比
15. **volume_ratio_gt_2** - 量比>2的天数 ⚠️ 缺失
16. **volume_ratio_gt_4** - 量比>4的天数 ⚠️ 缺失

#### 4. MACD特征（3个）
17. macd_mean - 平均MACD值
18. macd_positive_days - MACD>0的天数
19. **macd_max** - 最大MACD值 ⚠️ 缺失

#### 5. MA特征（4个）
20. ma5_mean - 平均MA5
21. price_above_ma5 - 价格>MA5的天数
22. ma10_mean - 平均MA10
23. price_above_ma10 - 价格>MA10的天数

#### 6. 市值特征（2个）
24. **total_mv_mean** - 平均总市值 ⚠️ 缺失
25. **circ_mv_mean** - 平均流通市值 ⚠️ 缺失

#### 7. 动量特征（2个）
26. **return_1w** - 1周收益率 ⚠️ 缺失
27. **return_2w** - 2周收益率 ⚠️ 缺失

---

## ✅ 已完成的修复

### 更新训练器特征提取逻辑

已更新 `src/models/lifecycle/trainer.py` 中的 `_extract_features` 方法，添加了缺失的6个特征：

1. ✅ `volume_ratio_gt_2` - 量比>2的天数
2. ✅ `volume_ratio_gt_4` - 量比>4的天数
3. ✅ `macd_max` - 最大MACD值
4. ✅ `total_mv_mean` - 平均总市值
5. ✅ `circ_mv_mean` - 平均流通市值
6. ✅ `return_1w` - 1周收益率
7. ✅ `return_2w` - 2周收益率

### 特征提取逻辑

```python
# 量比特征（新增2个）
feature_dict['volume_ratio_gt_2'] = (sample_data['volume_ratio'] > 2).sum()
feature_dict['volume_ratio_gt_4'] = (sample_data['volume_ratio'] > 4).sum()

# MACD特征（新增1个）
feature_dict['macd_max'] = macd_data.max()

# 市值特征（新增2个）
feature_dict['total_mv_mean'] = mv_data.mean()
feature_dict['circ_mv_mean'] = circ_mv_data.mean()

# 动量特征（新增2个）
feature_dict['return_1w'] = (最后一天 - 7天前) / 7天前 * 100
feature_dict['return_2w'] = (最后一天 - 14天前) / 14天前 * 100
```

---

## 🔄 后续操作

### 重新训练模型

由于当前 `v1.0.0` 版本使用的是21个特征，而预测脚本 `score_current_stocks.py` 使用的是27个特征，会导致特征不匹配的问题。

**建议操作：**

1. **重新训练模型**（推荐）
   ```bash
   python scripts/train_breakout_launch_scorer.py
   ```
   这将创建 `v1.0.1` 版本，使用完整的27个特征。

2. **验证特征匹配**
   训练完成后，检查新版本的特征列表：
   ```bash
   python scripts/check_model_versions.py
   ```

3. **更新预测脚本**
   确保 `score_current_stocks.py` 中的特征提取逻辑与训练器一致（已确认一致）。

---

## 📝 注意事项

1. **特征顺序**：确保训练和预测时特征顺序一致
2. **缺失值处理**：对于可能缺失的特征（如市值、MACD），已添加默认值处理
3. **数据要求**：确保训练数据包含所有必要的字段：
   - `volume_ratio` - 量比
   - `macd` - MACD值
   - `total_mv` - 总市值
   - `circ_mv` - 流通市值
   - `close` - 收盘价（用于计算动量特征）

---

## 🎯 总结

- ✅ 训练器已更新，支持27个特征
- ⚠️ 当前 `v1.0.0` 版本只有21个特征
- ✅ 下次训练将自动使用27个特征
- ✅ 预测脚本已支持27个特征

**建议：重新训练模型以使用完整的27个特征。**

---

**文档版本**: v1.0  
**创建日期**: 2025-12-28  
**最后更新**: 2025-12-28

