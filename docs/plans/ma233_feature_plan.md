# MA233因子实施计划

## 🎯 目标

通过添加5日均线与233日均线的关系特征，帮助模型在低波动市场（如2025年下半年）中更好地识别正样本，提升召回率。

### 背景问题

2025-05 ~ 2025-12 窗口召回率仅 69.57%，核心原因：
1. **市场波动率大幅下降**：窗口5波动率(0.69) 仅为训练集(1.37)的一半
2. **特征分布漂移**：正样本趋势斜率、超额收益等特征值远高于训练集
3. 模型在低波动环境下过于保守

### 解决思路

5日均线站上233日均线是经典的"牛熊分界"指标：
- 在低波动市场中，均线关系比动量指标更稳定
- 233日均线是机构常用的年线替代（约一年交易日）
- 可提供中长期趋势的判断依据

---

## 📊 新增特征列表

| 特征名 | 说明 | 类型 |
|--------|------|------|
| `ma_233d` | 233日均线值 | 连续 |
| `ma5_above_ma233` | 5日均线是否在233日均线之上 | 0/1 |
| `ma5_ma233_distance` | 5日均线与233日均线距离百分比 | 连续 |
| `price_vs_ma233` | 价格相对233日均线位置百分比 | 连续 |
| `ma5_golden_cross_233` | 5日均线刚金叉233日均线 | 0/1 |
| `ma5_death_cross_233` | 5日均线刚死叉233日均线 | 0/1 |
| `days_above_ma233` | 连续在233日均线之上的天数 | 整数 |
| `ma233_slope` | 233日均线5日斜率（趋势方向） | 连续 |
| `breakout_ma233` | 价格突破233日均线 | 0/1 |

---

## 📁 文件变更清单

### 1️⃣ 新建文件
- [ ] `scripts/add_ma233_factors.py` - MA233因子提取脚本

### 2️⃣ 修改文件
- [ ] `scripts/train_xgboost_timeseries.py`
  - 添加 `--use-ma233-factors` 命令行参数
  - 更新 `load_and_prepare_data()` 支持MA233特征文件
  - 更新 `extract_features_with_time()` 提取MA233统计特征

- [ ] `scripts/walk_forward_validation.py`
  - 添加 `--use-ma233-factors` 命令行参数
  - 更新 `load_and_prepare_data()` 支持MA233特征文件
  - 更新 `extract_features_with_time()` 提取MA233统计特征

### 3️⃣ 生成文件
- [ ] `data/training/processed/feature_data_34d_ma233.csv` - 正样本MA233特征
- [ ] `data/training/features/negative_feature_data_v2_34d_ma233.csv` - 负样本MA233特征

---

## ⏱️ 执行步骤

| 步骤 | 任务 | 预计时间 | 状态 |
|------|------|----------|------|
| 1 | 创建 `add_ma233_factors.py` 脚本 | 10分钟 | ⏳待完成 |
| 2 | 运行脚本提取正样本MA233特征 | ~30分钟 | ⏳待完成 |
| 3 | 运行脚本提取负样本MA233特征 | ~60分钟 | ⏳待完成 |
| 4 | 修改 `train_xgboost_timeseries.py` | 10分钟 | ⏳待完成 |
| 5 | 修改 `walk_forward_validation.py` | 10分钟 | ⏳待完成 |
| 6 | 运行模型训练 `--use-ma233-factors` | ~2分钟 | ⏳待完成 |
| 7 | 运行walk-forward验证 | ~1分钟 | ⏳待完成 |
| 8 | 对比分析结果 | 5分钟 | ⏳待完成 |

---

## 🔧 特征工程策略

在 `extract_features_with_time()` 中添加以下统计聚合：

```python
# MA233相关特征（如果存在）
if 'ma5_above_ma233' in sample_data.columns:
    data = sample_data['ma5_above_ma233'].dropna()
    if len(data) > 0:
        feature_dict['ma5_above_ma233_ratio'] = data.mean()  # 34天内在MA233上方的比例
        feature_dict['ma5_above_ma233_last'] = data.iloc[-1]  # 最后一天状态

if 'ma5_ma233_distance' in sample_data.columns:
    data = sample_data['ma5_ma233_distance'].dropna()
    if len(data) > 0:
        feature_dict['ma5_ma233_distance_last'] = data.iloc[-1]
        feature_dict['ma5_ma233_distance_mean'] = data.mean()

if 'days_above_ma233' in sample_data.columns:
    data = sample_data['days_above_ma233'].dropna()
    if len(data) > 0:
        feature_dict['days_above_ma233_last'] = data.iloc[-1]
        feature_dict['days_above_ma233_max'] = data.max()

if 'ma5_golden_cross_233' in sample_data.columns:
    data = sample_data['ma5_golden_cross_233'].dropna()
    if len(data) > 0:
        feature_dict['ma5_golden_cross_count'] = data.sum()  # 34天内金叉次数

if 'ma233_slope' in sample_data.columns:
    data = sample_data['ma233_slope'].dropna()
    if len(data) > 0:
        feature_dict['ma233_slope_last'] = data.iloc[-1]
        feature_dict['ma233_trend_up'] = 1 if data.iloc[-1] > 0 else 0
```

---

## 📈 预期效果

| 指标 | 当前（窗口5） | 预期改进 |
|------|--------------|----------|
| 召回率 | 69.57% | 75%+ |
| 精确率 | 86.15% | 保持 |
| F1 | 76.98% | 80%+ |

---

## ⚠️ 风险与注意事项

1. **数据要求**：计算233日均线需要至少233个交易日的历史数据，部分新股可能缺失
2. **时间成本**：每个样本需要获取约280天的日线数据，API调用较多
3. **缺失值处理**：上市不满233天的股票MA233将为NaN，需要合理填充

---

## 📝 变更记录

| 日期 | 变更内容 | 状态 |
|------|----------|------|
| 2026-01-01 | 创建计划文档 | ✅ |
| - | 待实施 | ⏳ |

---

*最后更新: 2026-01-01*

