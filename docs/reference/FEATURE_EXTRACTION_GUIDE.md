# 特征提取逻辑说明

## 📋 概述

本文档详细说明模型训练过程中特征提取的逻辑，包括：
1. 原始数据字段
2. 特征提取方法
3. 最终模型使用的特征

---

## 🔍 第一步：原始数据获取

### 数据来源

从 `scripts/prepare_positive_samples.py` 和 `scripts/prepare_negative_samples_v2.py` 生成的特征数据文件：
- 正样本：`data/processed/feature_data_34d.csv`
- 负样本：`data/processed/negative_feature_data_v2_34d.csv`

### 原始数据字段（每行是一天的数据）

| 字段名 | 说明 | 来源 | 示例 |
|--------|------|------|------|
| `sample_id` | 样本ID | 生成 | 0, 1, 2, ... |
| `trade_date` | 交易日期 | Tushare | 2015-10-19 |
| `name` | 股票名称 | Tushare | 万科A |
| `ts_code` | 股票代码 | Tushare | 000002.SZ |
| `close` | 收盘价 | Tushare | 12.50 |
| `pct_chg` | 当日涨跌幅(%) | Tushare | 2.35 |
| `total_mv` | 总市值(万元) | Tushare | 1500000 |
| `circ_mv` | 流通市值(万元) | Tushare | 1200000 |
| `ma5` | 5日均线 | 计算/Tushare | 12.30 |
| `ma10` | 10日均线 | 计算/Tushare | 12.20 |
| `volume_ratio` | 量比 | 计算 | 1.5 |
| `macd_dif` | MACD DIF线 | Tushare技术因子 | 0.15 |
| `macd_dea` | MACD DEA线 | Tushare技术因子 | 0.12 |
| `macd` | MACD柱状图 | Tushare技术因子 | 0.06 |
| `rsi_6` | 6日RSI | Tushare技术因子 | 60.5 |
| `rsi_12` | 12日RSI | Tushare技术因子 | 55.3 |
| `rsi_24` | 24日RSI | Tushare技术因子 | 50.2 |
| `days_to_t1` | 距离T1的天数 | 计算 | -34, -33, ..., -1 |

### 数据获取逻辑

在 `src/strategy/screening/positive_sample_screener.py` 的 `_extract_single_sample_features()` 方法中：

```python
# 1. 获取基础行情数据（包含收盘价、涨跌幅、市值等）
df = self.dm.get_complete_data(ts_code, start_date, end_date)

# 2. 获取Tushare技术因子（包含MACD、RSI等）
df_factor = self.dm.get_stk_factor(ts_code, start_date, end_date)
df = pd.merge(df, df_factor, on='trade_date', how='left')

# 3. 计算MA5和MA10（如果Tushare没有提供）
if 'ma5' not in df.columns:
    df['ma5'] = df['close'].rolling(window=5).mean()
if 'ma10' not in df.columns:
    df['ma10'] = df['close'].rolling(window=10).mean()

# 4. 只取T1前的最后34天
df = df.tail(34)
```

**注意**：
- MACD使用的是标准参数（12/26/9），不是macd5或macd20
- 量比是计算得出的：`volume_ratio = 当日成交量 / 5日平均成交量`

---

## 🔧 第二步：特征工程（时序转统计）

### 提取位置

在 `scripts/train_xgboost_timeseries.py` 的 `extract_features_with_time()` 函数中。

### 提取逻辑

将34天的时序数据转换为统计特征（每行是一个样本）：

```python
for sample_id in sample_ids:
    sample_data = df[df['unique_sample_id'] == sample_id].sort_values('days_to_t1')

    # 对34天的数据进行统计特征提取
    feature_dict = {
        # 1. 价格特征（基于close）
        'close_mean': sample_data['close'].mean(),      # 平均收盘价
        'close_std': sample_data['close'].std(),        # 收盘价标准差
        'close_max': sample_data['close'].max(),        # 最高收盘价
        'close_min': sample_data['close'].min(),        # 最低收盘价
        'close_trend': (最后一天 - 第一天) / 第一天 * 100,  # 价格趋势

        # 2. 涨跌幅特征（基于pct_chg）
        'pct_chg_mean': sample_data['pct_chg'].mean(),  # 平均涨跌幅
        'pct_chg_std': sample_data['pct_chg'].std(),   # 涨跌幅波动
        'pct_chg_sum': sample_data['pct_chg'].sum(),   # 累计涨跌幅
        'positive_days': (pct_chg > 0).sum(),           # 上涨天数
        'negative_days': (pct_chg < 0).sum(),           # 下跌天数
        'max_gain': sample_data['pct_chg'].max(),       # 最大单日涨幅
        'max_loss': sample_data['pct_chg'].min(),       # 最大单日跌幅

        # 3. 量比特征（基于volume_ratio）
        'volume_ratio_mean': sample_data['volume_ratio'].mean(),  # 平均量比
        'volume_ratio_max': sample_data['volume_ratio'].max(),    # 最大量比
        'volume_ratio_gt_2': (volume_ratio > 2).sum(),            # 量比>2的天数
        'volume_ratio_gt_4': (volume_ratio > 4).sum(),            # 量比>4的天数

        # 4. MACD特征（基于macd）
        'macd_mean': sample_data['macd'].mean(),                    # 平均MACD值
        'macd_positive_days': (macd > 0).sum(),                     # MACD>0的天数
        'macd_max': sample_data['macd'].max(),                      # 最大MACD值

        # 5. MA特征（基于ma5, ma10）
        'ma5_mean': sample_data['ma5'].mean(),                      # 平均MA5
        'price_above_ma5': (close > ma5).sum(),                     # 价格>MA5的天数
        'ma10_mean': sample_data['ma10'].mean(),                    # 平均MA10
        'price_above_ma10': (close > ma10).sum(),                   # 价格>MA10的天数

        # 6. 市值特征（基于total_mv, circ_mv）
        'total_mv_mean': sample_data['total_mv'].mean(),           # 平均总市值
        'circ_mv_mean': sample_data['circ_mv'].mean(),             # 平均流通市值

        # 7. 动量特征（分段收益率）
        'return_1w': (最后一天 - 7天前) / 7天前 * 100,              # 1周收益率
        'return_2w': (最后一天 - 14天前) / 14天前 * 100,             # 2周收益率
    }
```

---

## 📊 最终模型使用的特征列表

### 特征分类

#### 1. 价格特征（5个）
- `close_mean` - 平均收盘价
- `close_std` - 收盘价标准差
- `close_max` - 最高收盘价
- `close_min` - 最低收盘价
- `close_trend` - 价格趋势（34天累计涨跌幅）

#### 2. 涨跌幅特征（7个）
- `pct_chg_mean` - 平均涨跌幅
- `pct_chg_std` - 涨跌幅波动
- `pct_chg_sum` - 累计涨跌幅
- `positive_days` - 上涨天数
- `negative_days` - 下跌天数
- `max_gain` - 最大单日涨幅
- `max_loss` - 最大单日跌幅

#### 3. 量比特征（4个）
- `volume_ratio_mean` - 平均量比
- `volume_ratio_max` - 最大量比
- `volume_ratio_gt_2` - 量比>2的天数
- `volume_ratio_gt_4` - 量比>4的天数

#### 4. MACD特征（3个）
- `macd_mean` - 平均MACD值
- `macd_positive_days` - MACD>0的天数
- `macd_max` - 最大MACD值

**注意**：使用的是标准MACD（12/26/9），不是macd5或macd20。

#### 5. MA特征（4个）
- `ma5_mean` - 平均MA5
- `price_above_ma5` - 价格>MA5的天数
- `ma10_mean` - 平均MA10
- `price_above_ma10` - 价格>MA10的天数

#### 6. 市值特征（2个）
- `total_mv_mean` - 平均总市值
- `circ_mv_mean` - 平均流通市值

#### 7. 动量特征（2个）
- `return_1w` - 1周收益率
- `return_2w` - 2周收益率

### 总计

**约27个特征**（不含sample_id, label, t1_date等元数据字段）

---

## ⚠️ 用户要求 vs 实际实现

### 用户要求
- macd5、macd20
- 量比
- 收盘价
- 当日涨跌幅
- 总市值
- 流通市值

### 实际实现

| 用户要求 | 实际实现 | 说明 |
|---------|---------|------|
| macd5 | ❌ 无 | 使用标准MACD（12/26/9），不是macd5 |
| macd20 | ❌ 无 | 使用标准MACD（12/26/9），不是macd20 |
| 量比 | ✅ 有 | `volume_ratio`，并提取了统计特征（均值、最大值、阈值计数） |
| 收盘价 | ✅ 有 | `close`，并提取了统计特征（均值、标准差、最大最小值、趋势） |
| 当日涨跌幅 | ✅ 有 | `pct_chg`，并提取了统计特征（均值、标准差、累计、正负天数、最大最小） |
| 总市值 | ✅ 有 | `total_mv`，提取了平均值 |
| 流通市值 | ✅ 有 | `circ_mv`，提取了平均值 |

### 额外特征

除了用户要求的基础字段，还额外提取了：
- MA5和MA10（移动平均线）
- RSI（相对强弱指标，虽然原始数据有，但模型特征中未使用）
- 动量特征（1周、2周收益率）
- 各种统计特征（均值、标准差、最大值、最小值、阈值计数等）

---

## 🔄 特征提取流程总结

```
原始数据（34天时序数据）
    ↓
每个样本包含34行数据，每行有18个字段
    ↓
特征工程（extract_features_with_time）
    ↓
统计特征提取（均值、标准差、最大值、最小值、计数等）
    ↓
最终特征（27个特征，每行是一个样本）
    ↓
模型训练（XGBoost）
```

---

## 💡 如果需要添加macd5和macd20

如果确实需要macd5和macd20，需要修改以下位置：

### 1. 数据获取阶段

在 `src/strategy/screening/positive_sample_screener.py` 中添加：

```python
# 计算MACD5（5日EMA - 10日EMA）
ema5 = df['close'].ewm(span=5).mean()
ema10 = df['close'].ewm(span=10).mean()
df['macd5_dif'] = ema5 - ema10
df['macd5'] = df['macd5_dif'].ewm(span=3).mean() * 2

# 计算MACD20（20日EMA - 40日EMA）
ema20 = df['close'].ewm(span=20).mean()
ema40 = df['close'].ewm(span=40).mean()
df['macd20_dif'] = ema20 - ema40
df['macd20'] = df['macd20_dif'].ewm(span=9).mean() * 2
```

### 2. 特征提取阶段

在 `scripts/train_xgboost_timeseries.py` 的 `extract_features_with_time()` 中添加：

```python
# MACD5特征
if 'macd5' in sample_data.columns:
    macd5_data = sample_data['macd5'].dropna()
    if len(macd5_data) > 0:
        feature_dict['macd5_mean'] = macd5_data.mean()
        feature_dict['macd5_positive_days'] = (macd5_data > 0).sum()
        feature_dict['macd5_max'] = macd5_data.max()

# MACD20特征
if 'macd20' in sample_data.columns:
    macd20_data = sample_data['macd20'].dropna()
    if len(macd20_data) > 0:
        feature_dict['macd20_mean'] = macd20_data.mean()
        feature_dict['macd20_positive_days'] = (macd20_data > 0).sum()
        feature_dict['macd20_max'] = macd20_data.max()
```

---

## 📝 相关文件

- 数据准备：`src/strategy/screening/positive_sample_screener.py`
- 特征提取：`scripts/train_xgboost_timeseries.py` (extract_features_with_time函数)
- 数据文件：`data/processed/feature_data_34d.csv`

---

**最后更新**：2025-12-25
