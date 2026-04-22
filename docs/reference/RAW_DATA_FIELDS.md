# 原始数据字段说明

## 📋 数据文件

- **正样本特征数据**: `data/training/features/feature_data_34d.csv`
- **负样本特征数据**: `data/training/features/negative_feature_data_v2_34d.csv`

## ✅ 原始数据字段确认

根据实际数据文件和代码，原始数据包含以下字段：

### 基础字段（必选）

| 字段名 | 类型 | 说明 | 示例值 | 来源 |
|--------|------|------|--------|------|
| `sample_id` | int | 样本ID | 0, 1, 2, ... | 生成 |
| `trade_date` | str | 交易日期 | '2015-10-19' | Tushare |
| `name` | str | 股票名称 | '万科A' | Tushare |
| `ts_code` | str | 股票代码 | '000002.SZ' | Tushare |
| `days_to_t1` | int | 距离T1的天数 | -34, -33, ..., -1 | 计算 |

### 价格和涨跌幅字段（必选）

| 字段名 | 类型 | 说明 | 示例值 | 来源 |
|--------|------|------|--------|------|
| `close` | float | 收盘价（元） | 10.6 | Tushare |
| `pct_chg` | float | 当日涨跌幅(%) | -0.84 | Tushare |

### 市值字段（必选）

| 字段名 | 类型 | 说明 | 示例值 | 来源 |
|--------|------|------|--------|------|
| `total_mv` | float | 总市值（万元） | 14950122.74 | Tushare |
| `circ_mv` | float | 流通市值（万元） | 13150169.38 | Tushare |

### 技术指标字段（必选）

| 字段名 | 类型 | 说明 | 示例值 | 来源 |
|--------|------|------|--------|------|
| `ma5` | float | 5日均线 | 10.57 | 计算/Tushare |
| `ma10` | float | 10日均线 | 10.38 | 计算/Tushare |
| `volume_ratio` | float | 量比 | 1.25 | 计算 |

### MACD字段（可选，如果Tushare技术因子可用）

| 字段名 | 类型 | 说明 | 示例值 | 来源 |
|--------|------|------|--------|------|
| `macd_dif` | float | MACD DIF线 | 0.12 | Tushare技术因子 |
| `macd_dea` | float | MACD DEA线 | 0.08 | Tushare技术因子 |
| `macd` | float | MACD柱状图 | 0.04 | Tushare技术因子 |

### RSI字段（可选，如果Tushare技术因子可用）

| 字段名 | 类型 | 说明 | 示例值 | 来源 |
|--------|------|------|--------|------|
| `rsi_6` | float | 6日RSI | 58.3 | Tushare技术因子 |
| `rsi_12` | float | 12日RSI | 56.2 | Tushare技术因子 |
| `rsi_24` | float | 24日RSI | 54.1 | Tushare技术因子 |

### 标签字段（仅负样本文件）

| 字段名 | 类型 | 说明 | 示例值 | 来源 |
|--------|------|------|--------|------|
| `label` | int | 标签（0=负样本，1=正样本） | 0 | 生成 |

---

## 📊 数据格式示例

### 完整示例（包含所有字段）

```python
{
    'sample_id': 0,
    'trade_date': '2015-10-19',      # 或 '20241001' 格式
    'name': '万科A',
    'ts_code': '000002.SZ',
    'close': 10.6,                    # 收盘价 10.6元
    'pct_chg': -0.84,                 # 当日下跌 0.84%
    'total_mv': 14950122.74,           # 总市值 1495亿元（万元单位）
    'circ_mv': 13150169.38,            # 流通市值 1315亿元（万元单位）
    'ma5': 10.57,                      # 5日均线 10.57元
    'ma10': 10.38,                     # 10日均线 10.38元
    'volume_ratio': 1.25,              # 量比 1.25（温和放量）
    'macd_dif': 0.12,                  # MACD DIF 为正（上涨动能）
    'macd_dea': 0.08,                  # MACD DEA
    'macd': 0.04,                      # MACD柱为正（动能增强）
    'rsi_6': 58.3,                     # 6日RSI 58.3（中性偏强）
    'rsi_12': 56.2,                    # 12日RSI 56.2
    'rsi_24': 54.1,                    # 24日RSI 54.1
    'days_to_t1': -34                  # 距离T1还有34天
}
```

### CSV格式

```csv
sample_id,trade_date,name,ts_code,close,pct_chg,total_mv,circ_mv,ma5,ma10,volume_ratio,macd_dif,macd_dea,macd,rsi_6,rsi_12,rsi_24,days_to_t1
0,2015-10-19,万科A,000002.SZ,10.6,-0.84,14950122.74,13150169.38,10.57,10.38,1.25,0.12,0.08,0.04,58.3,56.2,54.1,-34
```

---

## 🔍 字段说明

### 1. 基础字段

- **sample_id**: 每个样本的唯一ID，同一个样本的34天数据共享相同的sample_id
- **trade_date**: 交易日期，格式为 'YYYY-MM-DD' 或 'YYYYMMDD'
- **name**: 股票名称
- **ts_code**: 股票代码，格式为 '000002.SZ' 或 '600000.SH'
- **days_to_t1**: 距离T1日期的天数，范围从 -34 到 -1

### 2. 价格字段

- **close**: 收盘价（元），已复权
- **pct_chg**: 当日涨跌幅（%），正数表示上涨，负数表示下跌

### 3. 市值字段

- **total_mv**: 总市值（万元），例如 14950122.74 表示约 1495 亿元
- **circ_mv**: 流通市值（万元），例如 13150169.38 表示约 1315 亿元

### 4. 技术指标字段

- **ma5**: 5日移动平均线（元）
- **ma10**: 10日移动平均线（元）
- **volume_ratio**: 量比，当日成交量与过去N日平均成交量的比值

### 5. MACD字段（可选）

- **macd_dif**: MACD DIF线（快线）
- **macd_dea**: MACD DEA线（慢线）
- **macd**: MACD柱状图 = (DIF - DEA) × 2

**注意**: 这些字段可能不存在，取决于Tushare技术因子数据是否可用。

### 6. RSI字段（可选）

- **rsi_6**: 6日相对强弱指标
- **rsi_12**: 12日相对强弱指标
- **rsi_24**: 24日相对强弱指标

**注意**: 这些字段可能不存在，取决于Tushare技术因子数据是否可用。

---

## 📝 数据获取逻辑

在 `src/strategy/screening/positive_sample_screener.py` 的 `_extract_single_sample_features()` 方法中：

```python
# 1. 获取基础行情数据（包含收盘价、涨跌幅、市值等）
df = self.dm.get_complete_data(ts_code, start_date, end_date)

# 2. 获取Tushare技术因子（包含MACD、RSI等）
df_factor = self.dm.get_stk_factor(ts_code, start_date, end_date)
df = pd.merge(df, df_factor, on='trade_date', how='left')

# 3. 计算移动平均线和量比
if 'ma5' not in df.columns:
    df['ma5'] = df['close'].rolling(window=5).mean()
if 'ma10' not in df.columns:
    df['ma10'] = df['close'].rolling(window=10).mean()

# 4. 选择需要的字段
base_fields = [
    'trade_date', 'ts_code', 'close', 'pct_chg',
    'total_mv', 'circ_mv', 'ma5', 'ma10', 'volume_ratio'
]

# 5. 如果有技术因子，也包含进来
extra_fields = []
for field in ['macd_dif', 'macd_dea', 'macd', 'rsi_6', 'rsi_12', 'rsi_24']:
    if field in df.columns:
        extra_fields.append(field)
```

---

## ✅ 确认结果

**您提供的数据格式是正确的！** ✅

所有字段都存在于原始数据文件中：
- ✅ 基础字段：sample_id, trade_date, name, ts_code, days_to_t1
- ✅ 价格字段：close, pct_chg
- ✅ 市值字段：total_mv, circ_mv
- ✅ 技术指标：ma5, ma10, volume_ratio
- ✅ MACD字段：macd_dif, macd_dea, macd（如果可用）
- ✅ RSI字段：rsi_6, rsi_12, rsi_24（如果可用）

**注意**：
1. `trade_date` 格式可能是 'YYYY-MM-DD' 或 'YYYYMMDD'，取决于数据来源
2. MACD和RSI字段是**可选的**，如果Tushare技术因子数据不可用，这些字段可能不存在
3. 市值单位是**万元**，不是元

---

**文档版本**: v1.0
**创建日期**: 2025-12-28
**最后更新**: 2025-12-28
