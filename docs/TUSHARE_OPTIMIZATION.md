# Tushare数据使用优化指南

## 📋 优化原则

**核心原则**：能用Tushare现成数据的，不要自己计算！

根据[Tushare官方文档](https://tushare.pro/document/2?doc_id=14)，Tushare提供了丰富的现成数据接口，我们应该充分利用。

---

## 🎯 已优化的部分

### 1. 技术指标 - 使用stk_factor接口 ✅

**Tushare提供**（stk_factor接口，需要5000积分）：
- **MACD**: `macd_dif`, `macd_dea`, `macd`
- **RSI**: `rsi_6`, `rsi_12`, `rsi_24`
- **KDJ**: `kdj_k`, `kdj_d`, `kdj_j`
- **BOLL**: `boll_upper`, `boll_mid`, `boll_lower`
- **MA**: `ma_5`, `ma_10`, `ma_20`, `ma_60`
- **其他**: `cci`, `adx`, `adxr`, `atr`等

**优化前**：
```python
# ❌ 自己计算MACD
df['macd_dif'] = calculate_macd_dif(df['close'])
df['macd_dea'] = calculate_macd_dea(df['close'])

# ❌ 自己计算RSI
df['rsi_6'] = calculate_rsi(df['close'], period=6)

# ❌ 自己计算MA20
df['ma20'] = df['close'].rolling(window=20).mean()

# ❌ 自己计算布林带
df['boll_lower'] = df['ma20'] - 2 * df['close'].rolling(window=20).std()
```

**优化后**：
```python
# ✅ 直接使用Tushare提供的现成数据
df_factor = dm.get_stk_factor(ts_code, start_date, end_date)
df = pd.merge(df, df_factor, on='trade_date', how='left')

# 直接使用：
# - df['macd_dif'], df['macd_dea']  # Tushare提供
# - df['rsi_6']  # Tushare提供
# - df['ma_20']  # Tushare提供（注意字段名是ma_20）
# - df['boll_lower']  # Tushare提供
```

---

### 2. 每日指标 - 使用daily_basic接口 ✅

**Tushare提供**（daily_basic接口，需要120积分）：
- `volume_ratio` - 量比
- `turnover_rate` - 换手率
- `total_mv` - 总市值
- `circ_mv` - 流通市值
- `pe`, `pe_ttm` - 市盈率
- `pb` - 市净率

**优化前**：
```python
# ❌ 自己计算量比
df['volume_ratio'] = df['vol'] / df['vol'].rolling(window=5).mean()
```

**优化后**：
```python
# ✅ 使用get_complete_data自动包含daily_basic数据
df = dm.get_complete_data(ts_code, start_date, end_date)
# df['volume_ratio'] 已经包含，来自daily_basic接口
```

---

### 3. 周线数据 - 使用weekly接口 ✅

**Tushare提供**（weekly接口，需要120积分）：
- 直接获取周线数据，无需本地转换

**优化前**：
```python
# ❌ 从日线数据转换周线
df_weekly = df_daily.resample('W').agg({...})
```

**优化后**：
```python
# ✅ 直接获取周线数据
df_weekly = dm.get_weekly_data(ts_code, start_date, end_date)
```

---

## 🔧 字段名映射

### Tushare字段名 vs 代码中使用的字段名

| Tushare字段 | 代码中使用 | 说明 |
|------------|----------|------|
| `ma_20` | `ma20` | 20日均线（注意下划线） |
| `boll_lower` | `boll_lower` | 布林带下轨（一致） |
| `boll_mid` | `boll_mid` | 布林带中轨（一致） |
| `boll_upper` | `boll_upper` | 布林带上轨（一致） |
| `rsi_6` | `rsi_6` | RSI(6)（一致） |
| `macd_dif` | `macd_dif` | MACD-DIF（一致） |
| `macd_dea` | `macd_dea` | MACD-DEA（一致） |

**注意**：代码中需要统一字段名时，可以这样处理：
```python
# 将Tushare的ma_20映射为ma20
if 'ma_20' in df.columns:
    df['ma20'] = df['ma_20']
```

---

## 📊 缓存优化

### 确保缓存所有Tushare字段

在`tushare_fetcher.py`中，确保缓存保存所有Tushare提供的字段：

```python
cols_to_save = [
    'ts_code', 'trade_date', 
    # MACD
    'macd_dif', 'macd_dea', 'macd',
    # RSI
    'rsi_6', 'rsi_12', 'rsi_24',
    # KDJ
    'kdj_k', 'kdj_d', 'kdj_j',
    # BOLL
    'boll_upper', 'boll_mid', 'boll_lower',
    # MA
    'ma_5', 'ma_10', 'ma_20', 'ma_60',
    # 其他
    'cci', 'adx', 'adxr', 'atr'
]
```

---

## ✅ 检查清单

在编写新代码时，检查以下事项：

- [ ] 是否使用了Tushare现成的技术指标（stk_factor）？
- [ ] 是否使用了Tushare现成的每日指标（daily_basic）？
- [ ] 是否使用了Tushare现成的周线数据（weekly）？
- [ ] 字段名是否正确（ma_20 vs ma20）？
- [ ] 缓存是否保存了所有需要的字段？
- [ ] 是否有兜底方案（Tushare数据缺失时）？

---

## 🎓 参考资源

- [Tushare官方文档](https://tushare.pro/document/2?doc_id=14)
- [Tushare Pro功能说明](../docs/TUSHARE_PRO_FEATURES.md)
- [API参考文档](../docs/API_REFERENCE.md)

---

## 💡 总结

**优化效果**：
- ✅ 减少代码量（无需自己实现技术指标计算）
- ✅ 提高数据质量（Tushare专业团队维护）
- ✅ 提升性能（避免重复计算）
- ✅ 降低错误率（使用经过验证的数据）

**关键点**：
1. 优先使用Tushare提供的现成数据
2. 注意字段名映射（ma_20 vs ma20）
3. 确保缓存所有需要的字段
4. 提供兜底方案（数据缺失时）

