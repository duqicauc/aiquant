# Tushare Pro 高级功能使用说明 💎

本文档说明AIQuant项目如何充分利用Tushare Pro的高级数据和API。

参考文档：[Tushare Pro 官方文档](https://tushare.pro/document/2?doc_id=14)

---

## 🎯 已集成的Tushare Pro功能

### 1. **周线数据API** ✅

**接口**: `weekly` / `pro_bar(freq='W')`
**积分要求**: 120积分
**优势**:
- 直接获取周线数据，无需本地转换
- 支持前复权、后复权
- 数据质量更高，计算更准确

**使用示例**:
```python
from src.data.data_manager import DataManager

dm = DataManager(source='tushare')

# 获取周线数据
df_weekly = dm.get_weekly_data(
    stock_code='600519.SH',
    start_date='20200101',
    end_date='20241231',
    adjust='qfq'  # 前复权
)
```

**应用场景**:
- 正样本筛选器的三连阳判断
- 周线级别的技术分析
- 中长期趋势判断

---

### 2. **每日指标API** ✅

**接口**: `daily_basic`
**积分要求**: 120积分
**包含指标**:
- 换手率 (turnover_rate)
- 量比 (volume_ratio)
- 市盈率 (pe, pe_ttm)
- 市净率 (pb)
- 总市值 (total_mv)
- 流通市值 (circ_mv)
- 总股本 (total_share)
- 流通股本 (float_share)
- 自由流通股本 (free_share)

**使用示例**:
```python
# 获取每日指标
df_basic = dm.get_daily_basic(
    stock_code='600519.SH',
    start_date='20200101',
    end_date='20241231'
)

# 自动合并到行情数据
df_complete = dm.get_complete_data(
    stock_code='600519.SH',
    start_date='20200101'
)
```

**应用场景**:
- 特征工程（市值、估值指标）
- 量比分析
- 基本面筛选

---

### 3. **技术因子API** ✅ 🔥

**接口**: `stk_factor`
**积分要求**: 5000积分（付费获取）
**包含指标**:
- **MACD**: macd_dif, macd_dea, macd
- **KDJ**: kdj_k, kdj_d, kdj_j
- **RSI**: rsi_6, rsi_12, rsi_24
- **BOLL**: boll_upper, boll_mid, boll_lower
- **MA**: ma_5, ma_10, ma_20, ma_60
- **其他**: cci, adx, adxr, atr等

**使用示例**:
```python
# 获取技术因子
df_factor = dm.get_stk_factor(
    stock_code='600519.SH',
    start_date='20200101',
    end_date='20241231'
)

# 包含的指标
print(df_factor.columns)
# ['trade_date', 'macd_dif', 'macd_dea', 'macd',
#  'kdj_k', 'kdj_d', 'kdj_j', 'rsi_6', 'rsi_12', 'rsi_24', ...]
```

**优势**:
- 🎯 **无需本地计算**，直接获取专业计算的技术指标
- 📊 **数据质量高**，经过Tushare专业团队验证
- ⚡ **性能更好**，避免重复计算
- 🔧 **指标全面**，涵盖主流技术指标

**应用场景**:
- 正样本特征提取
- 技术面分析
- 量化策略开发
- 机器学习特征

**⚠️ 积分说明**:
- 技术因子需要5000积分
- 可通过捐赠快速获得：[社区捐助](https://tushare.pro/community)
- **强烈推荐**：技术因子大幅提升开发效率

---

### 4. **筹码分布API** ✅ 💎

**接口**: `cyq_perf`
**积分要求**: 5000积分
**包含数据**:
- 筹码集中度
- 获利比例
- 平均成本
- 胜率数据

**使用示例**:
```python
# 获取筹码数据
df_cyq = dm.get_cyq_perf(
    stock_code='600519.SH',
    start_date='20200101',
    end_date='20241231'
)
```

**应用场景**:
- 主力资金分析
- 筹码分布研究
- 高级量化策略

---

### 5. **交易日历API** ✅

**接口**: `trade_cal`
**积分要求**: 2000积分
**功能**:
- 获取交易所交易日历
- 判断是否交易日
- 准确计算交易日天数

**使用示例**:
```python
# 获取交易日历
df_cal = dm.get_trade_calendar(
    start_date='20200101',
    end_date='20241231',
    exchange='SSE'  # 上交所
)

# 筛选交易日
trading_days = df_cal[df_cal['is_open'] == 1]
```

**应用场景**:
- 回看N个交易日
- 计算持仓天数
- 回测系统

---

### 6. **复权因子API** ✅

**接口**: `adj_factor`
**积分要求**: 120积分
**功能**:
- 获取复权因子
- 自定义复权计算

**使用示例**:
```python
# 获取复权因子
df_adj = dm.get_adj_factor(
    stock_code='600519.SH',
    start_date='20200101',
    end_date='20241231'
)
```

---

### 7. **ST股票列表** ✅

**接口**: `namechange`
**积分要求**: 120积分
**功能**:
- 获取ST股票列表
- 股票名称变更历史

**使用示例**:
```python
# 获取ST股票列表
df_st = dm.fetcher.get_st_list()
```

**应用场景**:
- 过滤ST股票
- 风险控制

---

## 📊 在正样本模型中的应用

### 优化前 vs 优化后

#### **优化前**（仅使用基础API）
```python
# ❌ 本地转换日线到周线
df_daily = dm.get_daily_data(stock_code, start_date, end_date)
df_weekly = df_daily.resample('W').agg(...)  # 本地计算

# ❌ 本地计算技术指标
df['ma5'] = df['close'].rolling(5).mean()
df['ma10'] = df['close'].rolling(10).mean()
df['macd'] = calculate_macd(df)  # 自己实现

# 问题：
# - 计算复杂
# - 容易出错
# - 性能较差
# - 需要自己实现所有指标
```

#### **优化后**（充分使用Tushare Pro）
```python
# ✅ 直接获取周线数据
df_weekly = dm.get_weekly_data(stock_code, start_date, end_date, adjust='qfq')

# ✅ 直接获取技术因子
df_factor = dm.get_stk_factor(stock_code, start_date, end_date)
# 包含: MA5, MA10, MACD, KDJ, RSI, BOLL等所有指标

# ✅ 直接获取市值和量比
df_basic = dm.get_daily_basic(stock_code, start_date, end_date)
# 包含: 市值、量比、换手率、PE、PB等

# 优势：
# ✅ 数据质量更高
# ✅ 无需本地计算
# ✅ 性能更好
# ✅ 代码更简洁
```

---

## 💰 积分要求总览

| 功能 | 接口 | 积分要求 | 推荐等级 |
|-----|------|---------|---------|
| 股票列表 | stock_basic | 0 | ⭐⭐⭐⭐⭐ |
| 日线行情 | daily | 0 | ⭐⭐⭐⭐⭐ |
| 周线行情 | weekly | 120 | ⭐⭐⭐⭐⭐ |
| 每日指标 | daily_basic | 120 | ⭐⭐⭐⭐⭐ |
| 复权因子 | adj_factor | 120 | ⭐⭐⭐⭐ |
| 交易日历 | trade_cal | 2000 | ⭐⭐⭐⭐⭐ |
| **技术因子** | **stk_factor** | **5000** | **⭐⭐⭐⭐⭐** 🔥 |
| 筹码分布 | cyq_perf | 5000 | ⭐⭐⭐⭐ |
| ST股票 | namechange | 120 | ⭐⭐⭐⭐ |

### 推荐配置

#### **基础版**（免费）
- 可完成基本量化分析
- 需要自己计算技术指标
- 积分：0

#### **进阶版**（120积分）
- 获取周线、每日指标
- 数据更丰富
- 积分：120（注册即可）

#### **专业版**（2000积分）⭐
- 获取交易日历
- 支持完整的量化系统
- 积分：2000（完善资料+签到可达）

#### **旗舰版**（5000积分）🔥
- **强烈推荐！**
- 直接获取所有技术指标
- 筹码分布数据
- 节省大量开发时间
- 积分：5000（建议直接捐赠获得）

**💡 获取积分方式**:
1. 注册：120积分
2. 完善资料：300积分
3. 每日签到：1积分/天
4. **捐赠**：推荐方式，快速获得5000+积分
   - 访问：https://tushare.pro/community

---

## 🚀 使用建议

### 1. **优先使用Tushare Pro的现成数据**
```python
# ✅ 好的做法
df_weekly = dm.get_weekly_data(...)  # 使用Tushare周线API
df_factor = dm.get_stk_factor(...)   # 使用技术因子API

# ❌ 不推荐
df_weekly = convert_to_weekly(df_daily)  # 本地转换
ma5 = df['close'].rolling(5).mean()      # 本地计算
```

### 2. **技术因子强烈推荐**
如果你的积分达到5000，**强烈建议**使用`stk_factor`接口：
- 节省大量计算时间
- 避免指标计算错误
- 数据质量更高
- 专业团队维护

### 3. **合理规划API调用**
```python
# ✅ 一次获取完整数据
df = dm.get_complete_data(stock_code, start_date, end_date)
df_factor = dm.get_stk_factor(stock_code, start_date, end_date)

# ❌ 避免多次重复调用
for date in dates:
    df = dm.get_daily_data(stock_code, date, date)  # 每次都调用
```

### 4. **使用交易日历**
```python
# ✅ 使用交易日历准确计算
df_cal = dm.get_trade_calendar('20200101', '20241231')
trading_days = df_cal[df_cal['is_open'] == 1]['cal_date'].tolist()
t1_minus_34 = trading_days[trading_days.index(t1) - 34]

# ❌ 简单减去自然日
t1_minus_34 = t1 - timedelta(days=34)  # 不准确
```

---

## 📝 API完整列表

AIQuant已集成以下Tushare Pro接口：

### DataManager方法对应

| 方法 | Tushare接口 | 说明 |
|-----|------------|------|
| `get_stock_list()` | stock_basic | 股票列表 |
| `get_daily_data()` | pro_bar | 日线行情 |
| `get_weekly_data()` | weekly/pro_bar | 周线行情 ✨ |
| `get_minute_data()` | pro_bar | 分钟行情 |
| `get_daily_basic()` | daily_basic | 每日指标 ✨ |
| `get_fundamental_data()` | fina_indicator | 财务指标 |
| `get_complete_data()` | 综合接口 | 行情+指标 ✨ |
| `get_stk_factor()` | stk_factor | 技术因子 🔥 |
| `get_cyq_perf()` | cyq_perf | 筹码数据 💎 |
| `get_trade_calendar()` | trade_cal | 交易日历 ✨ |

---

## 🎓 学习资源

- [Tushare Pro官方文档](https://tushare.pro/document/2?doc_id=14)
- [Tushare积分体系](https://tushare.pro/document/1?doc_id=13)
- [社区捐助](https://tushare.pro/community)
- [数据字典](https://tushare.pro/document/2)

---

## ❓ 常见问题

### Q1: 技术因子值得付费吗？
**A**: **非常值得！**
- 节省几周的开发时间
- 避免技术指标计算错误
- 专业团队维护，数据质量高
- 包含100+技术指标

### Q2: 如何快速获得5000积分？
**A**:
- 推荐方式：捐赠获得（https://tushare.pro/community）
- 慢速方式：完善资料+每日签到（需要约1年）

### Q3: 免费版本够用吗？
**A**:
- 学习阶段：够用
- 生产环境：建议至少2000积分（交易日历）
- 专业量化：建议5000积分（技术因子）

### Q4: 技术因子包含哪些指标？
**A**: 包含但不限于：
- 趋势类：MA、EMA、MACD、DMI、ADX
- 震荡类：RSI、KDJ、CCI、WR
- 波动类：BOLL、ATR
- 成交量：OBV、VOL

---

**总结**: AIQuant项目已经充分集成了Tushare Pro的高级功能。如果你的积分足够，可以获得更好的数据质量和开发体验！💎

**建议投资**: 如果认真做量化交易，5000积分的技术因子API非常值得！🚀
