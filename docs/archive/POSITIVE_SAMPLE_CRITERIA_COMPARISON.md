# 正负样本筛选逻辑说明

## 📋 概述

本文档详细说明正样本和负样本的筛选逻辑，两者使用相同的过滤规则以保证数据一致性。

---

## 🔧 统一过滤规则

正样本和负样本均使用以下过滤规则剔除不符合条件的股票：

| 过滤类型 | 说明 | 实现方式 |
|---------|------|---------|
| **ST** | 剔除ST股票 | 名称包含 `ST`（不区分大小写，含ST、*ST、S*ST、SST等） |
| **HALT** | 剔除停牌股票 | 使用 `suspend_d` 接口查询T1日期停牌的股票 |
| **DELISTING** | 剔除退市股票 | 使用 `list_status='L'` 只获取上市状态的股票 |
| **DELISTING_SORTING** | 剔除退市整理期股票 | 名称包含 `退` 字 |
| **北交所** | 剔除北交所股票 | 股票代码以 `.BJ` 结尾 |
| **上市时间** | 上市满180天 | T1日期与上市日期间隔 ≥ 180天 |

---

## ✅ 正样本筛选条件

### 一、筛选符合条件的股票和T1

筛选历史上满足以下**所有条件**的股票：

| 条件 | 说明 | 实现 |
|------|------|------|
| 1. 周K三连阳 | 连续3周收盘价 > 开盘价 | `week['close'] > week['open']` |
| 2. 总涨幅 > 50% | (第3周收盘价 - 第1周开盘价) / 第1周开盘价 > 50% | `total_return > 50` |
| 3. 最高涨幅 > 70% | (3周内最高价 - 第1周开盘价) / 第1周开盘价 > 70% | `max_return > 70` |
| 4. 过滤规则 | 剔除ST、停牌、退市、退市整理期、北交所 | 见上方统一规则 |
| 5. 上市时间 ≥ 180天 | 上市日期距离T1至少180天 | `days_since_list >= 180` |
| 6. T1定义 | T1 = 第一周的第一个交易日 | `t1_date = week1['trade_date']` |
| 7. 去重规则 | 重叠时间段合并（选最早T1），不重叠保留多个 | `_merge_overlapping_samples()` |

### 二、特征数据

提取**T1前34天**的交易数据，包含以下字段：

| 字段 | 说明 | 来源 |
|------|------|------|
| sample_id | 样本ID | 自动生成 |
| ts_code | 股票代码 | Tushare |
| name | 股票名称 | Tushare |
| trade_date | 交易日期 | Tushare |
| close | 收盘价 | Tushare |
| pct_chg | 当日涨跌幅(%) | Tushare |
| total_mv | 总市值(万元) | Tushare |
| circ_mv | 流通市值(万元) | Tushare |
| ma5 | 5日移动平均线 | 计算 |
| ma10 | 10日移动平均线 | 计算 |
| volume_ratio | 量比 | Tushare |
| macd_dif | MACD DIF | Tushare Pro |
| macd_dea | MACD DEA | Tushare Pro |
| macd | MACD柱 | Tushare Pro |
| rsi_6 | 6日RSI | Tushare Pro |
| rsi_12 | 12日RSI | Tushare Pro |
| rsi_24 | 24日RSI | Tushare Pro |
| days_to_t1 | 距离T1的天数 | 计算（-34到-1） |
| label | 样本标签 | 1（正样本） |

---

## 📉 负样本筛选条件（V2 - 同周期其他股票法）

### 一、筛选逻辑

| 步骤 | 说明 |
|------|------|
| 1. 获取正样本T1日期 | 按T1日期分组正样本 |
| 2. 获取有效股票池 | 应用统一过滤规则（ST、退市、退市整理期、北交所） |
| 3. 排除正样本股票 | 从股票池中排除已在正样本中的股票 |
| 4. 检查上市时间 | 确保股票在T1日期前已上市满180天 |
| 5. 检查停牌状态 | 使用 `suspend_d` 接口剔除T1日期停牌的股票 |
| 6. 随机选择 | 从符合条件的股票池中随机选择（每个正样本对应1个负样本） |

### 二、与正样本的区别

| 方面 | 正样本 | 负样本 |
|------|--------|--------|
| 来源 | 满足三连阳条件的股票 | 同T1日期的其他股票 |
| 行业/板块 | 无限制 | 无限制（不考虑行业板块） |
| 过滤规则 | ST、停牌、退市、退市整理期、北交所 | 完全相同 |
| 特征数据 | T1前34天 | T1前34天（完全相同） |
| 标签 | label=1 | label=0 |

---

## 📁 代码位置

### 正样本筛选器

```
src/models/screening/positive_sample_screener.py
```

关键方法：
- `_get_valid_stock_list()`: 获取有效股票列表（应用ST、退市、退市整理期、北交所过滤）
- `_check_three_week_pattern()`: 检查三周模式（包含停牌检查）
- `_merge_overlapping_samples()`: 处理重叠时间段去重

### 负样本筛选器

```
src/models/screening/negative_sample_screener_v2.py
```

关键方法：
- `_get_valid_stock_list()`: 获取有效股票列表（与正样本相同的过滤规则）
- `screen_negative_samples()`: 筛选负样本（包含停牌检查）

---

## 🔍 过滤规则实现详情

### 1. ST过滤

```python
# 剔除ST股票（ST、*ST、S*ST、SST等）
st_mask = stock_list['name'].str.contains('ST', na=False, case=False)
stock_list = stock_list[~st_mask]
```

### 2. HALT（停牌）过滤

```python
# 使用suspend_d接口查询T1日期停牌的股票
suspend_info = self.dm.get_suspend_info(trade_date=t1_date_str, suspend_type='S')
if not suspend_info.empty:
    suspended_stocks = set(suspend_info['ts_code'].tolist())
    eligible_stocks = eligible_stocks[~eligible_stocks['ts_code'].isin(suspended_stocks)]
```

### 3. DELISTING（退市）过滤

```python
# 使用list_status='L'只获取上市股票
stock_list = self.dm.get_stock_list(list_status='L')
```

### 4. DELISTING_SORTING（退市整理期）过滤

```python
# 剔除名称包含"退"的股票
delisting_sorting_mask = stock_list['name'].str.contains('退', na=False)
stock_list = stock_list[~delisting_sorting_mask]
```

### 5. 北交所过滤

```python
# 剔除代码以.BJ结尾的股票
bj_mask = stock_list['ts_code'].str.endswith('.BJ')
stock_list = stock_list[~bj_mask]
```

### 6. 上市时间过滤

```python
# 至少上市180天
days_since_list = (t1_date - list_date).days
if days_since_list < 180:
    return None
```

---

## 📊 日志输出示例

运行筛选脚本时，会输出详细的过滤统计：

```
股票过滤统计:
  原始数量: 5500
  剔除ST: 120
  剔除北交所: 280
  剔除退市整理期: 5
  有效股票: 5095
```

---

**文档版本**: v2.0
**创建日期**: 2025-12-28
**最后更新**: 2025-12-30
