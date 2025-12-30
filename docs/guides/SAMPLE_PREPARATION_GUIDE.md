# 样本数据准备指南 📊

完整的正负样本数据准备流程说明

---

## 🎯 概述

本指南介绍如何准备用于机器学习模型训练的正负样本数据。

### 数据流程

```
1. 正样本筛选
   ↓
2. 正样本特征提取
   ↓
3. 负样本筛选（同周期其他股票法）
   ↓
4. 负样本特征提取
   ↓
5. 合并数据用于模型训练
```

---

## 🔧 统一过滤规则

正样本和负样本均使用以下过滤规则：

| 过滤类型 | 说明 | 实现方式 |
|---------|------|---------|
| **ST** | 剔除ST股票 | 名称包含 `ST`（含ST、*ST、S*ST、SST等） |
| **HALT** | 剔除停牌股票 | 使用 `suspend_d` 接口查询T1日期停牌的股票 |
| **DELISTING** | 剔除退市股票 | 使用 `list_status='L'` 只获取上市股票 |
| **DELISTING_SORTING** | 剔除退市整理期股票 | 名称包含 `退` 字 |
| **北交所** | 剔除北交所股票 | 股票代码以 `.BJ` 结尾 |
| **上市时间** | 上市满180天 | T1日期与上市日期间隔 ≥ 180天 |

---

## 📋 正样本筛选规则

### 一、筛选符合条件的股票和T1

筛选历史上满足以下**所有条件**的股票：

| 条件 | 说明 |
|------|------|
| 1. 周K三连阳 | 连续3周收盘价 > 开盘价 |
| 2. 总涨幅 > 50% | (第3周收盘价 - 第1周开盘价) / 第1周开盘价 > 50% |
| 3. 最高涨幅 > 70% | (3周内最高价 - 第1周开盘价) / 第1周开盘价 > 70% |
| 4. 过滤规则 | 剔除ST、停牌(HALT)、退市(DELISTING)、退市整理期(DELISTING_SORTING)、北交所 |
| 5. 上市时间 ≥ 180天 | 上市日期距离T1至少180天（半年） |
| 6. T1定义 | T1 = 第一周的第一个交易日 |
| 7. 去重规则 | 重叠时间段合并（选最早T1），不重叠时间段分别保留为不同样本 |

### 二、提取T1前34天的交易数据

对每个筛选出的股票，提取T1之前的34个交易日数据，包含以下字段：

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

---

## 📉 负样本筛选规则（V2 - 同周期其他股票法）

### 一、筛选逻辑

基于正样本的T1日期，在同一日期选择其他股票作为负样本：

| 步骤 | 说明 |
|------|------|
| 1 | 获取正样本的T1日期列表 |
| 2 | 对每个T1日期，从所有股票中排除正样本股票 |
| 3 | 应用统一过滤规则（ST、停牌、退市、退市整理期、北交所） |
| 4 | 确保股票在T1日期前已上市满180天 |
| 5 | 从符合条件的股票池中随机选择 |
| 6 | 不考虑行业、板块等因素 |

### 二、负样本数量

- 每个正样本对应1个负样本
- 负样本总数 = 正样本总数

### 三、特征数据

与正样本相同，提取T1前34天的交易数据，额外添加：
- `label` = 0（负样本标签）

---

## 🚀 快速开始

### 1. 准备正样本

```bash
# 运行正样本筛选脚本
python scripts/prepare_positive_samples.py
```

**输出文件：**
- `data/training/samples/positive_samples.csv` - 正样本列表
- `data/training/features/feature_data_34d.csv` - 正样本特征数据（label=1）
- `data/training/samples/sample_statistics.json` - 统计报告

**预期结果：**
- 正样本数量：约1000-2000个（取决于时间范围）
- 特征数据量：正样本数 × 34

### 2. 准备负样本（V2）

```bash
# 运行负样本筛选脚本（需要先完成正样本）
python scripts/prepare_negative_samples_v2.py
```

**输出文件：**
- `data/training/samples/negative_samples_v2.csv` - 负样本列表
- `data/training/features/negative_feature_data_v2_34d.csv` - 负样本特征数据（label=0）
- `data/training/samples/negative_sample_statistics_v2.json` - 统计报告

### 3. 一键准备数据并训练

```bash
# 一键完成数据准备和模型训练
python scripts/prepare_data_and_retrain_v1.4.0.py
```

---

## 📁 数据格式

### 正样本列表格式

```csv
ts_code,name,t1_date,week1_start,week1_open,week3_end,week3_close,three_week_high,total_return,max_return,days_since_list
000006.SZ,深振业A,20241018,20241018,5.64,20241101,8.85,10.22,56.91,81.21,11862
```

### 负样本列表格式

```csv
ts_code,name,t1_date,days_since_list
000001.SZ,平安银行,20241018,8520
```

### 特征数据格式

```csv
sample_id,trade_date,name,ts_code,close,pct_chg,total_mv,circ_mv,ma5,ma10,volume_ratio,macd_dif,macd_dea,macd,rsi_6,rsi_12,rsi_24,days_to_t1,label
0,20241018,深振业A,000006.SZ,5.64,2.35,350000,280000,5.2,5.1,1.25,0.12,0.08,0.04,58.3,56.2,54.1,-34,1
```

---

## 💡 注意事项

### 1. 数据质量

- ✅ **完整性**：确保34天数据无缺失
- ✅ **一致性**：涨跌幅计算正确
- ✅ **准确性**：复权数据、交易日历准确
- ⚠️ **异常值**：检查极端涨幅（>200%）是否合理

### 2. 时间范围

- 建议从**2000年**开始（数据更全）
- 根据需求调整`START_DATE`

### 3. API限制

- Tushare积分越高，API调用越快
- 使用本地缓存避免重复调用
- 自动限流避免超限

### 4. 正负样本平衡

- 负样本数量 = 正样本数量
- 同周期选择确保时间分布一致
- 避免类别不平衡问题

---

## 🔍 日志输出示例

### 股票过滤统计

```
股票过滤统计:
  原始数量: 5500
  剔除ST: 120
  剔除北交所: 280
  剔除退市整理期: 5
  有效股票: 5095
```

### 负样本筛选

```
T1=20241018: 剔除 85 只停牌股票
进度: 100/2101 | 已生成负样本: 100
```

---

## 📊 下一步：模型训练

完成正负样本准备后，可以进行：

### 1. 数据合并

```python
import pandas as pd

# 加载正样本特征（label=1）
df_pos = pd.read_csv('data/training/features/feature_data_34d.csv')
df_pos['label'] = 1

# 加载负样本特征（label=0）
df_neg = pd.read_csv('data/training/features/negative_feature_data_v2_34d.csv')

# 合并
df_all = pd.concat([df_pos, df_neg], ignore_index=True)

# 打乱顺序
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
```

### 2. 模型训练

```bash
python scripts/train_breakout_launch_scorer.py --version v1.4.0 --neg-version v2
```

---

## 📚 相关文档

- [正负样本筛选逻辑说明](POSITIVE_SAMPLE_CRITERIA_COMPARISON.md) - 详细过滤规则
- [选股模型说明](STOCK_SELECTION_MODEL.md) - 模型逻辑
- [API参考文档](API_REFERENCE.md) - API使用

---

## ❓ 常见问题

### Q1: 正样本数量太少怎么办？

**A**: 调整筛选条件：
- 降低涨幅阈值（如40%、60%）
- 扩大时间范围（从2000年开始）
- 放宽上市时间限制（如3个月）

### Q2: 北交所股票为什么要剔除？

**A**: 原因：
- 流动性较差
- 数据质量可能不稳定
- 如需包含，修改`_get_valid_stock_list`方法

### Q3: 如何加快筛选速度？

**A**: 优化方法：
- 提高Tushare积分（更高频率）
- 使用本地缓存（避免重复调用）
- 调整时间范围（先测试小范围）

---

**文档版本**: v2.0  
**创建时间**: 2024-12-23  
**最后更新**: 2025-12-30
