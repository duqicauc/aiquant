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
3. 正样本特征统计分析
   ↓
4. 基于统计特征筛选负样本
   ↓
5. 负样本特征提取
   ↓
6. 合并数据用于模型训练
```

---

## 📋 正样本筛选规则

### 一、筛选符合条件的股票和T1

筛选历史上满足以下**所有条件**的股票：

| 条件 | 说明 |
|------|------|
| 1. 周K三连阳 | 连续3周收盘价 > 开盘价 |
| 2. 总涨幅 | (第3周收盘价 - 第1周开盘价) / 第1周开盘价 > 50% |
| 3. 最高涨幅 | (3周内最高价 - 第1周开盘价) / 第1周开盘价 > 70% |
| 4. 剔除特殊股票 | 剔除ST、*ST、HALT（停牌）、DELISTING（退市） |
| 5. 剔除北交所 | 剔除股票代码以.BJ结尾的股票 |
| 6. 上市时间 | 上市日期距离T1至少180天（半年） |
| 7. T1定义 | T1 = 第一周的第一个交易日 |
| 8. 去重 | 同一股票有多个时间段满足条件时，选日期最小的 |

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

## 📉 负样本筛选规则 🆕

### 一、分析正样本特征统计

对所有正样本的34天数据，统计以下特征的分布：

1. **量比>2的次数**
   - 统计34天内量比大于2的交易日数量
   - 计算：平均值、中位数、Q25、Q75、最小值、最大值

2. **MACD负转正的次数**
   - 统计MACD_DIF从负数变为正数的次数
   - 计算：平均值、中位数、Q25、Q75、最小值、最大值

3. **34天累计涨跌幅**
   - 计算：(1+r1/100) × (1+r2/100) × ... × (1+r34/100) - 1
   - 统计：平均值、中位数、Q25、Q75、最小值、最大值

### 二、筛选负样本

使用正样本的特征统计，在全部历史数据中搜索符合条件的连续34个交易日：

| 条件 | 说明 |
|------|------|
| 1. 量比条件 | 量比>2的次数在[Q25, Q75]范围内 |
| 2. MACD条件 | MACD负转正次数在[Q25, Q75]范围内 |
| 3. 涨跌幅条件 | 累计涨跌幅在[Q25, Q75]范围内 |
| 4. 排除正样本 | 排除已在正样本中的股票和时间段 |
| 5. 股票限制 | 使用与正样本相同的股票筛选规则 |
| 6. 数量控制 | 负样本数量与正样本数量相同 |

### 三、提取负样本特征

对每个负样本，提取34天的交易数据，字段与正样本相同，额外添加：

- `label` = 0（负样本标签）

---

## 🚀 快速开始

### 1. 准备正样本

```bash
# 运行正样本筛选脚本
python scripts/prepare_positive_samples.py
```

**输出文件：**
- `data/processed/positive_samples.csv` - 正样本列表
- `data/processed/feature_data_34d.csv` - 正样本特征数据（label=1）
- `data/processed/sample_statistics.json` - 统计报告

**预期结果：**
- 正样本数量：约1000-2000个（取决于时间范围）
- 特征数据量：正样本数 × 34

### 2. 准备负样本 🆕

```bash
# 运行负样本筛选脚本（需要先完成正样本）
python scripts/prepare_negative_samples.py
```

**输出文件：**
- `data/processed/negative_samples.csv` - 负样本列表
- `data/processed/negative_feature_data_34d.csv` - 负样本特征数据（label=0）
- `data/processed/negative_sample_statistics.json` - 统计报告

**预期结果：**
- 负样本数量：与正样本数量相同
- 特征数据量：负样本数 × 34

### 3. 数据质量检查

```bash
# 自动质量检查（7大类检查）
python scripts/check_sample_quality.py

# 可视化分析
python scripts/visualize_sample_quality.py
```

**检查内容：**
- ✅ 基础统计
- ✅ 数据完整性
- ✅ 数据一致性
- ✅ 涨幅计算验证
- ✅ 异常值检测
- ✅ 重复数据检查
- ✅ 日期合理性

**质量评分：**
- 100分：优秀 ⭐⭐⭐⭐⭐
- 80-99分：良好 ⭐⭐⭐⭐
- 60-79分：一般 ⭐⭐⭐
- 40-59分：较差 ⭐⭐
- <40分：差 ⭐

---

## 📁 数据格式

### 正样本列表格式

```csv
ts_code,name,t1_date,week1_start,week1_open,week3_end,week3_close,three_week_high,total_return,max_return,days_since_list
000006.SZ,深振业A,20241018,20241018,5.64,20241101,8.85,10.22,56.91,81.21,11862
```

### 负样本列表格式 🆕

```csv
ts_code,name,start_date,end_date,volume_ratio_gt2_count,macd_turn_positive_count,cumulative_return_34d
000001.SZ,平安银行,20220301,20220415,5,2,12.35
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

- 建议从**2022年**开始（数据较新）
- 或从**2000年**开始（数据更全）
- 根据需求调整`START_DATE`

### 3. API限制

- Tushare积分越高，API调用越快
- 使用本地缓存避免重复调用
- 自动限流避免超限

### 4. 负样本平衡 🆕

- 负样本数量 = 正样本数量
- 负样本特征与正样本相似（Q25-Q75范围）
- 避免类别不平衡问题

---

## 🔍 数据验证

### 人工抽样验证

1. **随机抽样**（10-20个）
   ```bash
   # 使用Excel或pandas查看
   import pandas as pd
   df = pd.read_csv('data/processed/positive_samples.csv')
   sample = df.sample(10)
   ```

2. **重点验证**
   - 极端涨幅样本（>200%）
   - 边缘样本（50-55%）
   - MACD负转正次数异常的样本

3. **交叉验证**
   - 使用同花顺/东方财富查看K线图
   - 验证涨幅计算是否正确
   - 检查是否有遗漏的ST股票

详见：[数据质量核查指南](QUALITY_CHECK_GUIDE.md)

---

## 📊 下一步：模型训练

完成正负样本准备后，可以进行：

### 1. 数据合并

```python
import pandas as pd

# 加载正样本特征（label=1）
df_pos = pd.read_csv('data/processed/feature_data_34d.csv')
df_pos['label'] = 1

# 加载负样本特征（label=0）
df_neg = pd.read_csv('data/processed/negative_feature_data_34d.csv')

# 合并
df_all = pd.concat([df_pos, df_neg], ignore_index=True)

# 打乱顺序
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
```

### 2. 特征工程

- 标准化/归一化
- 特征选择（相关性分析、重要性排序）
- 创建新特征（技术指标组合）

### 3. 划分数据集

```python
from sklearn.model_selection import train_test_split

X = df_all.drop(['label', 'sample_id', 'trade_date', 'ts_code', 'name'], axis=1)
y = df_all['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 4. 模型训练

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)
```

### 5. 模型评估

```python
from sklearn.metrics import classification_report, roc_auc_score

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

---

## 📚 相关文档

- [快速开始指南](QUICK_START_GUIDE.md) - 完整流程
- [选股模型说明](STOCK_SELECTION_MODEL.md) - 详细逻辑
- [数据质量工作流程](DATA_QUALITY_WORKFLOW.md) - 质量保证
- [Tushare Pro功能](TUSHARE_PRO_FEATURES.md) - 高级功能
- [API参考文档](API_REFERENCE.md) - API使用

---

## ❓ 常见问题

### Q1: 正样本数量太少怎么办？

**A**: 调整筛选条件：
- 降低涨幅阈值（如40%、60%）
- 扩大时间范围（从2000年开始）
- 放宽上市时间限制（如3个月）

### Q2: 负样本特征不够相似怎么办？

**A**: 调整筛选范围：
- 使用[Q10, Q90]而不是[Q25, Q75]
- 增加特征维度
- 调整负样本数量上限

### Q3: 数据质量评分低怎么办？

**A**: 检查问题类型：
- 数据完整性问题：重新获取数据
- 异常值问题：人工验证是否合理
- 重复数据问题：检查去重逻辑

### Q4: 北交所股票为什么要剔除？

**A**: 原因：
- 流动性较差
- 数据质量可能不稳定
- 如需包含，修改`_get_valid_stock_list`方法

### Q5: 如何加快筛选速度？

**A**: 优化方法：
- 提高Tushare积分（更高频率）
- 使用本地缓存（避免重复调用）
- 调整时间范围（先测试小范围）

---

**文档版本**: v1.0  
**创建时间**: 2024-12-23  
**最后更新**: 2024-12-23


