# 负样本筛选方案对比 ⚖️

两种负样本筛选方法的详细对比与使用指南

---

## 📋 方案概览

| 方案 | 方法 | 文件 | 特点 |
|------|------|------|------|
| **方案1 (V1)** | 基于特征统计 | `negative_sample_screener.py` | 负样本与正样本特征相似 |
| **方案2 (V2)** | 同周期其他股票 | `negative_sample_screener_v2.py` | 随机选择，更真实 |

---

## 🔍 详细对比

### 方案1：基于特征统计筛选

#### 筛选逻辑

```
1. 统计正样本T1前34天的特征分布
   - 量比>2的次数
   - MACD负转正次数
   - 34天累计涨跌幅

2. 使用Q25-Q75范围作为筛选条件

3. 在全部历史数据中搜索符合条件的连续34天

4. 排除已在正样本中的股票和时间段
```

#### 优点 ✅

| 优点 | 说明 |
|------|------|
| 🎯 **针对性强** | 负样本特征与正样本相似，模型更有挑战 |
| 🧠 **学习精细** | 模型学习"在相似特征下区分真正的牛股" |
| 📈 **泛化能力强** | 避免过于明显的负样本，提高泛化 |
| 🔬 **科学严谨** | 基于统计学方法，有理论支撑 |

#### 缺点 ⚠️

| 缺点 | 说明 |
|------|------|
| ⏱️ **耗时较长** | 需要遍历所有股票所有时间段，计算特征 |
| 🔢 **数据量可能不足** | 严格的筛选条件可能导致负样本不够 |
| 💻 **实现复杂** | 需要处理特征计算、日期匹配等 |
| 🎲 **可能过拟合** | 过于相似可能导致模型学不到本质差异 |

#### 适用场景

- ✅ 追求模型精度，愿意花时间训练
- ✅ 正样本特征明显，需要"难负样本"
- ✅ 数据充足，不担心负样本数量

---

### 方案2：同周期其他股票法

#### 筛选逻辑

```
1. 对每个正样本，获取其T1日期

2. 在同一T1日期，从所有股票中随机选择
   （排除正样本股票）

3. 提取这些股票T1前34天的数据作为负样本

4. 确保符合基本条件（非ST、非北交所等）
```

#### 优点 ✅

| 优点 | 说明 |
|------|------|
| ⚡ **速度极快** | 不需要复杂的特征计算，直接随机选择 |
| 📊 **数据充足** | 股票池大，负样本数量不受限 |
| 🌍 **真实场景** | 更接近实际应用：从所有股票中挑选 |
| 🎲 **多样性高** | 涵盖各种类型的股票，特征分布广 |
| 🔧 **实现简单** | 代码简洁，易于维护和修改 |

#### 缺点 ⚠️

| 缺点 | 说明 |
|------|------|
| 📉 **可能太简单** | 负样本可能包含极差股票，区分度太明显 |
| ⚖️ **分布不均** | 特征分布与正样本差异可能很大 |
| 🎯 **学习重点偏移** | 模型可能学到"好股票vs差股票"而非"潜力股vs普通股" |

#### 适用场景

- ✅ 需要快速迭代实验
- ✅ 数据量优先
- ✅ 模拟真实选股场景
- ✅ 初期探索，快速建立baseline

---

## 🚀 使用指南

### 方案1使用

```bash
# 准备负样本（V1 - 基于特征统计）
python scripts/prepare_negative_samples.py
```

**输出文件：**
```
data/processed/
├── negative_samples.csv              # 负样本列表
├── negative_feature_data_34d.csv     # 负样本特征
└── negative_sample_statistics.json   # 统计报告
```

**预计时间：** 1-2小时（取决于时间范围和股票数）

---

### 方案2使用

```bash
# 准备负样本（V2 - 同周期其他股票）
python scripts/prepare_negative_samples_v2.py
```

**输出文件：**
```
data/processed/
├── negative_samples_v2.csv           # 负样本列表
├── negative_feature_data_v2_34d.csv  # 负样本特征
└── negative_sample_statistics_v2.json # 统计报告
```

**预计时间：** 10-30分钟

---

## 🔬 对比实验设计

### 实验1：单独使用V1

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# 加载数据
df_pos = pd.read_csv('data/processed/feature_data_34d.csv')
df_pos['label'] = 1

df_neg = pd.read_csv('data/processed/negative_feature_data_34d.csv')
# label=0已包含

# 合并
df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42)

# 训练模型...
```

### 实验2：单独使用V2

```python
# 加载数据
df_pos = pd.read_csv('data/processed/feature_data_34d.csv')
df_pos['label'] = 1

df_neg = pd.read_csv('data/processed/negative_feature_data_v2_34d.csv')
# label=0已包含

# 合并
df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42)

# 训练模型...
```

### 实验3：混合使用（50% + 50%）

```python
# 加载数据
df_pos = pd.read_csv('data/processed/feature_data_34d.csv')
df_pos['label'] = 1

df_neg_v1 = pd.read_csv('data/processed/negative_feature_data_34d.csv')
df_neg_v2 = pd.read_csv('data/processed/negative_feature_data_v2_34d.csv')

# 各取50%
n_samples = min(len(df_neg_v1), len(df_neg_v2))
df_neg_v1_sample = df_neg_v1.sample(n=n_samples//2, random_state=42)
df_neg_v2_sample = df_neg_v2.sample(n=n_samples//2, random_state=42)

# 合并
df = pd.concat([df_pos, df_neg_v1_sample, df_neg_v2_sample])
df = df.sample(frac=1, random_state=42)

# 训练模型...
```

---

## 📊 评估指标

### 推荐评估指标

| 指标 | 说明 | 重要性 |
|------|------|--------|
| **Accuracy** | 整体准确率 | ⭐⭐⭐ |
| **Precision** | 预测为牛股的准确率 | ⭐⭐⭐⭐⭐ |
| **Recall** | 真正牛股被找出的比例 | ⭐⭐⭐⭐⭐ |
| **F1-Score** | Precision和Recall的调和平均 | ⭐⭐⭐⭐⭐ |
| **AUC-ROC** | 分类器性能综合指标 | ⭐⭐⭐⭐ |
| **混淆矩阵** | 详细的分类情况 | ⭐⭐⭐⭐ |

### 实际业务指标

- **Top-K准确率**：预测概率最高的K只股票中，实际牛股的比例
- **回测收益**：基于模型选股的回测收益率
- **夏普比率**：风险调整后收益

---

## 💡 使用建议

### 快速原型阶段

**推荐：方案2（V2）**

理由：
- ✅ 速度快，适合快速迭代
- ✅ 数据充足，不用担心样本不够
- ✅ 实现简单，出错概率低

### 模型优化阶段

**推荐：方案1（V1）或混合**

理由：
- ✅ 更有挑战性的负样本，提升模型精度
- ✅ 通过混合使用，兼顾两者优势

### 生产环境

**推荐：根据离线评估结果选择**

步骤：
1. 两种方案都训练模型
2. 在验证集上对比效果
3. 回测对比收益
4. 选择效果最好的方案

---

## 📈 预期效果差异

### 方案1（V1）

```
预期：
- 准确率：较高（85-92%）
- Precision：较高（80-88%）
- Recall：中等（65-75%）
- 特点：挑剔，精准但可能漏掉一些
```

### 方案2（V2）

```
预期：
- 准确率：较高（82-90%）
- Precision：中等（75-85%）
- Recall：较高（70-82%）
- 特点：更全面，但可能有误判
```

### 混合方案

```
预期：
- 平衡两者优势
- 更鲁棒
- 适应性强
```

---

## 🎯 决策树

```
开始
│
├─ 时间紧迫？
│  ├─ 是 → 使用V2
│  └─ 否 ↓
│
├─ 追求极致精度？
│  ├─ 是 → 使用V1
│  └─ 否 ↓
│
├─ 不确定哪个好？
│  └─ 是 → 两个都做，对比效果
│
└─ 最终建议：
    1. 先用V2快速建立baseline
    2. 再用V1提升精度
    3. 对比效果，选择最佳
```

---

## 🔧 高级技巧

### 1. 动态调整负样本比例

```python
# 根据正样本数量动态调整
positive_count = len(df_positive)

# 初期：1:1
negative_count_v2 = positive_count * 1

# 后期：可以增加负样本
# negative_count_v2 = positive_count * 2
```

### 2. 分层负样本

```python
# V1: 困难负样本（30%）
# V2: 普通负样本（70%）
df_neg_hard = df_neg_v1.sample(n=int(total_neg * 0.3))
df_neg_normal = df_neg_v2.sample(n=int(total_neg * 0.7))
```

### 3. 负样本质量筛选

对V2的结果进行后处理：
```python
# 排除极端情况
df_neg_v2_filtered = df_neg_v2[
    (df_neg_v2['pct_chg'].abs() < 20) &  # 单日涨跌幅不超过20%
    (df_neg_v2['volume_ratio'] < 10)     # 量比不超过10
]
```

---

## 📚 相关文档

- [样本准备完整指南](SAMPLE_PREPARATION_GUIDE.md) - 基础流程
- [选股模型说明](STOCK_SELECTION_MODEL.md) - 模型逻辑
- [数据质量核查](QUALITY_CHECK_GUIDE.md) - 质量保证

---

## ❓ 常见问题

### Q1: 两个方案可以同时使用吗？

**A**: 可以！推荐的做法是：
1. 先用V2快速训练baseline
2. 再用V1训练精细模型
3. 对比效果，选择最佳

### Q2: 哪个方案更适合新手？

**A**: V2更适合新手：
- 实现简单
- 速度快
- 容易理解
- 出错少

### Q3: 生产环境推荐哪个？

**A**: 根据离线评估结果选择：
- 如果两者差不多 → 选V2（速度快）
- 如果V1明显更好 → 选V1（精度高）
- 如果不确定 → 使用混合方案

### Q4: 负样本比例应该是多少？

**A**: 常见比例：
- **1:1** - 平衡，推荐起点
- **1:2** - 更多负样本，降低假阳性
- **1:0.5** - 更少负样本，提高召回率

根据实际业务需求调整。

### Q5: 如何判断负样本质量？

**A**: 检查指标：
- 特征分布是否合理
- 是否有极端异常值
- 时间分布是否均匀
- 行业分布是否多样

---

**文档版本**: v1.0  
**创建时间**: 2024-12-23  
**最后更新**: 2024-12-23


