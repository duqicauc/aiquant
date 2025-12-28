# 左侧起爆点模型开发文档 🚀

## 模型概述

这是一个专门用于识别"左侧潜力牛股起爆点"的模型，核心目标是：
- **低风险高收益的不对称交易**
- **寻找起爆点，减少时间成本**
- **在上涨趋势开始前（左侧）识别潜力股**

## 模型思路

### 核心逻辑

与现有模型（三连阳模型）不同，本模型关注的是**起爆前的特征**，而不是起爆后的特征。

**现有模型（三连阳模型）**：
- 正样本：已经出现三连阳且涨幅>50%的股票
- 特征：T1前34天的数据（起爆前）
- 目标：识别已经起爆的股票

**新模型（左侧起爆点模型）**：
- 正样本：在起爆点前1-2周处于底部区域，随后3周内涨幅>50%的股票
- 特征：起爆点前1-2周的特征（更关注底部形态、成交量、技术指标背离）
- 目标：在起爆前识别潜力股

### 正样本定义

对于每只股票，找到满足以下条件的起爆点：

1. **起爆点T0定义**：
   - T0后3周内，总涨幅>50%
   - T0后3周内，最高涨幅>70%

2. **起爆前特征（T0前1-2周）**：
   - 价格处于相对低位（过去60天内的低点附近）
   - 成交量萎缩后开始放量（量比>1.5）
   - 技术指标出现背离或企稳信号
   - 价格接近或突破关键均线（MA5/MA10）

3. **排除条件**：
   - ST股票
   - 上市不足180天
   - 北交所股票
   - 停牌/退市股票

### 负样本定义

1. **方法1（推荐）**：同周期其他股票法
   - 对每个正样本，在同一T0日期选择其他股票作为负样本
   - 这些股票在T0后3周内涨幅<20%（未起爆）

2. **方法2**：特征相似法
   - 找到与正样本特征相似（Q25-Q75范围）但未起爆的股票

### 特征工程

重点关注以下特征：

#### 1. 底部形态特征
- `price_position_60d`: 当前价格在60天内的位置（0-1，0=最低，1=最高）
- `price_near_low`: 是否接近60天低点（距离<5%）
- `price_recovery`: 从低点反弹幅度

#### 2. 成交量特征
- `volume_ratio_recent`: 最近5天平均量比
- `volume_surge`: 成交量是否突然放大（当日量比>1.5且前5日均量比<1.0）
- `volume_trend`: 成交量趋势（最近5天 vs 前5天）

#### 3. 技术指标特征
- `rsi_oversold`: RSI是否超卖（<30）
- `macd_divergence`: MACD是否出现背离
- `macd_cross`: MACD是否金叉
- `kdj_oversold`: KDJ是否超卖

#### 4. 均线系统特征
- `price_above_ma5`: 价格是否突破MA5
- `price_above_ma10`: 价格是否突破MA10
- `ma5_above_ma10`: MA5是否上穿MA10
- `distance_to_ma5`: 价格距离MA5的距离
- `distance_to_ma10`: 价格距离MA10的距离

#### 5. 动量特征
- `momentum_5d`: 5日动量
- `momentum_10d`: 10日动量
- `momentum_change`: 动量变化（最近5天 vs 前5天）

#### 6. 波动率特征
- `volatility_20d`: 20日波动率
- `volatility_compression`: 波动率是否压缩（最近波动率<历史波动率）

## 数据组织

### 目录结构

```
data/
├── training/
│   ├── models/
│   │   ├── breakout/              # 新模型目录
│   │   │   ├── samples/          # 样本数据
│   │   │   │   ├── positive_samples.csv
│   │   │   │   └── negative_samples.csv
│   │   │   ├── features/         # 特征数据
│   │   │   │   ├── positive_features.csv
│   │   │   │   └── negative_features.csv
│   │   │   ├── models/           # 模型文件
│   │   │   │   └── xgboost_breakout_*.json
│   │   │   └── metrics/          # 评估指标
│   │   │       └── metrics_*.json
│   │   └── timeseries/           # 现有模型目录（保持不变）
│   └── ...
```

## 训练流程

### 1. 准备正样本

```bash
python scripts/prepare_breakout_positive_samples.py
```

**输出**：
- `data/training/models/breakout/samples/positive_samples.csv`
- `data/training/models/breakout/features/positive_features.csv`

### 2. 准备负样本

```bash
python scripts/prepare_breakout_negative_samples.py
```

**输出**：
- `data/training/models/breakout/samples/negative_samples.csv`
- `data/training/models/breakout/features/negative_features.csv`

### 3. 训练模型

```bash
python scripts/train_breakout_model.py
```

**输出**：
- `data/training/models/breakout/models/xgboost_breakout_*.json`
- `data/training/models/breakout/metrics/metrics_*.json`

### 4. 评估模型

```bash
python scripts/evaluate_breakout_model.py
```

## 评测流程

### 评估指标

1. **分类指标**：
   - Accuracy（准确率）
   - Precision（精确率）
   - Recall（召回率）
   - F1-Score

2. **业务指标**：
   - 起爆点识别准确率（T0后3周内是否真的起爆）
   - 平均起爆时间（从T0到实际起爆的天数）
   - 平均涨幅（起爆后的实际涨幅）

3. **回测指标**：
   - 年化收益率
   - 最大回撤
   - 夏普比率
   - 胜率

## 预测工具

新模型可以与现有模型共用预测工具，通过模型名称区分：

```python
# 使用左侧起爆点模型
python scripts/score_current_stocks.py --model breakout

# 使用现有三连阳模型
python scripts/score_current_stocks.py --model timeseries
```

## 模型对比

| 特性 | 三连阳模型 | 左侧起爆点模型 |
|------|-----------|---------------|
| **目标** | 识别已起爆的股票 | 识别起爆前的潜力股 |
| **正样本** | 三连阳+涨幅>50% | 起爆前1-2周特征+后续涨幅>50% |
| **特征重点** | 起爆前34天整体特征 | 起爆前1-2周底部形态特征 |
| **优势** | 识别确定性高 | 提前布局，时间成本低 |
| **劣势** | 起爆后才识别 | 可能误判，需要更多验证 |

## 使用建议

1. **组合使用**：
   - 左侧起爆点模型：用于提前布局
   - 三连阳模型：用于确认起爆

2. **风险控制**：
   - 左侧模型信号需要结合基本面验证
   - 设置止损位（-10%）
   - 分批建仓

3. **时间管理**：
   - 左侧模型：持有时间可能较长（1-3个月）
   - 三连阳模型：持有时间较短（3-6周）

---

**文档版本**: v1.0  
**创建时间**: 2024-12-25  
**模型状态**: 开发中

