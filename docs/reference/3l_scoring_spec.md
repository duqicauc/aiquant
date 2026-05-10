# 3L 评分系统技术规格书

## 1. 概述

3L（Three-Light）评分系统是 AIQuant 的多时间框架股票评分体系，通过**短期动量**、**中期概率**、**长期质量**三个维度对股票进行综合评估，辅助投资决策。

| 维度 | 名称 | 时间框架 | 核心问题 |
|------|------|----------|----------|
| L1 | 短期动量 | 5-10 天 | 股票短期是否有上涨动能？ |
| L2 | 中期概率 | 34 天 | 股票未来 34 天成为牛股的概率？ |
| L3 | 长期质量 | 120 天+ | 股票基本面是否支撑长期持有？ |

**设计目标：**
- 三灯输出均为**校准后的真实概率**（0-1），在同一尺度上可比
- 支持**共振评分**，综合三灯信号给出统一置信度
- 支持**回测验证**，每个过滤器的胜率可量化
- 支持**自动监控**，模型失效时自动回退到规则打分

---

## 2. 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          3L 多时间框架模型架构                             │
├─────────────────────────────────────────────────────────────────────────┤
│  短期模型              中期模型（已有 v294）            长期模型           │
│  ├─ ShortTermScorer   ├─ EnsemblePredictor            ├─ LongTermScorer  │
│  ├─ LightGBM 单模型   ├─ XGB/LGB/Cat 集成            ├─ LightGBM 单模型 │
│  ├─ 预测周期: 5天     ├─ 预测周期: 34天              ├─ 预测周期: 120天 │
│  ├─ 标注: 涨幅>=5%    ├─ 标注: 涨幅>=30%             ├─ 标注: 超额>=10% │
│  │   且回撤>=-3%      │   (已有标注)                 │   (跑赢大盘)     │
│  ├─ 特征: 动量/技术   ├─ 特征: UnifiedFeature        ├─ 特征: 基本面     │
│  │   RSI/MACD/量比    │   (80+ Tushare因子)          │   PE/PB/ROE/趋势 │
│  └─ 输出: prob_short  └─ 输出: prob                  └─ 输出: prob_long │
│                                                                          │
│  三个模型独立推理，互不依赖                                                │
│  中期模型是核心（已有），短期/长期是辅助过滤器（新增）                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 数据流

```
Model Inference (v294)
    ↓
predictions_YYYYMMDD_all.csv
    ↓
scripts/enrich_predictions.py
    ├── 加载 ShortTermScorer 模型 → prob_short
    ├── 加载 LongTermScorer 模型 → prob_long
    ├── 回退：模型不存在时启用规则打分
    ├── classify_market_stage() → market_stage
    ├── calc_left_side_signal() → left_side_signal
    └── calc_resonance() → resonance_score
    ↓
predictions_YYYYMMDD_all_enriched.csv
    ↓
Frontend / API
```

---

## 3. 模型详情

### 3.1 短期动量模型 (ShortTermScorer)

**标签定义：**
- 正样本：未来 5 日收盘价涨幅 ≥ 5% **且** 未来 5 日最大回撤 ≥ -3%
- 负样本：不满足正样本条件的所有股票

**特征集（20 个）：**

| 特征 | 来源 | 说明 |
|------|------|------|
| rsi_6, rsi_12, rsi_24 | Tushare stk_factor | RSI 多周期 |
| macd, macd_dif, macd_dea | Tushare stk_factor | MACD 指标 |
| kdj_k, kdj_d, kdj_j | Tushare stk_factor | KDJ 指标 |
| return_1d, return_3d, return_5d, return_10d | ohlcv | 多周期涨幅 |
| vol_ratio | Tushare stk_factor | 量比 |
| turnover_rate | Tushare daily_basic | 换手率 |
| volatility_5d, volatility_10d | ohlcv | 波动率 |
| excess_return_5d | ohlcv + 大盘 | 相对大盘超额收益 |
| close_ma20_ratio, close_ma60_ratio | ohlcv | 价格在均线上方/下方幅度 |

**模型：** LightGBM 单模型（`binary` 目标）
**校准：** Platt Scaling（Logistic Regression 拟合 OOF 预测）
**训练频率：** 每月 1 日自动重训练

### 3.2 长期质量模型 (LongTermScorer)

**标签定义：**
- 正样本：未来 120 日**超额收益**（个股收益 - 大盘收益）≥ 10%
- 负样本：不满足正样本条件的所有股票

**特征集（15 个）：**

| 特征 | 来源 | 说明 |
|------|------|------|
| pe, pb | Tushare daily_basic | 估值倍数 |
| pe_industry_zscore, pb_industry_zscore | daily_basic + industry | 行业内 z-score |
| total_mv_log, circ_mv_log | daily_basic | 市值对数 |
| turnover_rate, volume_ratio | daily_basic | 流动性 |
| return_20d, return_60d, return_120d | ohlcv | 长期动量 |
| volatility_60d, volatility_120d | ohlcv | 长期波动率 |
| close_ma60_ratio, close_ma120_ratio | ohlcv | 长期均线位置 |
| max_drawdown_60d | ohlcv | 60 日最大回撤 |
| trend_strength_60d | ohlcv | 60 日趋势强度（斜率/标准误） |

**模型：** LightGBM 单模型（`binary` 目标）
**校准：** Platt Scaling
**训练频率：** 每月 1 日自动重训练

### 3.3 回退规则打分

当模型文件不存在或监控触发回退时，enrich 脚本自动启用规则打分：

**短期规则（calc_prob_short_rule）：**
- 基础分 0.5
- RSI 50-65：+0.2；RSI > 65：+0.1
- MACD > 0：+0.15
- 近 5 日涨幅 5%-20%：+0.15；>20%：-0.1
- 量比 1.2-3.0：+0.1

**长期规则（calc_prob_long_rule）：**
- 基础分 0.5
- PE 10-30：+0.2；PE < 10：+0.1；PE > 50：-0.15
- PB 1-3：+0.15；PB > 5：-0.1
- 市值 50-500 亿：+0.1

---

## 4. 共振评分

### 4.1 公式

```
resonance_score = w_short * prob_short
                + w_mid   * prob
                + w_long  * prob_long
                + w_stage * stage_bonus
```

默认权重：
- w_short = 0.30
- w_mid   = 0.40
- w_long  = 0.20
- w_stage = 0.10

### 4.2 阶段加成

| 条件 | 加成 |
|------|------|
| 三灯全绿 + 拉升初期 | +0.15 |
| 三灯全绿 + 拉升中期 | +0.10 |
| 两绿一黄 + 拉升初期 | +0.05 |
| 任何一红 | 0 |

**三灯颜色定义：**
- 🟢 绿：概率 ≥ 0.70
- 🟡 黄：概率 ≥ 0.50
- ⚪ 灰：概率 < 0.50

---

## 5. 3L 过滤器

### 5.1 L1 动量主线

```
prob_short >= 0.50
AND market_stage IN ("拉升初期", "拉升中期")
```

### 5.2 L2 最强逻辑

```
prob_long >= 0.50
AND market_stage NOT IN ("下跌", "顶部")
```

### 5.3 L3 量价择时

```
(left_side_signal 信号数量 >= 2)
OR market_stage IN ("筑底", "拉升初期")
```

### 5.4 左侧信号

| 信号 | 条件 |
|------|------|
| RSI超卖 | RSI < 35 |
| 缩量 | 量比 < 0.7 |
| 深度回调 | 近20日跌幅 > 15% |
| 止跌迹象 | 近5日跌 > 5% 且 近1日跌 < 2% |

---

## 6. 配置

所有阈值和权重均在 `config/3l_scoring.yaml` 中配置，**修改无需改代码**。

关键配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `filters.l1_momentum.min_prob` | 0.50 | L1 概率阈值 |
| `filters.l2_quality.min_prob` | 0.50 | L2 概率阈值 |
| `filters.l3_timing.min_left_signals` | 2 | L3 左侧信号最小数量 |
| `resonance.weights.short` | 0.30 | 短期权重 |
| `resonance.weights.mid` | 0.40 | 中期权重 |
| `resonance.weights.long` | 0.20 | 长期权重 |
| `strike_zone.min_resonance` | 0.75 | 击球区共振阈值 |
| `fallback.auc_threshold` | 0.55 | 模型回退 AUC 阈值 |
| `fallback.consecutive_days` | 14 | 连续低 AUC 天数触发回退 |

---

## 7. 监控

### 7.1 每日监控项

| 监控项 | 阈值 | 告警级别 |
|--------|------|----------|
| 短期模型 AUC | < 0.55 | warning / error |
| 长期模型 AUC | < 0.55 | warning / error |
| 预测分布 KL 散度 | > 0.10 | warning |
| 模型文件缺失 | — | warning |

### 7.2 监控报告

每日生成：`data/monitoring/3l/report_YYYYMMDD.json`

### 7.3 自动回退

当模型 AUC 连续 `consecutive_days`（默认 14）天低于 `auc_threshold`（默认 0.55）时：
1. enrich 脚本自动切换为规则打分
2. 前端显示 🔴 模型漂移标签
3. 触发训练脚本重新训练

---

## 8. 回测

### 8.1 回测脚本

```bash
python scripts/backtest_3l_filters.py \
    --start-date 20250101 \
    --end-date 20251231 \
    --hold-days 34
```

### 8.2 回测组合

| 组合 | 过滤条件 |
|------|----------|
| 全部 | 无过滤 |
| L1 | 仅 L1 |
| L2 | 仅 L2 |
| L3 | 仅 L3 |
| L1+L2 | L1 AND L2 |
| L1+L2+L3_共振 | L1 AND L2 AND L3 |
| 高共振>=0.75 | resonance >= 0.75 |
| L1+L2+高共振 | L1 AND L2 AND resonance >= 0.75 |

### 8.3 输出指标

- 胜率（未来收益 > 0 的比例）
- 平均收益
- 中位数收益
- 最大回撤
- 夏普比率

---

## 9. 训练

### 9.1 首次训练

```bash
python scripts/train_3l_models.py --init
```

使用过去 2 年数据训练短期和长期模型。

### 9.2 增量训练

```bash
python scripts/train_3l_models.py --start-date 20240101 --end-date 20241231
```

### 9.3 训练输出

```
data/models/short_term_scorer/versions/v1.0.0/model/
├── lightgbm.txt          # LightGBM 模型
├── feature_names.json    # 特征列表
├── calibrator.pkl        # Platt 校准器
├── metrics.json          # 训练指标（AUC、特征重要性等）
└── metadata.json         # 模型元数据

data/models/long_term_scorer/versions/v1.0.0/model/
├── ...（同上）
```

---

## 10. 变更日志

| 日期 | 版本 | 变更内容 | 作者 |
|------|------|----------|------|
| 2026-05-05 | v1.0.0 | 首次发布：新增 ShortTermScorer、LongTermScorer，替换原有规则打分，引入共振评分和自动回退机制 | AI Agent |

---

*文档与代码同步更新。任何对 3L 评分逻辑的修改必须同步更新本文档（AGENTS.md 原则 #30）。*
