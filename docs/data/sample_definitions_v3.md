# AIQuant 样本定义审计报告 v3.0

> **审计日期**: 2026-05-02
> **审计范围**: 正样本、负样本、硬负样本的定义、生成逻辑、特征提取
> **结论**: 当前样本系统存在**严重不一致**，必须重建数据管道后才能训练 3.x 模型

---

## 一、样本定义现状

### 1.1 正样本（Positive Samples）

| 项目 | 当前定义 |
|------|---------|
| **筛选逻辑** | 周线三连阳（close > open 连续3周） |
| **总涨幅阈值** | > 50%（week3.close vs week1.open） |
| **最高涨幅阈值** | > 70%（three_week_high vs week1.open） |
| **上市天数** | ≥ 180天（settings.yaml 配置为 300天） |
| **反追龙头** | T1前34天涨幅 ≤ 20%，日均波动率 ≤ 3% |
| **过滤规则** | 排除ST、北交所、退市、停牌 |
| **当前数量** | ~3,158 unique samples（3,324 rows） |
| **时间跨度** | 2000-01-07 ~ 2026 |

**问题发现**:
- `settings.yaml` 中 `min_listing_days: 300`，但 `breakout_launch_scorer.yaml` 中为 `180`
- `PositiveSampleScreener` 类默认配置为 `180`，存在配置漂移
- 正样本基于**周线**筛选，但特征提取使用**日线**T1前34天，周期错配问题显著

### 1.2 负样本（Negative Samples）

| 项目 | 当前定义 |
|------|---------|
| **筛选逻辑** | V2方法：同T1日期的其他随机股票 |
| **比例** | 2个负样本 / 1个正样本 |
| **上市天数** | ≥ 180天（T1前） |
| **过滤规则** | 排除正样本股票、ST、北交所、停牌 |
| **市值分层** | 可选（v3新增），但未在默认流程中启用 |
| **当前数量** | ~6,234 unique samples（6,506 rows） |

**问题发现**:
- 负样本是**完全随机**的同期其他股票，与正样本的特征分布可能存在系统性偏差
- 未验证市值、行业、波动率分布是否与正样本匹配
- 负样本过于"简单"——与正样本差异大，模型容易区分，无法学到精细模式

### 1.3 硬负样本（Hard Negative Samples）

| 项目 | 当前定义 |
|------|---------|
| **类型1: near_miss** | 34日涨幅 15%-35%（接近50%但未达标） |
| **类型2: high_position_fail** | T1前涨≥25%，T1后21天跌≤0% |
| **类型3: false_breakout** | 突破20日高点后5日内回落>5% |
| **采样数量** | 每类型每T1日 15/15/10 只 |
| **当前数量** | v291扩展后 ~2,111 个 |
| **历史比例** | 早期 15.54% (998/6422)，v291 后约 31.4% |

**问题发现**:
- `near_miss` 下限从 20% 降到 15%，上限从 45% 降到 35%，**边界越来越模糊**
- `high_position_fail` 使用 T1后21天数据判断"失败"，存在**未来函数嫌疑**
- `false_breakout` 的检测逻辑在 T1前40天内寻找，但 T1 的定义是正样本三周阳线的起点，时间窗口可能不匹配
- 硬负比例波动极大（1.7% → 15.54% → 31.4%），缺乏理论依据

---

## 二、特征提取不一致（严重问题）

### 2.1 代码层面的不一致

三类样本各自有独立的 `_extract_single_sample_features` 方法：

| 特征/数据源 | 正样本 | 负样本 | 硬负样本 |
|------------|--------|--------|---------|
| **特征提取类** | `PositiveSampleScreener` | `NegativeSampleScreenerV2` | `HardNegativeSampleScreener` |
| **获取原始数据** | `get_complete_data()` | `get_complete_data()` | `get_complete_data()` |
| **OHLCV完整性检查** | ✅ 有（v6修复） | ❌ 无 | ❌ 无 |
| **Tushare技术因子** | `get_stk_factor()` 12列 | `get_stk_factor()` 12列 | `get_stk_factor()` 12列 |
| **本地计算MA** | ma5, ma10 | ma5, ma10 | ma5, ma10 |
| **特征字段选择** | 14个基础 + 7个技术因子 | 9个基础 + 7个技术因子 | 9个基础 + 7个技术因子 |
| **label列** | 无（训练时添加） | 有（label=0） | 有（label=0） |

**关键差异**：
- 正样本原始提取包含 `open`, `high`, `low`, `vol`, `change`, `amount`
- 负样本和硬负样本原始提取**只包含 `close`**，没有 `open/high/low/vol`

### 2.2 Enhanced 数据中的 NaN 问题

虽然 enhanced 版本通过后续处理让列数看起来一致（178/179列），但**硬负样本中存在大量 100% NaN 的特征**：

```
硬负样本 100% NaN 的特征:
- breakout_confirmed_10d, breakout_confirmed_20d, breakout_resonance
- breakout_rsi_interaction, breakout_strength_10d/20d/55d
- breakout_volume_strength, breakout_with_volume
- market_momentum_5d/10d/20d, market_position_20d, market_regime
- momentum_market_interaction, relative_volatility, resonance_volume_confirm
- rsi_kdj_divergence, trend_consistency, volume_price_divergence
- excess_return_consistency, breakout_strength_avg, breakout_strength_max
```

**根因分析**：
- 硬负样本的 enhanced 数据生成路径与正/负样本不同
- `scripts/extract_hard_negative_features_v291.py` 等脚本使用独立的特征计算逻辑
- 市场环境特征（market_*）只在正样本和负样本的 enhanced 流程中计算，硬负样本遗漏
- `prev_high_10d/20d/55d` 在所有样本中都是 100% NaN，说明这个特征从未被正确计算

### 2.3 Tushare 因子使用不充分

`stk_factor_pro` 提供 80+ 专业因子，但当前训练数据只使用了约 19 个：

```python
# 实际替换的列（retrain_prepare_tushare_factors.py）
macd_dif, macd_dea, macd, rsi_6, rsi_12, rsi_24,
kdj_k, kdj_d, kdj_j, obv, ema_5, ema_10, ema_20, ema_60,
bias_short, bias_mid, bias_long, ma5, ma10, ma_20d, atr_14
```

**大量 Tushare 因子未被使用**：
- BOLL（上轨/中轨/下轨）
- CCI
- DMI（PDI/MDI/ADX/ADXR）
- WR（威廉指标）
- MFI（资金流量指标）
- MTM/ROC/MAROC
- PSY/PSYMA
- VR/CR/BRAR
- EMV/MAEMV
- BBI/DPO/DFMA
- KTN（肯特纳通道）
- TAQ（海龟通道）
- TRI/TRMA
- MASS/MA_MASS
- EXPMA_12/50
- ASI/ASIT
- XSII_TD1-4

---

## 三、数据管道架构问题

### 3.1 双存储系统不一致

| 存储 | daily_data | daily_basic | stk_factor |
|------|-----------|-------------|------------|
| SQLite | 12列 OHLCV | 8列 | 12列（旧） |
| ArcticDB | 完整 | 完整 | 完整（80+列） |

**问题**：`fill_missing_flat_data.py` 对 `stk_factor` 只写 12 列到 SQLite，导致部分代码路径获取不到完整因子。

### 3.2 样本→特征→训练的断点

```
正样本:  screening → _extract_features (老旧) → reextract_positive_features.py → enhanced?
负样本:  screening → _extract_features (老旧) → reextract_negative_features.py → enhanced?
硬负:    screening → _extract_features (老旧) → extract_hard_negative_features_v291.py → enhanced?
         ↓
    三套独立的提取脚本，各自有 bug 和遗漏
         ↓
    retrain_prepare_tushare_factors.py (部分替换 Tushare 因子)
         ↓
    compute_enhanced_market_features.py (市场环境特征，但硬负样本可能没合并)
         ↓
    训练脚本（发现 NaN → 崩溃）
```

---

## 四、修正建议（v3.0 重建方案）

### 4.1 样本定义统一

1. **正样本**：保持三连阳 + 50%/70% 阈值，但增加以下校验：
   - 统一 `min_listing_days = 300`（与 settings.yaml 一致）
   - 反追龙头约束默认启用（pre_t1_return_max=20, pre_t1_volatility_max=3）
   - 验证样本时间分布均匀性，避免牛市区间过度采样

2. **负样本**：从"随机其他股票"改为"特征匹配采样"：
   - 按市值、行业、波动率分层匹配正样本分布
   - 确保负样本与正样本在T1前的价格形态有可比性
   - 负/正比例保持 2:1

3. **硬负样本**：收紧定义，消除未来函数：
   - `near_miss`: 34日涨幅 **20%-40%**（下限回到20，避免太接近普通负样本）
   - `high_position_fail`: **禁止使用T1后数据**。改为：T1前已涨≥20%，且T1当日出现冲高回落（上影线>3%）
   - `false_breakout`: 保持突破20日高点后5日回落>5%，但必须在T1前34天内发生
   - 硬负比例控制在 **15-20%**

### 4.2 特征提取统一

**核心原则：三类样本使用完全相同的特征计算管道。**

推荐架构：
```python
class UnifiedFeaturePipeline:
    def extract_features(self, samples_df, sample_type):
        # 1. 统一获取原始数据（daily_data + daily_basic + stk_factor_pro）
        # 2. 统一调用 FeatureEngineer.compute_all_features()
        # 3. 统一添加市场环境特征
        # 4. 统一完整性校验
        # 5. 统一输出格式
```

**Tushare 因子使用策略**：
- `stk_factor_pro` 提供的 80+ 因子**全部使用**
- 仅对 Tushare 未覆盖的特征进行本地计算（自定义 breakout 识别、成交量交互等）
- 建立 `TUSHARE_FACTOR_MAP` 明确映射关系

### 4.3 特征完整性保障

```python
def validate_features(df):
    """训练前强制校验"""
    # 1. NaN 率必须为 0%
    # 2. Inf 值检查
    # 3. 三类样本特征列完全一致
    # 4. 日期范围校验（无未来数据）
    # 5. 样本ID唯一性校验
```

---

## 五、待执行动作清单

| # | 动作 | 优先级 | 状态 |
|---|------|--------|------|
| 1 | 编写 `UnifiedFeaturePipeline` 统一特征提取管道 | P0 | 待执行 |
| 2 | 扩展 `STK_FACTOR_RENAME` 包含全部 80+ Tushare 因子 | P0 | 待执行 |
| 3 | 重新生成正样本特征（使用新管道） | P0 | 待执行 |
| 4 | 重新生成负样本特征（使用新管道+特征匹配采样） | P0 | 待执行 |
| 5 | 重新生成硬负样本特征（使用新管道+收紧定义） | P0 | 待执行 |
| 6 | 实现 `validate_features()` 完整性校验 | P0 | 待执行 |
| 7 | 运行校验，确保 NaN 率为 0% | P0 | 待执行 |
| 8 | 训练 v3.0 模型 | P1 | 待执行 |
| 9 | 回测验证 | P1 | 待执行 |
