# 新模型（四分类）训练计划

基于「底部放量突破严格版规则筛股 + T+5/T+15 四类重标」的结论，本计划明确样本构造、标签定义、模型形态与训练流程。

---

## 一、已确认结论（约束条件）

| 项 | 结论 |
|----|------|
| **样本粒度** | 每个 (股票, T0) 一条样本；T0 = 满足条件的那一天。 |
| **样本唯一性** | 每个 (ts_code, T0) 仅保留一条样本；同一只股在不同 T0 可产生多条样本。 |
| **模型形态** | 多分类：4 类（硬正、普正、普负、硬负）。 |
| **T+5/T+15 缺失** | 若 T0 后 5 或 15 个交易日内停牌、退市或数据缺失，则**延后取最近可用交易日**的收盘价计算涨幅。 |

---

## 二、整体流程概览

```
全市场日频行情 + 基础特征
        ↓
[T0 筛选] resonance_volume_confirm==1 且 price_position_34d<40
        ↓
候选 (ts_code, T0) 列表，每个 (ts_code, T0) 一条
        ↓
[T+5/T+15 收益] 延后取最近可用日，计算 return_T5, return_T15
        ↓
[四类打标] 硬正 / 普正 / 普负 / 硬负
        ↓
[T0 特征] 取 T0 日（或 T0 前 N 日）特征，与 v270 特征体系对齐
        ↓
按 trade_date 时间序列划分：训练 / 验证 / 测试
        ↓
多分类模型训练（XGBoost/LightGBM multi:softmax）
        ↓
评估：准确率、宏 F1、混淆矩阵、各类别 P/R
```

---

## 三、数据与样本构造

### 3.1 T0 候选筛选规则（严格版）

- **条件**（同时满足）：
  - `resonance_volume_confirm == 1`
  - `price_position_34d < 40`
- **`resonance_volume_confirm` 计算公式**（逐日、按 ts_code 分组后滚动）：
  1. 前高：`prev_high_{10,20,55}d = close.shift(1).rolling(period).max()`（用收盘价，不用 high）
  2. 突破强度：`breakout_strength_{10,20,55}d = (close - prev_high) / (prev_high + 1e-8) * 100`
  3. 突破共振：`breakout_resonance = (breakout_strength_10d>0).astype(int) + (breakout_strength_20d>0).astype(int) + (breakout_strength_55d>0).astype(int)`，取值 0～3
  4. 20 日均量：`vol_ma20 = vol.rolling(20, min_periods=5).mean()`
  5. 量能强度：`breakout_volume_strength = vol / (vol_ma20 + 1e-8)`（每日都算）
  6. **resonance_volume_confirm** = `(breakout_resonance > 0).astype(int) * (breakout_volume_strength > 1.2).astype(int)` → **0/1**
  7. 含义：至少一个周期突破前高 **且** 当日量能 > 20 日均量的 1.2 倍 → 为 1
- **`price_position_34d` 计算公式**（逐日、按 ts_code 分组后滚动）：
  1. `rolling_high_34 = high.rolling(34, min_periods=17).max()`
  2. `rolling_low_34 = low.rolling(34, min_periods=17).min()`
  3. `price_position_34d = (close - rolling_low_34) / (rolling_high_34 - rolling_low_34 + 1e-8) * 100`
  4. 取值 0～100，**< 40 表示价格在 34 日区间偏下沿（底部）**
- **数据要求**：按 (ts_code, trade_date) 逐日计算上述两个指标，满足 `resonance_volume_confirm == 1` 且 `price_position_34d < 40` 的 (ts_code, trade_date) 即为候选 T0。

### 3.2 样本唯一性

- **一条样本** = 一个 (ts_code, T0)。
- 同一 ts_code 在不同 T0 可多次出现（例如连续多日满足条件 → 多条样本）。
- 不做「同一股只取首次 T0」的约束；若后续需要可在此步骤后增加去重策略（如同一股 20 日内只保留第一次 T0）。

### 3.3 T+5 / T+15 收益（延后取最近可用日）

- **return_T5**：以 T0 为基准，目标为 T0 后第 5 个**交易日**的收盘价；若该日停牌/退市/缺失，则顺延至其后**最近一个有效交易日**的收盘价。
- **return_T15**：同理，目标为 T0 后第 15 个交易日，缺失则延后取最近可用日。
- **公式**：
  - `close_T5_actual = 取 T0 后第 5 个交易日（或顺延）的 close`
  - `close_T15_actual = 取 T0 后第 15 个交易日（或顺延）的 close`
  - `return_T5 = (close_T5_actual - close_T0) / close_T0 * 100`
  - `return_T15 = (close_T15_actual - close_T0) / close_T0 * 100`
- **顺延上限**：建议约定顺延最多不超过 N 个交易日（如 5 日），超出则将该样本标记为「收益缺失」并从训练集中剔除或单独评估。

---

## 四、四类标签定义（实现用简化版）

按**优先级**依次判断，保证互斥且全覆盖：

| 优先级 | 标签 | 条件 | 说明 |
|--------|------|------|------|
| 1 | **硬正** | T+5 涨幅 ≥ 20% **或** T+15 涨幅 ≥ 50% | 短期大涨或中期暴涨 |
| 2 | **硬负** | T+15 涨幅 < 0% **且** 0% ≤ T+5 涨幅 < 20% | 中期跌、短期未大涨 |
| 3 | **普负** | T+5 涨幅 < 5% **且** T+15 涨幅 < 10% | 短期/中期都偏弱（含 T+15<0） |
| 4 | **普正** | 其余 | 未达硬正/硬负/普负，即有一定正收益但未达硬正 |

**类别编码建议**（与 sklearn/XGB 一致）：
`硬正=0, 普正=1, 普负=2, 硬负=3`，或按业务顺序自定，训练与评估时保持一致即可。

---

## 五、特征体系

- **特征时点**：T0 当日的截面特征。对每个 (ts_code, T0)，取该股截至 T0 日（含）的日频行情（close/high/low/vol/pct_chg/turnover_rate 等），按附录 B 的公式逐日滚动计算所有特征，最终只取 **T0 当日那一行** 作为该样本的特征向量。
- **特征范围**：复用 v270 的数值特征集（见下文**附录 A**完整列表和**附录 B**计算公式），去掉以下列：
  - **排除的低效二值特征（6 个）**：`breakout_high_10d`, `breakout_high_20d`, `breakout_ma10`, `breakout_ma55`, `high_volume_breakout`, `volume_price_match`
  - **排除的元数据列**：label, sample_id, ts_code, name, trade_date, weekly_return_1/2/3, total_return_34d, weekly_volume_1/2/3, days_to_t1
  - **排除的标签泄漏列**：return_T5, return_T15（若存在）
- **缺失值**：统一填 0（与 v270 训练脚本 `fillna(0)` 一致）。
- **特征总数**：约 170+ 维（含 14 个增强特征），以附录 A 为准。

---

## 附录 A：v270 特征列表（按类别）

v270 使用 v5 共同数值特征（排除 6 个低效二值特征）+ 14 个增强特征，总约 170+ 维。

**训练时排除的列**：`breakout_high_10d`, `breakout_high_20d`, `breakout_ma10`, `breakout_ma55`, `high_volume_breakout`, `volume_price_match`；以及元数据 label、sample_id、ts_code、name、trade_date、weekly_return_1/2/3、total_return_34d、weekly_volume_1/2/3、days_to_t1。

| 类别 | 特征名（示例） |
|------|----------------|
| 价格与均线 | close, open, high, low, change, pct_chg, ma5, ma10, ma_5d, ma_8d, ma_10d, ma_20d, ma_34d, ma_55d, ema_5, ema_10, ema_20, ema_60, price_vs_ma_34d, price_vs_ma_55d, price_vs_ma_8d, bias_short, bias_mid, bias_long, high_8d, high_34d, high_55d, low_8d, low_34d, low_55d |
| 位置与趋势 | price_position_8d, price_position_34d, price_position_55d, price_vs_hist_mean, price_vs_hist_high, volatility_vs_hist, trend_slope_8d, trend_slope_34d, trend_slope_55d, trend_consistency |
| 动量与收益 | momentum_5d, momentum_10d, momentum_20d, return_8d, return_34d, return_55d, volatility_8d, volatility_34d, volatility_55d |
| 成交量与换手 | vol, volume_ratio, vol_ma5_ratio, vol_ma20_ratio, volume_breakout_count_20d, volume_change, volume_shrink_ratio, volume_trend_slope_10d, volume_trend_slope_20d, volume_rsv_20d, volume_price_corr_10d, volume_price_corr_20d, price_down_vol_up, price_up_vol_down, price_down_vol_up_count_10d, price_up_vol_down_count_10d, volume_price_match_sum_10d, turnover_rate, turnover_rate_f, obv, obv_calc, obv_ma10, obv_trend |
| 突破相关 | prev_high_10d, prev_high_20d, prev_high_55d, breakout_strength_10d, breakout_strength_20d, breakout_strength_55d, breakout_confirmed_10d, breakout_confirmed_20d, breakout_resonance, breakout_volume_strength, breakout_volume_ratio, breakout_with_volume, resonance_volume_confirm, breakout_rsi_interaction, breakout_high_55d |
| 支撑阻力 | support_10d, support_20d, support_55d, resistance_10d, resistance_20d, resistance_55d, dist_to_support_10d/20d/55d, dist_to_resistance_10d/20d/55d, support_strength_10d/20d/55d, resistance_strength_10d/20d/55d |
| 技术指标 | rsi_6, rsi_12, rsi_24, macd_dif, macd_dea, macd, kdj_k, kdj_d, kdj_j, rsi_kdj_divergence |
| 风险与波动 | max_drawdown_10d, max_drawdown_20d, max_drawdown_55d, atr_14, atr_ratio_14, atr_expansion, days_from_high_20d, days_from_high_55d, recovery_ratio_20d, price_range_pct, channel_width_20d, relative_volatility |
| 其他 | close_vs_ma10_std, days_near_ma10, ma10_cross_count, consecutive_new_high, is_limit_up, total_mv, circ_mv, amount, momentum_market_interaction, market_*（若存在） |
| 增强特征（14 个） | turnover_zscore, turnover_change_rate, turnover_spike, rsi_kdj_golden_cross, rsi_kdj_strength, rsi_zone, volume_price_divergence_strength, volume_price_confirm, breakout_strength_avg, breakout_strength_max, ma_alignment_score, momentum_acceleration, price_position_avg, sharpe_like_34d |

---

## 附录 B：v270 特征计算公式（与 v5 对齐一致）

以下公式按「前高 → 突破 → 量能 → 位置 → 其他」顺序列出，**按 (ts_code 或 sample_id) 分组、组内按 trade_date 排序后**在组内滚动计算。`close`/`high`/`low`/`vol`/`pct_chg` 为日频序列。

### B.1 前高与突破（T0 筛选与核心特征）

| 特征 | 计算公式 |
|------|----------|
| prev_high_{10,20,55}d | `prev_high = close.shift(1).rolling(period).max()` |
| breakout_strength_{10,20,55}d | `(close - prev_high) / (prev_high + 1e-8) * 100` |
| breakout_confirmed_10d/20d | `(breakout_strength_*d > 0).astype(int)` |
| breakout_resonance | `(breakout_strength_10d>0).astype(int) + (breakout_strength_20d>0).astype(int) + (breakout_strength_55d>0).astype(int)`，取值 0～3 |
| vol_ma20 | `vol.rolling(20, min_periods=5).mean()` |
| breakout_volume_strength | `vol / (vol_ma20 + 1e-8)`（每日都算） |
| **resonance_volume_confirm** | `(breakout_resonance > 0).astype(int) * (breakout_volume_strength > 1.2).astype(int)` → 0/1 |
| breakout_with_volume | `((close > close.shift(1)) & (vol > vol_ma20 * 1.5)).astype(int)` → 0/1 |

### B.2 底部与位置（含 T0 筛选用）

| 特征 | 计算公式 |
|------|----------|
| **price_position_{8,34,55}d** | `rolling_high = high.rolling(period, min_periods=period//2).max()`；`rolling_low = low.rolling(period, min_periods=period//2).min()`；`(close - rolling_low) / (rolling_high - rolling_low + 1e-8) * 100`（0～100，低≈底部） |
| price_vs_hist_mean | `(close - close.rolling(55).mean()) / (close.rolling(55).mean() + 1e-8) * 100` |
| price_vs_hist_high | `(close - close.rolling(55).max()) / (close.rolling(55).max() + 1e-8) * 100` |
| volatility_vs_hist | `pct_chg.rolling(10).std() / (pct_chg.rolling(55).std() + 1e-8)` |

### B.3 量能与量价

| 特征 | 计算公式 |
|------|----------|
| volume_ratio | `vol / (vol.rolling(5).mean() + 1e-8)` |
| vol_ma5_ratio, vol_ma20_ratio | `vol / (vol.rolling(5).mean() + 1e-8)`，`vol / (vol.rolling(20).mean() + 1e-8)` |
| volume_breakout_count_20d | `(vol > vol.rolling(20).mean() * 2).astype(int).rolling(20).sum()` |
| volume_shrink_ratio | `vol.rolling(5).mean() / (vol.rolling(20).mean() + 1e-8)` |
| volume_trend_slope_10d/20d | `vol.diff(10) / (vol.shift(10) + 1e-8)`，`vol.diff(20) / (vol.shift(20) + 1e-8)` |
| volume_rsv_20d | `(vol - vol.rolling(20).min()) / (vol.rolling(20).max() - vol.rolling(20).min() + 1e-8)`，缺省 0.5 |
| volume_price_corr_10d/20d | `close.rolling(period).corr(vol)` |
| volume_price_match_sum_10d | 先算 `volume_price_match = (vol>vol.shift(1)) == (pct_chg>0)`，再 `rolling(10).sum()` |

### B.4 均线、动量、波动率

| 特征 | 计算公式 |
|------|----------|
| ma_5d, ma_10d, ma_20d, ma_34d, ma_55d | `close.rolling(period, min_periods=period//2).mean()`；ma5=ma_5d, ma10=ma_10d |
| ema_5, ema_10, ema_20, ema_60 | `close.ewm(span=period, adjust=False).mean()` |
| bias_short, bias_mid, bias_long | 对应 period=5,10,20：`(close - close.rolling(period).mean()) / (close.rolling(period).mean() + 1e-8) * 100` |
| momentum_5d, momentum_10d, momentum_20d | `close.pct_change(period) * 100` |
| return_8d, return_34d, return_55d | `(close - close.shift(period)) / (close.shift(period) + 1e-8) * 100` |
| volatility_8d, volatility_34d, volatility_55d | `pct_chg.rolling(period, min_periods=period//2).std()` |
| trend_slope_8d, trend_slope_34d, trend_slope_55d | 对 close 过去 period 日做线性回归斜率，再除以当日 close 归一化（或 `close.diff(period)/close.shift(period)*100` 等实现，与项目内一致即可） |

### B.5 支撑阻力、风险

| 特征 | 计算公式 |
|------|----------|
| support_{10,20,55}d, resistance_{10,20,55}d | `close.rolling(period).min()`，`close.rolling(period).max()` |
| dist_to_support_{10,20,55}d | `(close - support) / (close + 1e-8) * 100` |
| dist_to_resistance_{10,20,55}d | `(resistance - close) / (close + 1e-8) * 100` |
| support_strength_55d, resistance_strength_55d | `(close - support_55d) / (support_55d + 1e-8) * 100`，`(resistance_55d - close) / (resistance_55d + 1e-8) * 100` |
| max_drawdown_{10,20,55}d | `(close - close.rolling(period).max()) / (close.rolling(period).max() + 1e-8) * 100`，再对结果做 `rolling(period).min()` |
| atr_14 | TR = max(high-low, abs(high-close.shift(1)), abs(low-close.shift(1)))；`atr_14 = TR.rolling(14).mean()` |
| atr_ratio_14 | `atr_14 / (close + 1e-8) * 100` |
| recovery_ratio_20d | `(close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min() + 1e-8)` |
| price_range_pct | `(high - low) / (close + 1e-8) * 100` |
| channel_width_20d | `(close.rolling(20).max() - close.rolling(20).min()) / (close + 1e-8) * 100` |

### B.6 RSI、MACD、KDJ（若未从行情接口直接取）

| 特征 | 计算公式 |
|------|----------|
| rsi_6, rsi_12, rsi_24 | delta=close.diff()；gain=delta.where(delta>0,0)；loss=(-delta).where(delta<0,0)；avg_gain/gain_loss=rolling(period).mean()；RS=avg_gain/(avg_loss+1e-8)；`RSI = 100 - 100/(1+RS)` |
| macd_dif | `close.ewm(span=12).mean() - close.ewm(span=26).mean()` |
| macd_dea | `macd_dif.ewm(span=9).mean()` |
| macd | `(macd_dif - macd_dea) * 2` |
| kdj_k, kdj_d, kdj_j | RSV = (close - low.rolling(9).min()) / (high.rolling(9).max() - low.rolling(9).min() + 1e-8) * 100；K = RSV.ewm(com=2).mean()；D = K.ewm(com=2).mean()；J = 3*K - 2*D |

### B.7 增强特征（14 个）

| 特征 | 计算公式 |
|------|----------|
| turnover_zscore | `(turnover_rate - turnover_rate.rolling(20).mean()) / (turnover_rate.rolling(20).std() + 1e-8)` |
| turnover_change_rate | `turnover_rate.pct_change(5)` |
| turnover_spike | `(turnover_rate > turnover_rate.rolling(20).mean() * 2).astype(int)` |
| rsi_kdj_golden_cross | `(rsi_6 > 50) & (kdj_j > kdj_k)` → 0/1 |
| rsi_kdj_strength | `(rsi_6/100 + kdj_j/100) / 2` |
| rsi_zone | RSI>70→1, RSI<30→-1, 否则 0 |
| volume_price_divergence_strength | `abs(close.pct_change(10) - vol.pct_change(10))` |
| volume_price_confirm | `(close.pct_change(10)>0) == (vol.pct_change(10)>0)` → 0/1 |
| breakout_strength_avg | `breakout_strength_10d/20d/55d 的 mean(axis=1)` |
| breakout_strength_max | `breakout_strength_10d/20d/55d 的 max(axis=1)` |
| ma_alignment_score | 对 ma5, ma10, ma_20d, ma_34d, ma_55d 按从大到小排序，计算排序一致性：`sorted_idx = argsort(row)[::-1]`；`expected = arange(len)`；`score = 1 - abs(sorted_idx - expected).sum() / (n*(n-1)/2 + 1e-8)`；完全多头排列（短期>长期）得分接近 1，完全空头接近 0 |
| momentum_acceleration | `momentum_10d.diff(5)` |
| price_position_avg | 各 `price_position_*d` 的 mean |
| sharpe_like_34d | `return_34d / (volatility_34d + 1e-8)` |

**T0 严格版筛选所用两特征**（必须与上面公式一致）：

- **`resonance_volume_confirm == 1`**（B.1）：
  `(breakout_resonance > 0).astype(int) * (breakout_volume_strength > 1.2).astype(int)`
  其中 breakout_resonance = 10/20/55 日三个周期突破前高的计数（0～3），breakout_volume_strength = vol / vol_ma20。
  含义：至少一个周期突破前高 **且** 当日量能 > 20 日均量 1.2 倍。
- **`price_position_34d < 40`**（B.2）：
  `(close - low.rolling(34, min_periods=17).min()) / (high.rolling(34, min_periods=17).max() - low.rolling(34, min_periods=17).min() + 1e-8) * 100`
  取值 0～100，< 40 表示价格在 34 日区间偏下沿（底部）。

---

## 六、模型与训练

### 6.1 模型形态

- **任务**：4 分类（硬正 / 普正 / 普负 / 硬负）。
- **算法**：XGBoost 或 LightGBM 的 **multi-class**（如 `objective='multi:softmax'` 或 `'multiclass'`），输出 4 个类别的概率或类别标签。

### 6.2 划分方式

- **按时间序列划分**：按样本的 T0（trade_date）排序，按比例切分，例如：
  - 训练集：最早 65% 的 T0 区间内样本
  - 验证集：接下来 15%
  - 测试集：最近 20%
- 确保**同一 (ts_code, T0) 只出现在一个集合**，且验证/测试的 T0 晚于训练集，避免前视。

### 6.3 类别不平衡

- 若四类样本量差异大，可采用：
  - **class_weight**（如 `scale_pos_weight` 或 `class_weight='balanced'`）；
  - 或对少数类过采样 / 多数类欠采样（在训练集内做，验证/测试保持原分布）。
- 建议先统计四类占比，再决定是否上采样/权重。

### 6.4 超参数与早停

- 以 v270 超参数作为初值，将 objective 改为多分类：

| 参数 | v270 值（作为初值） | 新模型调整说明 |
|------|---------------------|----------------|
| objective | `binary:logistic` | → `multi:softmax`（或 `multi:softprob`），`num_class=4` |
| max_depth | 6 | 保持 |
| learning_rate | 0.1 | 保持 |
| subsample | 0.9 | 保持 |
| colsample_bytree | 0.8 | 保持 |
| min_child_weight | 5 | 保持 |
| gamma | 0.1 | 保持 |
| reg_alpha | 0.1 | 保持 |
| reg_lambda | 0.5 | 保持 |
| scale_pos_weight | 1.5 | → 移除（多分类不适用），改用 `sample_weight` 按类别加权 |
| eval_metric | `auc`, `aucpr` | → `mlogloss` |
| tree_method | `hist` | 保持 |
| random_state | 42 | 保持 |
| num_boost_round | 1000（v270 默认） | 保持，配合早停 |

- 使用验证集早停（early_stopping_rounds=50），以 `mlogloss`（或 macro F1）为监控指标。

---

## 七、评估指标

- **整体**：准确率（Accuracy）、宏平均 F1（macro F1）、多分类 logloss。
- **每类**：精确率（Precision）、召回率（Recall）、F1；支持数（support）。
- **混淆矩阵**：4×4，行为真实类别，列为预测类别。
- **业务向**：可将「硬正+普正」视为“正”，「普负+硬负」视为“负”，算二分类 P/R/F1，便于与 v270 二分类对比。

---

## 八、实现步骤（建议顺序）

| 步骤 | 内容 | 产出/脚本 |
|------|------|-----------|
| 1 | 全市场日频计算 resonance_volume_confirm、price_position_34d（按 v270 公式） | 特征表或特征管道 |
| 2 | 筛选 T0：满足严格版条件的 (ts_code, trade_date)，去重得到 (ts_code, T0) 列表 | 候选表 |
| 3 | 对每个 (ts_code, T0) 计算 return_T5、return_T15（延后取最近可用日），并打四类标签 | 带 label 的样本表 |
| 4 | 对每个 (ts_code, T0) 提取 T0 日截面特征（v270 特征集） | 特征矩阵 + 标签 |
| 5 | 按 T0 时间序列划分 train/val/test | 三个子集 |
| 6 | 训练多分类模型（XGB/LGB），验证集早停 | 模型文件 + 训练日志 |
| 7 | 在测试集上计算准确率、macro F1、混淆矩阵、各类 P/R | 评估报告 |
| 8 | （可选）与 v270 二分类在同一区间上对比「规则筛 + 四类」的收益分布 | 对比分析 |

---

## 九、脚本与配置建议

- **数据与打标**：新建脚本，如 `scripts/prepare_4class_samples.py`，完成：候选筛选 → T+5/T+15 收益（延后取最近可用日）→ 四类打标 → 输出 `data/training/4class/xxx.csv`。
- **特征**：按附录 B 的公式计算全部特征，取 T0 日截面；可封装为独立函数供复用。
- **训练**：新建 `scripts/train_4class_model.py`，读入上述样本 + 特征，时间序列划分，多分类训练与评估，保存模型到如 `data/models/breakout_4class/versions/v1.0/`。
- **配置**：可将「严格版阈值」（如 price_position_34d 的 40）、顺延上限、划分比例、类别编码等写入配置文件（如 `config/4class_training.yaml`），便于复现与调参。

---

## 十、风险与后续可调

- **候选量**：严格版可能使 T0 候选偏少，若样本不足可考虑放宽 price_position_34d（如 < 50）或先做时间跨度更大的历史扫描。
- **顺延**：延后取最近可用日会令部分样本的“T+5/T+15”实际对应更晚的交易日，可在表中保留 actual_t5_date、actual_t15_date 便于分析。
- **版本**：本计划对应「新模型 v1.0（四分类）」，与现有 v270 二分类并存，不替换现有训练管线。

以上为新模型（四分类）的完整训练计划，可按步骤落地实现与迭代。
