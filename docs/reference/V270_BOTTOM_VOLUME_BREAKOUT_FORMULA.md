# 基于 v270 的「底部放量突破」固定计算方式

## 一、结论先说

- **v270 没有单独一个“底部放量突破”的标签或输出**，选股是模型对 170+ 维特征打分后的综合结果。
- 但代码里对「突破」「放量」「共振+量能」等都有**固定、可复现的计算公式**，与 v270 训练/预测使用的特征一致（以 v5 对齐逻辑为准，见 `scripts/align_v6_to_v5_features.py`）。
- 若你要**用规则单独定义**“底部放量突破”一条信号，可以基于下面公式和阈值组合出一个**固定计算方式**。

---

## 二、与 v270 一致的固定公式（出处：v5 对齐逻辑）

以下按「前高 → 突破 → 量能 → 共振确认」顺序，全部为标量/逐日计算，按 `sample_id` 分组时在组内按 `trade_date` 排序后滚动计算。

### 1. 前 N 日高点（用于定义“突破”）

- **prev_high_{period}d**（period = 10, 20, 55）  
  - 公式：`prev_high = close.shift(1).rolling(period).max()`  
  - 含义：当日**之前** period 日内的**收盘价**最高值（不含当日）。  
  - 注意：这里用的是**收盘价**的前高，不是 `high` 的前高。

### 2. 突破强度（连续值，%）

- **breakout_strength_{period}d**（period = 10, 20, 55）  
  - 公式：  
    `breakout_strength = (close - prev_high) / (prev_high + 1e-8) * 100`  
  - 含义：收盘价相对前 period 日收盘前高的偏离百分比。  
  - \> 0：突破前高；= 0：刚好在前高；< 0：未突破。

### 3. 突破确认（二值，简化版）

- **breakout_confirmed_10d / breakout_confirmed_20d**  
  - 公式（与 align 一致）：  
    `breakout_confirmed = (breakout_strength > 0).astype(int)`  
  - 含义：当前是否站在对应周期前高之上（1=站稳，0=未站稳）。  
  - 注：更严的“3 日站稳”在 `enrich_breakout_features.py` 中有另一套实现（用 `low_3d_min > prev_high`），v5 训练数据用的是上述简化版。

### 4. 多周期突破共振（0～3 整数）

- **breakout_resonance**  
  - 公式：  
    `breakout_resonance = (breakout_strength_10d > 0).astype(int) + (breakout_strength_20d > 0).astype(int) + (breakout_strength_55d > 0).astype(int)`  
  - 含义：10/20/55 日三个周期里，有多少个周期“收盘价突破前高”。  
  - 取值 0、1、2、3。

### 5. 量能相关（放量）

- **vol_ma20**  
  - 公式：`vol_ma20 = vol.rolling(20, min_periods=5).mean()`

- **breakout_volume_strength**（v5 对齐版）  
  - 公式：`breakout_volume_strength = vol / (vol_ma20 + 1e-8)`  
  - 含义：当日成交量 / 20 日均量。**每天都算**，不限定“是否突破日”。  
  - 注：`enrich_breakout_features.py` 里有一版“仅突破日非 0”的定义：当 `close > prev_high_20d` 时为 `vol/vol_ma20`，否则 0；v270 训练数据采用的是“每天都算”的版本。

- **volume_ratio**（若存在）  
  - 常见定义：`volume_ratio = vol / vol_ma5`，即量比（5 日均量）。

### 6. 共振 + 量能确认（二值）

- **resonance_volume_confirm**  
  - 公式：  
    `resonance_volume_confirm = (breakout_resonance > 0).astype(int) * (breakout_volume_strength > 1.2).astype(int)`  
  - 含义：至少有一个周期突破 **且** 当日量能 > 20 日均量的 1.2 倍时为 1，否则 0。  
  - 即：**多周期突破 + 放量（>1.2 倍）** 的固定定义。

### 7. “放量突破”二值（价涨 + 放量）

- **breakout_with_volume**（v5 对齐版）  
  - 公式：  
    `breakout_with_volume = ((close > close.shift(1)) & (vol > vol_ma20 * 1.5)).astype(int)`  
  - 含义：**当日收涨** 且 **成交量 > 20 日均量的 1.5 倍** 时为 1，否则 0。  
  - 即：代码里“放量突破”的**规则化定义**是：上涨 + 量 > 1.5 倍 20 日均量。

### 8. “底部”相关（位置，用于规则化时筛选）

- **price_position_{period}d**（period = 8, 34, 55）  
  - 公式：在区间 [rolling_low, rolling_high] 内，  
    `price_position = (close - rolling_low) / (rolling_high - rolling_low + 1e-8) * 100`  
  - 含义：0～100，低值表示接近区间下沿（偏“底部”）。

- **price_vs_hist_high**  
  - 公式：`(close - rolling_55d_max) / (rolling_55d_max + 1e-8) * 100`  
  - 含义：相对 55 日最高价的偏离%，负值表示在 55 日高点下方（偏“底部”）。

- **price_vs_hist_mean**  
  - 公式：`(close - rolling_55d_mean) / (rolling_55d_mean + 1e-8) * 100`  
  - 含义：相对 55 日均价的偏离%，负值表示在均线下方。

---

## 三、用规则定义“底部放量突破”的固定计算方式

在**不改变 v270 特征计算**的前提下，若你要一条**可复现的、规则化的“底部放量突破”**信号，可以按下面方式组合上述公式（取的是与 v270 一致的实现和常用阈值）。

### 方式 A：严格版（底部 + 多周期突破 + 放量）

1. **底部**（满足其一即可，或同时满足更严）：  
   - `price_position_34d < 40` 或 `price_vs_hist_high < 0`  
2. **突破**：  
   - `breakout_resonance >= 1`（至少一个周期突破前高）  
3. **放量**：  
   - `breakout_volume_strength > 1.2`（与 `resonance_volume_confirm` 一致）或 `> 1.5`（与 `breakout_with_volume` 一致）  
4. **规则信号**：  
   - `bottom_volume_breakout = 底部条件 & (breakout_resonance >= 1) & (breakout_volume_strength > 1.2)`  
   - 或直接用已有特征：  
     `resonance_volume_confirm == 1` 且 再叠加 `price_position_34d < 40` 等“底部”条件。

### 方式 B：与代码完全一致的二值“放量突破”+ 底部过滤

1. **放量突破**（与 v5 对齐完全一致）：  
   - `breakout_with_volume = (close > close.shift(1)) & (vol > vol_ma20 * 1.5)`  
2. **底部**：  
   - `price_vs_hist_high < 0` 或 `price_position_34d < 40`  
3. **规则信号**：  
   - `bottom_volume_breakout = breakout_with_volume & 底部条件`

### 方式 C：连续值“强度”（用于排序或阈值）

- 用 **breakout_with_volume** 的连续版（若采用 compare_v260 那套）：  
  `breakout_strength_20d * breakout_volume_ratio`（突破强度 × 量比），再乘以“底部”权重（例如 `(1 - price_position_34d/100)`），得到连续得分，再设阈值。

---

## 四、实现时注意事项

1. **前高口径**：v270 训练数据里前高是 **close** 的滚动 max（`close.shift(1).rolling(period).max()`），不是 `high`；若你希望与回测/实盘一致，请统一用同一口径。  
2. **分组**：所有滚动量都要按 `sample_id`（或按股票+时间）分组后，在组内按日期排序再算，避免不同样本混在一起。  
3. **缺失值**：`prev_high` 不足 period 日时为 NaN，对应 `breakout_strength` 也为 NaN，规则里可视为“未突破”或排除该日。  
4. **v270 选股**：模型是把这些特征一起喂给 XGBoost/集成模型，**没有**显式的一步“先算 bottom_volume_breakout 再选股”；若你要规则与模型并存，建议规则只作为过滤或解释用，特征计算仍与上面保持一致。

---

## 五、公式汇总表（复制即用）

| 名称 | 公式 |
|------|------|
| prev_high_{10,20,55}d | `close.shift(1).rolling(period).max()` |
| breakout_strength_{10,20,55}d | `(close - prev_high) / (prev_high + 1e-8) * 100` |
| breakout_resonance | `(bs_10>0)+(bs_20>0)+(bs_55>0)`，取值 0～3 |
| vol_ma20 | `vol.rolling(20, min_periods=5).mean()` |
| breakout_volume_strength | `vol / (vol_ma20 + 1e-8)` |
| resonance_volume_confirm | `(breakout_resonance > 0) & (breakout_volume_strength > 1.2)` → 0/1 |
| breakout_with_volume | `(close > close.shift(1)) & (vol > vol_ma20 * 1.5)` → 0/1 |
| 规则化“底部放量突破” | 例如：`resonance_volume_confirm==1` 且 `price_position_34d<40`；或 `breakout_with_volume==1` 且 `price_vs_hist_high<0` |

以上即为基于 v270 选股逻辑的、**固定且可复现**的“底部放量突破”计算方式；规则化定义可按方式 A/B/C 任选或组合使用。
