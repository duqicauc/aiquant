# v270 模型特征说明与「底部放量突破」相关特征

## 一、v270 特征来源

- **数据**：`feature_data_34d_v5_enhanced.csv`（正样本）、`negative_feature_data_v2_34d_v5_enhanced.csv`、`hard_negative_feature_data_34d_v5_enhanced.csv`
- **特征**：正/负/硬负三个文件的**共同列**中，去掉元数据与 6 个低效二值特征后，只保留数值型列；再加上 14 个增强特征。
- **训练时排除的列**：`breakout_high_10d`, `breakout_high_20d`, `breakout_ma10`, `breakout_ma55`, `high_volume_breakout`, `volume_price_match`（以及 label、sample_id、ts_code、name、trade_date、weekly_return_1/2/3、total_return_34d、weekly_volume_1/2/3、days_to_t1）。

---

## 二、v270 使用的全部特征（按类别）

### 1. 价格与均线

| 特征 | 说明 |
|------|------|
| close, open, high, low | 价格 |
| change, price_change, pct_chg | 涨跌 |
| ma5, ma10, ma_5d, ma_8d, ma_10d, ma_20d, ma_34d, ma_55d | 均线 |
| ema_5, ema_10, ema_20, ema_60 | EMA |
| price_vs_ma_34d, price_vs_ma_55d, price_vs_ma_8d | 价格相对均线 |
| bias_short, bias_mid, bias_long | 乖离率 |
| high_8d, high_34d, high_55d, low_8d, low_34d, low_55d | 区间高低点 |

### 2. 位置与趋势（含“底部”信息）

| 特征 | 说明 |
|------|------|
| price_position_8d, price_position_34d, price_position_55d | 价格在区间内位置（低≈底部） |
| price_vs_hist_mean, price_vs_hist_high | 相对历史均值/高点 |
| volatility_vs_hist | 波动率相对历史 |
| trend_slope_8d, trend_slope_34d, trend_slope_55d | 趋势斜率 |
| trend_consistency | 趋势一致性 |

### 3. 动量与收益

| 特征 | 说明 |
|------|------|
| momentum_5d, momentum_10d, momentum_20d | 动量 |
| return_8d, return_34d, return_55d | 区间收益 |
| volatility_8d, volatility_34d, volatility_55d | 波动率 |
| momentum_acceleration | 动量加速度（增强） |

### 4. 成交量与换手（“放量”相关）

| 特征 | 说明 |
|------|------|
| vol | 成交量 |
| volume_ratio | 量比（当日量/5日均量） |
| vol_ma5_ratio, vol_ma20_ratio | 量比类 |
| volume_breakout_count_20d | 20 日内放量次数 |
| volume_change, volume_shrink_ratio | 量能变化、缩量比 |
| volume_trend_slope_10d, volume_trend_slope_20d | 量能趋势斜率 |
| volume_rsv_20d | 量能相对位置 |
| volume_price_corr_10d, volume_price_corr_20d | 量价相关 |
| price_down_vol_up, price_up_vol_down | 价跌量增/价涨量缩 |
| price_down_vol_up_count_10d, price_up_vol_down_count_10d | 对应天数 |
| volume_price_match_sum_10d | 量价匹配汇总 |
| turnover_rate, turnover_rate_f | 换手率 |
| obv, obv_calc, obv_ma10, obv_trend | OBV 相关 |

### 5. 突破相关（“突破”核心）

| 特征 | 说明 |
|------|------|
| prev_high_10d, prev_high_20d, prev_high_55d | 前 N 日高点 |
| breakout_strength_10d, breakout_strength_20d, breakout_strength_55d | 突破强度（相对前高） |
| breakout_confirmed_10d, breakout_confirmed_20d | 突破确认（如 3 日站稳） |
| breakout_resonance | 多周期突破共振 |
| breakout_volume_strength | 突破时成交量/20 日均量 |
| breakout_volume_ratio | 突破时量比 |
| breakout_with_volume | 突破且放量（强度×量比） |
| resonance_volume_confirm | 共振 + 量能确认 |
| breakout_rsi_interaction | 突破与 RSI 交互 |
| breakout_high_55d | 是否突破 55 日高 |

### 6. 支撑阻力

| 特征 | 说明 |
|------|------|
| support_10d, support_20d, support_55d | 支撑位 |
| resistance_10d, resistance_20d, resistance_55d | 阻力位 |
| dist_to_support_10d/20d/55d | 距支撑距离 |
| dist_to_resistance_10d/20d/55d | 距阻力距离 |
| support_strength_10d/20d/55d | 支撑强度 |
| resistance_strength_10d/20d/55d | 阻力强度 |
| support_strength_55d, resistance_strength_55d | 55 日支撑/阻力强度 |

### 7. 技术指标

| 特征 | 说明 |
|------|------|
| rsi_6, rsi_12, rsi_24 | RSI |
| macd_dif, macd_dea, macd | MACD |
| kdj_k, kdj_d, kdj_j | KDJ |
| rsi_kdj_divergence | RSI-KDJ 背离 |

### 8. 风险与波动

| 特征 | 说明 |
|------|------|
| max_drawdown_10d, max_drawdown_20d, max_drawdown_55d | 最大回撤 |
| atr_14, atr_ratio_14, atr_expansion | ATR |
| days_from_high_20d, days_from_high_55d | 距高点天数 |
| recovery_ratio_20d | 恢复比例 |
| price_range_pct, channel_width_20d | 振幅、通道宽度 |
| relative_volatility | 相对波动率 |

### 9. 其他

| 特征 | 说明 |
|------|------|
| close_vs_ma10_std, days_near_ma10, ma10_cross_count | 与 MA10 关系 |
| consecutive_new_high | 连续创新高 |
| is_limit_up | 是否涨停 |
| total_mv, circ_mv | 市值 |
| amount | 成交额 |
| momentum_market_interaction | 动量与市场交互 |
| market_* | 市场环境（若存在） |

### 10. 增强特征（14 个）

| 特征 | 说明 |
|------|------|
| turnover_zscore | 换手率 Z 分数 |
| turnover_change_rate | 换手率变化率 |
| turnover_spike | 换手率是否突增 |
| rsi_kdj_golden_cross | RSI-KDJ 金叉 |
| rsi_kdj_strength | RSI-KDJ 综合强度 |
| rsi_zone | RSI 超买超卖区间 |
| volume_price_divergence_strength | 量价背离强度 |
| volume_price_confirm | 量价同向确认 |
| breakout_strength_avg | 突破强度平均 |
| breakout_strength_max | 突破强度最大 |
| ma_alignment_score | 均线多头排列得分 |
| momentum_acceleration | 动量加速度 |
| price_position_avg | 价格位置平均 |
| sharpe_like_34d | 34 日类夏普比 |

---

## 三、与「底部放量突破」直接相关的特征

**底部放量突破**：价格在相对低位 + 成交量明显放大 + 突破前高/关键均线或阻力。

### 1. 识别“放量”

| 特征 | 含义 |
|------|------|
| **volume_ratio** | 量比，放量核心 |
| **vol_ma5_ratio**, **vol_ma20_ratio** | 相对 5/20 日均量 |
| **breakout_volume_strength** | 突破日成交量/20 日均量（仅突破时非 0） |
| **breakout_volume_ratio** | 突破时量比 |
| **volume_breakout_count_20d** | 20 日内放量次数 |
| **resonance_volume_confirm** | 多周期突破共振 + 量能确认（如 >1.2 倍） |
| **breakout_with_volume** | 突破强度 × 量比，直接刻画“突破且放量” |
| turnover_rate, turnover_zscore, turnover_spike | 换手与异常放量 |
| volume_price_confirm（增强） | 量价同向，放量上涨 |

### 2. 识别“突破”

| 特征 | 含义 |
|------|------|
| **breakout_strength_10d/20d/55d** | 相对前高的突破强度（连续值） |
| **breakout_confirmed_10d/20d** | 突破是否被确认（如 3 日站稳） |
| **breakout_resonance** | 10/20/55 日多周期突破共振 |
| **breakout_strength_avg**, **breakout_strength_max**（增强） | 突破强度综合 |
| breakout_high_55d | 是否突破 55 日高 |
| prev_high_10d/20d/55d | 前高，用于计算突破强度 |

### 3. 识别“底部”（相对低位）

| 特征 | 含义 |
|------|------|
| **price_position_8d/34d/55d** | 在区间内位置，低值≈底部区域 |
| **price_vs_hist_mean**, **price_vs_hist_high** | 相对历史均值/高点，负或偏低≈底部 |
| **dist_to_support_10d/20d/55d** | 距支撑距离，近支撑≈底部 |
| **support_strength_10d/20d/55d** | 支撑强度，底部常伴强支撑 |
| volatility_vs_hist | 波动率相对历史，底部有时波动收敛或放大 |

### 4. 量价与突破的交叉

| 特征 | 含义 |
|------|------|
| **breakout_with_volume** | 突破 × 量能，直接对应“放量突破” |
| **resonance_volume_confirm** | 共振突破 + 量能确认 |
| **breakout_rsi_interaction** | 突破与 RSI 的交互（底部反弹常伴随 RSI 从低位回升） |
| volume_price_divergence, volume_price_divergence_strength | 量价背离，可辅助判断底部反转 |

---

## 四、小结

- v270 实际使用的特征 = v5 共同数值特征（去掉上述 6 个排除列）+ 14 个增强特征，具体以训练时 `feature_names` 为准（约一百多个）。
- **与“底部放量突破”直接相关的特征**可归纳为：
  - **放量**：volume_ratio, vol_ma5_ratio, vol_ma20_ratio, breakout_volume_strength, breakout_volume_ratio, volume_breakout_count_20d, resonance_volume_confirm, breakout_with_volume, turnover_*, volume_price_confirm。
  - **突破**：breakout_strength_10d/20d/55d, breakout_confirmed_10d/20d, breakout_resonance, breakout_strength_avg/max, breakout_high_55d。
  - **底部/位置**：price_position_*d, price_vs_hist_mean/high, dist_to_support_*d, support_strength_*d。
- 其中 **breakout_volume_strength**、**breakout_with_volume**、**resonance_volume_confirm** 是同时刻画“突破”与“放量”的联合特征，对识别底部放量突破最直接。
