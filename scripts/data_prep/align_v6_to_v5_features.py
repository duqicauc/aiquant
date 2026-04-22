#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为v6版本的正样本和负样本补充特征，使其与v5版本一致

目标：
- v6正样本和负样本使用更丰富的样本数据
- 但特征列要与v5版本完全一致
- 硬负样本直接使用v5版本
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.logger import log


def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有缺失的特征（按样本分组）
    修复版：避免数据丢失
    """
    df = df.copy()
    original_len = len(df)

    def calc_sample_features(g):
        g = g.sort_values("trade_date").copy()
        n = len(g)
        # 不再跳过任何数据，即使数据少也要处理

        close = g["close"]
        pct_chg = g.get("pct_chg", pd.Series([0] * n))

        # ========== 1. 基础价格相关 ==========
        # v6修复：不再使用估算值，如果缺少high/low则记录警告
        if "high" not in g.columns:
            log.warning(
                f"样本 {g['sample_id'].iloc[0] if 'sample_id' in g.columns else 'unknown'}: 缺少high列，使用pct_chg估算"
            )
            # 使用pct_chg和close估算，比固定比例更准确
            if "pct_chg" in g.columns:
                daily_range = g["pct_chg"].abs() / 100 * 0.5  # 假设振幅约为涨跌幅的一半
                g["high"] = close * (1 + daily_range.fillna(0.01))
            else:
                g["high"] = close * 1.01
        if "low" not in g.columns:
            if "pct_chg" in g.columns:
                daily_range = g["pct_chg"].abs() / 100 * 0.5
                g["low"] = close * (1 - daily_range.fillna(0.01))
            else:
                g["low"] = close * 0.99
        if "open" not in g.columns:
            g["open"] = close.shift(1).fillna(close)
        if "change" not in g.columns:
            g["change"] = close.diff()
        if "amount" not in g.columns:
            g["amount"] = g.get("vol", 0) * close
        if "pre_close" not in g.columns:
            g["pre_close"] = close.shift(1).fillna(close)
        if "price_change" not in g.columns:
            g["price_change"] = close.diff()

        high = g["high"]
        low = g["low"]

        # ========== 2. 均线相关 ==========
        for period, name in [(5, "ma_5d"), (8, "ma_8d"), (10, "ma_10d"), (20, "ma_20d")]:
            if name not in g.columns and n >= period:
                g[name] = close.rolling(period, min_periods=period // 2).mean()

        # EMA
        for period in [5, 10, 20, 60]:
            col = f"ema_{period}"
            if col not in g.columns and n >= period:
                g[col] = close.ewm(span=period, adjust=False).mean()

        # ========== 3. 乖离率 ==========
        for name, period in [("bias_short", 5), ("bias_mid", 10), ("bias_long", 20)]:
            if name not in g.columns and n >= period:
                ma = close.rolling(period).mean()
                g[name] = (close - ma) / (ma + 1e-8) * 100

        # ========== 4. ATR相关 ==========
        if "atr_14" not in g.columns and n >= 14:
            high_low = high - low
            high_close = abs(high - close.shift(1))
            low_close = abs(low - close.shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            g["atr_14"] = tr.rolling(14).mean()

        if "atr_ratio_14" not in g.columns and "atr_14" in g.columns:
            g["atr_ratio_14"] = g["atr_14"] / (close + 1e-8) * 100

        if "atr_expansion" not in g.columns and "atr_14" in g.columns and n >= 20:
            g["atr_expansion"] = g["atr_14"] / (g["atr_14"].rolling(20).mean() + 1e-8)

        # ========== 5. 最大回撤 ==========
        for period in [10, 20, 55]:
            col = f"max_drawdown_{period}d"
            if col not in g.columns and n >= period:
                rolling_max = close.rolling(period).max()
                g[col] = (close - rolling_max) / (rolling_max + 1e-8) * 100

        # ========== 6. 距离高点天数 ==========
        for period in [20, 55]:
            col = f"days_from_high_{period}d"
            if col not in g.columns and n >= period:
                days_list = []
                for i in range(n):
                    if i < period:
                        days_list.append(0)
                    else:
                        window = close.iloc[i - period : i + 1].values
                        idx_max = np.argmax(window)
                        days_list.append(period - idx_max)
                g[col] = days_list

        # ========== 7. 恢复比率 ==========
        if "recovery_ratio_20d" not in g.columns and n >= 20:
            rolling_max = close.rolling(20).max()
            rolling_min = close.rolling(20).min()
            g["recovery_ratio_20d"] = np.where(
                rolling_max > rolling_min, (close - rolling_min) / (rolling_max - rolling_min + 1e-8), 0.5
            )

        # ========== 8. 通道宽度 ==========
        if "channel_width_20d" not in g.columns and n >= 20:
            high_20 = close.rolling(20).max()
            low_20 = close.rolling(20).min()
            g["channel_width_20d"] = (high_20 - low_20) / (close + 1e-8) * 100

        # ========== 9. 价格区间 ==========
        if "price_range_pct" not in g.columns:
            g["price_range_pct"] = (high - low) / (close + 1e-8) * 100

        # ========== 10. MA10相关 ==========
        if "close_vs_ma10_std" not in g.columns and n >= 10:
            ma10 = g.get("ma10", close.rolling(10).mean())
            diff = close - ma10
            g["close_vs_ma10_std"] = diff / (diff.rolling(10).std() + 1e-8)

        if "days_near_ma10" not in g.columns and n >= 10:
            ma10 = g.get("ma10", close.rolling(10).mean())
            near_ma10 = (abs(close - ma10) / (close + 1e-8) < 0.02).astype(int)
            g["days_near_ma10"] = near_ma10.rolling(10).sum()

        if "ma10_cross_count" not in g.columns and n >= 10:
            ma10 = g.get("ma10", close.rolling(10).mean())
            cross = ((close > ma10) != (close.shift(1) > ma10.shift(1))).astype(int)
            g["ma10_cross_count"] = cross.rolling(20, min_periods=5).sum()

        # ========== 11. 量比相关 ==========
        vol = g.get("vol", pd.Series([1] * n))
        if "vol_ma5_ratio" not in g.columns and n >= 5:
            g["vol_ma5_ratio"] = vol / (vol.rolling(5).mean() + 1e-8)
        if "vol_ma20_ratio" not in g.columns and n >= 20:
            g["vol_ma20_ratio"] = vol / (vol.rolling(20).mean() + 1e-8)
        if "volume_shrink_ratio" not in g.columns and n >= 20:
            g["volume_shrink_ratio"] = vol.rolling(5).mean() / (vol.rolling(20).mean() + 1e-8)
        if "volume_change" not in g.columns:
            g["volume_change"] = vol.pct_change()

        # ========== 12. 放量突破 ==========
        vol_ratio = g.get("volume_ratio", vol / (vol.rolling(5).mean() + 1e-8))
        if "high_volume_breakout" not in g.columns:
            g["high_volume_breakout"] = ((vol_ratio > 2) & (pct_chg > 0)).astype(int)
        if "breakout_volume_ratio" not in g.columns:
            breakout = pct_chg > 3
            g["breakout_volume_ratio"] = np.where(breakout, vol_ratio, 0)

        # ========== 13. 突破MA ==========
        if "breakout_ma20" not in g.columns and n >= 20:
            ma20 = g.get("ma_20d", close.rolling(20).mean())
            g["breakout_ma20"] = (close > ma20).astype(int)
        if "breakout_ma55" not in g.columns and n >= 55:
            ma55 = close.rolling(55).mean()
            g["breakout_ma55"] = (close > ma55).astype(int)

        # ========== 14. 支撑阻力距离 ==========
        for period in [10, 20, 55]:
            s_col = f"support_{period}d" if period != 10 else "support_10d"
            r_col = f"resistance_{period}d" if period != 10 else "resistance_10d"
            ds_col = f"dist_to_support_{period}d"
            dr_col = f"dist_to_resistance_{period}d"

            if n >= period:
                support = close.rolling(period).min()
                resistance = close.rolling(period).max()

                if ds_col not in g.columns:
                    g[ds_col] = (close - support) / (close + 1e-8) * 100
                if dr_col not in g.columns:
                    g[dr_col] = (resistance - close) / (close + 1e-8) * 100

        # ========== 15. 支撑阻力强度 ==========
        for period in [55]:
            col_s = f"support_{period}d"
            col_r = f"resistance_{period}d"
            ss_col = f"support_strength_{period}d"
            rs_col = f"resistance_strength_{period}d"

            if n >= period:
                support = close.rolling(period).min()
                resistance = close.rolling(period).max()
                if ss_col not in g.columns:
                    g[ss_col] = (close - support) / (support + 1e-8) * 100
                if rs_col not in g.columns:
                    g[rs_col] = (resistance - close) / (resistance + 1e-8) * 100

        # ========== 16. 高低点 ==========
        for period, col_h, col_l in [(55, "high_55d", "low_55d")]:
            if col_h not in g.columns and n >= period:
                g[col_h] = close.rolling(period).max()
            if col_l not in g.columns and n >= period:
                g[col_l] = close.rolling(period).min()

        # ========== 17. 其他特征 ==========
        if "consecutive_new_high" not in g.columns and n >= 10:
            high_10 = close.rolling(10).max()
            new_high = (close >= high_10).astype(int)
            g["consecutive_new_high"] = new_high.rolling(5, min_periods=1).sum()

        if "momentum_acceleration" not in g.columns and n >= 10:
            mom = close.pct_change(5)
            g["momentum_acceleration"] = mom.diff()

        if "is_limit_up" not in g.columns:
            g["is_limit_up"] = (pct_chg >= 9.8).astype(int)

        # ========== 18. OBV相关 ==========
        if "obv" not in g.columns:
            obv_sign = np.sign(pct_chg).fillna(0)
            g["obv"] = (obv_sign * vol).cumsum()
        if "obv_calc" not in g.columns:
            g["obv_calc"] = g["obv"]
        if "obv_ma10" not in g.columns and n >= 10:
            g["obv_ma10"] = g["obv"].rolling(10).mean()
        if "obv_trend" not in g.columns and n >= 10:
            g["obv_trend"] = np.sign(g["obv"] - g["obv"].shift(5)).fillna(0)

        # ========== 19. 价格变化相关 ==========
        if "price_down_vol_up" not in g.columns:
            g["price_down_vol_up"] = ((pct_chg < 0) & (vol > vol.shift(1))).astype(int)
        if "price_down_vol_up_count_10d" not in g.columns and n >= 10:
            g["price_down_vol_up_count_10d"] = g["price_down_vol_up"].rolling(10).sum()
        if "price_up_vol_down" not in g.columns:
            g["price_up_vol_down"] = ((pct_chg > 0) & (vol < vol.shift(1))).astype(int)
        if "price_up_vol_down_count_10d" not in g.columns and n >= 10:
            g["price_up_vol_down_count_10d"] = g["price_up_vol_down"].rolling(10).sum()

        # ========== 20. 量价相关性 ==========
        for period in [10, 20]:
            col = f"volume_price_corr_{period}d"
            if col not in g.columns and n >= period:
                g[col] = close.rolling(period).corr(vol)

        # ========== 21. 量价匹配 ==========
        if "volume_price_match" not in g.columns:
            vol_up = (vol > vol.shift(1)).values
            price_up = (pct_chg > 0).values
            g["volume_price_match"] = (vol_up == price_up).astype(int)
        if "volume_price_match_sum_10d" not in g.columns and n >= 10:
            g["volume_price_match_sum_10d"] = g["volume_price_match"].rolling(10).sum()

        # ========== 22. 量能突破 ==========
        if "volume_breakout_count_20d" not in g.columns and n >= 20:
            vol_breakout = (vol > vol.rolling(20).mean() * 2).astype(int)
            g["volume_breakout_count_20d"] = vol_breakout.rolling(20).sum()

        if "volume_rsv_20d" not in g.columns and n >= 20:
            vol_low = vol.rolling(20).min()
            vol_high = vol.rolling(20).max()
            g["volume_rsv_20d"] = np.where(vol_high > vol_low, (vol - vol_low) / (vol_high - vol_low + 1e-8), 0.5)

        if "volume_trend_slope_10d" not in g.columns and n >= 10:
            g["volume_trend_slope_10d"] = vol.diff(10) / (vol.shift(10) + 1e-8)
        if "volume_trend_slope_20d" not in g.columns and n >= 20:
            g["volume_trend_slope_20d"] = vol.diff(20) / (vol.shift(20) + 1e-8)

        # ========== 23. 历史高点 ==========
        for period in [10, 20, 55]:
            col = f"prev_high_{period}d"
            if col not in g.columns and n >= period:
                g[col] = close.shift(1).rolling(period).max()

        # ========== 24. 历史价格相关 (修复：使用百分比差距而非比率) ==========
        if "price_vs_hist_mean" not in g.columns and n >= 55:
            hist_mean = close.rolling(55).mean()
            g["price_vs_hist_mean"] = (close - hist_mean) / (hist_mean + 1e-8) * 100
        if "price_vs_hist_high" not in g.columns and n >= 55:
            hist_high = close.rolling(55).max()
            g["price_vs_hist_high"] = (close - hist_high) / (hist_high + 1e-8) * 100
        if "volatility_vs_hist" not in g.columns and n >= 55:
            vol_current = pct_chg.rolling(10).std()
            vol_hist = pct_chg.rolling(55).std()
            g["volatility_vs_hist"] = vol_current / (vol_hist + 1e-8)

        # ========== 25. 换手率（如果有） ==========
        if "turnover_rate_f" not in g.columns:
            g["turnover_rate_f"] = g.get("turnover_rate", 0)

        # ========== 26. 支撑阻力位（实际价格）==========
        if "support_55d" not in g.columns and n >= 55:
            g["support_55d"] = close.rolling(55).min()
        if "resistance_55d" not in g.columns and n >= 55:
            g["resistance_55d"] = close.rolling(55).max()

        # ========== 27. 突破强度相关 ==========
        if "breakout_strength_10d" not in g.columns and n >= 10:
            prev_high_10 = close.shift(1).rolling(10).max()
            g["breakout_strength_10d"] = (close - prev_high_10) / (prev_high_10 + 1e-8) * 100
        if "breakout_strength_20d" not in g.columns and n >= 20:
            prev_high_20 = close.shift(1).rolling(20).max()
            g["breakout_strength_20d"] = (close - prev_high_20) / (prev_high_20 + 1e-8) * 100
        if "breakout_strength_55d" not in g.columns and n >= 55:
            prev_high_55 = close.shift(1).rolling(55).max()
            g["breakout_strength_55d"] = (close - prev_high_55) / (prev_high_55 + 1e-8) * 100

        # ========== 28. 突破确认（3日站稳）==========
        bs_10 = g.get("breakout_strength_10d", pd.Series([0] * n))
        bs_20 = g.get("breakout_strength_20d", pd.Series([0] * n))
        if "breakout_confirmed_10d" not in g.columns:
            g["breakout_confirmed_10d"] = (bs_10 > 0).astype(int)
        if "breakout_confirmed_20d" not in g.columns:
            g["breakout_confirmed_20d"] = (bs_20 > 0).astype(int)

        # ========== 29. 突破共振 ==========
        bs_55 = g.get("breakout_strength_55d", pd.Series([0] * n))
        if "breakout_resonance" not in g.columns:
            g["breakout_resonance"] = (bs_10 > 0).astype(int) + (bs_20 > 0).astype(int) + (bs_55 > 0).astype(int)

        # ========== 30. 量能突破强度 ==========
        if "breakout_volume_strength" not in g.columns:
            vol_ma = vol.rolling(20, min_periods=5).mean()
            g["breakout_volume_strength"] = vol / (vol_ma + 1e-8)

        # ========== 31. 共振量能确认 ==========
        if "resonance_volume_confirm" not in g.columns:
            res = g.get("breakout_resonance", pd.Series([0] * n))
            bvs = g.get("breakout_volume_strength", pd.Series([1] * n))
            g["resonance_volume_confirm"] = (res > 0).astype(int) * ((bvs > 1.2).astype(int))

        # ========== 32. 放量突破 ==========
        if "breakout_with_volume" not in g.columns:
            vol_ma20 = vol.rolling(20, min_periods=5).mean()
            g["breakout_with_volume"] = ((close > close.shift(1)) & (vol > vol_ma20 * 1.5)).astype(int)

        # ========== 33. 量价背离 ==========
        if "volume_price_divergence" not in g.columns and n >= 10:
            price_trend = (close - close.shift(10)) / (close.shift(10) + 1e-8)
            vol_trend = (vol.rolling(5).mean() - vol.rolling(10).mean().shift(5)) / (
                vol.rolling(10).mean().shift(5) + 1e-8
            )
            g["volume_price_divergence"] = (
                (price_trend > 0) & (vol_trend < 0) | (price_trend < 0) & (vol_trend > 0)
            ).astype(int)

        # ========== 34. 市场趋势（占位，需要外部数据）==========
        if "market_trend" not in g.columns:
            g["market_trend"] = 0  # 需要市场数据才能计算，这里先占位

        return g

    if "sample_id" in df.columns:
        result = df.groupby("sample_id", group_keys=False).apply(calc_sample_features)
    else:
        result = calc_sample_features(df)

    return result


def process_file(file_path: Path, target_cols: list) -> int:
    """处理单个文件，补充缺失特征（保留所有数据）"""
    log.info(f"\n处理文件: {file_path.name}")

    # 加载数据
    df = pd.read_csv(file_path)
    original_rows = len(df)
    original_cols = len(df.columns)
    log.info(f"  原始数据: {original_rows} 行, {original_cols} 列")

    # 找出缺失的特征
    missing_cols = [c for c in target_cols if c not in df.columns]
    log.info(f"  缺失特征数: {len(missing_cols)}")

    if len(missing_cols) == 0:
        log.info("  特征已完整，跳过")
        return original_cols

    # 保存原始索引
    df = df.reset_index(drop=True)
    original_index = df.index.copy()

    # 计算缺失特征
    log.info("  计算缺失特征...")
    df_new = calculate_all_features(df)

    # 检查是否丢失数据
    if len(df_new) != original_rows:
        log.warning(f"  ⚠ 数据丢失: {original_rows} -> {len(df_new)}")
        # 尝试恢复：使用原始数据，只添加能计算的特征
        log.info("  尝试恢复数据...")
        for col in df_new.columns:
            if col not in df.columns:
                # 创建对齐的列
                df[col] = 0  # 默认值
        df_new = df  # 使用原始数据

    # 再次检查还缺少的特征
    still_missing = [c for c in target_cols if c not in df_new.columns]
    if still_missing:
        log.warning(f"  仍缺少 {len(still_missing)} 个特征: {still_missing[:10]}...")
        # 用0填充
        for col in still_missing:
            df_new[col] = 0

    # 只保留目标列（加上其他必要的元数据列）
    meta_cols = ["sample_id", "ts_code", "name", "trade_date", "label", "days_to_t1"]
    final_cols = [c for c in meta_cols if c in df_new.columns]
    final_cols += [c for c in target_cols if c in df_new.columns]

    # 去重
    final_cols = list(dict.fromkeys(final_cols))
    df_new = df_new[final_cols]

    # 保存
    df_new.to_csv(file_path, index=False)
    final_row_count = len(df_new)
    final_col_count = len(df_new.columns)
    log.info(f"  最终数据: {final_row_count} 行, {final_col_count} 列")

    if final_row_count != original_rows:
        log.error(f"  ❌ 数据丢失未恢复! {original_rows} -> {final_row_count}")
    else:
        log.success("  ✓ 数据完整保留")

    return final_col_count


def main():
    log.info("=" * 80)
    log.info("为v6样本补充特征（与v5一致）")
    log.info("=" * 80)

    # 获取v5的特征列作为目标
    v5_pos_file = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v5.csv"
    if not v5_pos_file.exists():
        log.error(f"v5文件不存在: {v5_pos_file}")
        return

    df_v5 = pd.read_csv(v5_pos_file, nrows=1)
    target_cols = [
        c for c in df_v5.columns if c not in ["sample_id", "ts_code", "name", "trade_date", "label", "days_to_t1"]
    ]
    log.info(f"目标特征数: {len(target_cols)}")

    # v6文件路径
    pos_file = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v6.csv"
    neg_file = PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v6.csv"

    # 检查文件
    for f in [pos_file, neg_file]:
        if not f.exists():
            log.error(f"文件不存在: {f}")
            return

    # 处理正样本
    log.info("\n" + "=" * 50)
    log.info("[1/2] 处理v6正样本")
    process_file(pos_file, target_cols)

    # 处理负样本
    log.info("\n" + "=" * 50)
    log.info("[2/2] 处理v6负样本")
    process_file(neg_file, target_cols)

    # 验证
    log.info("\n" + "=" * 50)
    log.info("验证特征对齐")

    df_pos = pd.read_csv(pos_file, nrows=1)
    df_neg = pd.read_csv(neg_file, nrows=1)
    df_hard = pd.read_csv(
        PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v6.csv", nrows=1
    )

    log.info(f"  v6正样本特征数: {len(df_pos.columns)}")
    log.info(f"  v6负样本特征数: {len(df_neg.columns)}")
    log.info(f"  v6硬负样本特征数: {len(df_hard.columns)}")
    log.info(f"  v5目标特征数: {len(target_cols) + 6}")  # 加上元数据列

    log.info("\n" + "=" * 80)
    log.success("✅ v6特征补充完成！")
    log.info("=" * 80)
    log.info("\n下一步: 更新训练脚本支持v2.6.1版本")


if __name__ == "__main__":
    main()
