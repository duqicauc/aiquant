#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新生成v6负样本并补充特征（不丢失数据版本）

解决 align_v6_to_v5_features.py 导致数据丢失的问题
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
from src.data.data_manager import DataManager
from src.models.screening.negative_sample_screener_v2 import NegativeSampleScreenerV2


def add_features_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """
    直接在DataFrame上计算特征，不使用groupby避免数据丢失
    """
    log.info("计算补充特征...")

    df = df.copy()
    n = len(df)

    # 确保有基础数据
    close = df["close"].values if "close" in df.columns else np.ones(n)
    pct_chg = df["pct_chg"].values if "pct_chg" in df.columns else np.zeros(n)

    # 估算缺失的基础列
    if "high" not in df.columns:
        df["high"] = df["close"] * 1.01
    if "low" not in df.columns:
        df["low"] = df["close"] * 0.99
    if "open" not in df.columns:
        df["open"] = df["close"]
    if "change" not in df.columns:
        df["change"] = df["close"].diff()
    if "amount" not in df.columns:
        df["amount"] = df.get("vol", 0) * df["close"]
    if "pre_close" not in df.columns:
        df["pre_close"] = df["close"].shift(1).fillna(df["close"])
    if "price_change" not in df.columns:
        df["price_change"] = df["close"].diff()
    if "vol" not in df.columns:
        df["vol"] = 0
    if "turnover_rate" not in df.columns:
        df["turnover_rate"] = 0
    if "turnover_rate_f" not in df.columns:
        df["turnover_rate_f"] = df.get("turnover_rate", 0)

    # 按sample_id分组计算滚动特征
    log.info("按样本分组计算滚动特征...")

    def safe_rolling(series, window, func="mean", min_periods=None):
        """安全的滚动计算"""
        if min_periods is None:
            min_periods = max(1, window // 2)
        if func == "mean":
            return series.rolling(window, min_periods=min_periods).mean()
        elif func == "std":
            return series.rolling(window, min_periods=min_periods).std()
        elif func == "max":
            return series.rolling(window, min_periods=min_periods).max()
        elif func == "min":
            return series.rolling(window, min_periods=min_periods).min()
        elif func == "sum":
            return series.rolling(window, min_periods=min_periods).sum()
        return series

    # 分组处理
    groups = df.groupby("sample_id", group_keys=False)

    # EMA
    for period in [5, 10, 20, 60]:
        col = f"ema_{period}"
        if col not in df.columns:
            df[col] = groups["close"].transform(lambda x: x.ewm(span=period, adjust=False, min_periods=1).mean())

    # MA
    for period, name in [(5, "ma_5d"), (8, "ma_8d"), (10, "ma_10d"), (20, "ma_20d")]:
        if name not in df.columns:
            df[name] = groups["close"].transform(lambda x: safe_rolling(x, period, "mean"))

    # 乖离率
    for name, period in [("bias_short", 5), ("bias_mid", 10), ("bias_long", 20)]:
        if name not in df.columns:
            ma = groups["close"].transform(lambda x: safe_rolling(x, period, "mean"))
            df[name] = (df["close"] - ma) / (ma + 1e-8) * 100

    # ATR
    if "atr_14" not in df.columns:
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = (
            groups.apply(
                lambda g: safe_rolling(g[tr.name] if tr.name in g.columns else tr.loc[g.index], 14, "mean")
            ).values
            if "sample_id" in df.columns
            else safe_rolling(tr, 14, "mean")
        )
        # 简化：直接用整体滚动
        df["atr_14"] = safe_rolling(tr, 14, "mean")

    if "atr_ratio_14" not in df.columns:
        df["atr_ratio_14"] = df["atr_14"] / (df["close"] + 1e-8) * 100

    if "atr_expansion" not in df.columns:
        df["atr_expansion"] = df["atr_14"] / (safe_rolling(df["atr_14"], 20, "mean") + 1e-8)

    # 最大回撤
    for period in [10, 20, 55]:
        col = f"max_drawdown_{period}d"
        if col not in df.columns:
            rolling_max = groups["close"].transform(lambda x: safe_rolling(x, period, "max"))
            df[col] = (df["close"] - rolling_max) / (rolling_max + 1e-8) * 100

    # 距高点天数 (简化版)
    for period in [20, 55]:
        col = f"days_from_high_{period}d"
        if col not in df.columns:
            df[col] = 0  # 简化处理

    # 恢复比率
    if "recovery_ratio_20d" not in df.columns:
        rolling_max = groups["close"].transform(lambda x: safe_rolling(x, 20, "max"))
        rolling_min = groups["close"].transform(lambda x: safe_rolling(x, 20, "min"))
        df["recovery_ratio_20d"] = np.where(
            rolling_max > rolling_min, (df["close"] - rolling_min) / (rolling_max - rolling_min + 1e-8), 0.5
        )

    # 通道宽度
    if "channel_width_20d" not in df.columns:
        high_20 = groups["close"].transform(lambda x: safe_rolling(x, 20, "max"))
        low_20 = groups["close"].transform(lambda x: safe_rolling(x, 20, "min"))
        df["channel_width_20d"] = (high_20 - low_20) / (df["close"] + 1e-8) * 100

    # 价格区间
    if "price_range_pct" not in df.columns:
        df["price_range_pct"] = (df["high"] - df["low"]) / (df["close"] + 1e-8) * 100

    # MA10相关
    if "close_vs_ma10_std" not in df.columns:
        ma10 = df.get("ma10", groups["close"].transform(lambda x: safe_rolling(x, 10, "mean")))
        diff = df["close"] - ma10
        diff_std = (
            groups.apply(lambda g: safe_rolling(g["close"] - ma10.loc[g.index], 10, "std")).values
            if "sample_id" in df.columns
            else safe_rolling(diff, 10, "std")
        )
        df["close_vs_ma10_std"] = diff / (safe_rolling(diff, 10, "std") + 1e-8)

    if "days_near_ma10" not in df.columns:
        ma10 = df.get("ma10", groups["close"].transform(lambda x: safe_rolling(x, 10, "mean")))
        near_ma10 = (abs(df["close"] - ma10) / (df["close"] + 1e-8) < 0.02).astype(int)
        df["days_near_ma10"] = (
            groups.apply(lambda g: safe_rolling(near_ma10.loc[g.index], 10, "sum")).values
            if "sample_id" in df.columns
            else safe_rolling(near_ma10, 10, "sum")
        )

    if "ma10_cross_count" not in df.columns:
        df["ma10_cross_count"] = 0  # 简化

    # 量比相关
    vol = df.get("vol", pd.Series([1] * n))
    if "vol_ma5_ratio" not in df.columns:
        df["vol_ma5_ratio"] = (
            vol / (groups["vol"].transform(lambda x: safe_rolling(x, 5, "mean")) + 1e-8)
            if "sample_id" in df.columns
            else vol / (safe_rolling(vol, 5, "mean") + 1e-8)
        )
    if "vol_ma20_ratio" not in df.columns:
        df["vol_ma20_ratio"] = (
            vol / (groups["vol"].transform(lambda x: safe_rolling(x, 20, "mean")) + 1e-8)
            if "sample_id" in df.columns
            else vol / (safe_rolling(vol, 20, "mean") + 1e-8)
        )
    if "volume_shrink_ratio" not in df.columns:
        vol_ma5 = (
            groups["vol"].transform(lambda x: safe_rolling(x, 5, "mean"))
            if "sample_id" in df.columns
            else safe_rolling(vol, 5, "mean")
        )
        vol_ma20 = (
            groups["vol"].transform(lambda x: safe_rolling(x, 20, "mean"))
            if "sample_id" in df.columns
            else safe_rolling(vol, 20, "mean")
        )
        df["volume_shrink_ratio"] = vol_ma5 / (vol_ma20 + 1e-8)
    if "volume_change" not in df.columns:
        df["volume_change"] = vol.pct_change()

    # 放量突破
    vol_ratio = df.get("volume_ratio", vol / (safe_rolling(vol, 5, "mean") + 1e-8))
    pct_chg_series = df["pct_chg"]
    if "high_volume_breakout" not in df.columns:
        df["high_volume_breakout"] = ((vol_ratio > 2) & (pct_chg_series > 0)).astype(int)
    if "breakout_volume_ratio" not in df.columns:
        breakout = pct_chg_series > 3
        df["breakout_volume_ratio"] = np.where(breakout, vol_ratio, 0)

    # 突破MA
    if "breakout_ma20" not in df.columns:
        ma20 = df.get("ma_20d", groups["close"].transform(lambda x: safe_rolling(x, 20, "mean")))
        df["breakout_ma20"] = (df["close"] > ma20).astype(int)
    if "breakout_ma55" not in df.columns:
        ma55 = (
            groups["close"].transform(lambda x: safe_rolling(x, 55, "mean"))
            if "sample_id" in df.columns
            else safe_rolling(df["close"], 55, "mean")
        )
        df["breakout_ma55"] = (df["close"] > ma55).astype(int)

    # 支撑阻力距离
    for period in [10, 20, 55]:
        ds_col = f"dist_to_support_{period}d"
        dr_col = f"dist_to_resistance_{period}d"

        support = groups["close"].transform(lambda x: safe_rolling(x, period, "min"))
        resistance = groups["close"].transform(lambda x: safe_rolling(x, period, "max"))

        if ds_col not in df.columns:
            df[ds_col] = (df["close"] - support) / (df["close"] + 1e-8) * 100
        if dr_col not in df.columns:
            df[dr_col] = (resistance - df["close"]) / (df["close"] + 1e-8) * 100

    # 支撑阻力强度 55d
    if "support_strength_55d" not in df.columns:
        support = groups["close"].transform(lambda x: safe_rolling(x, 55, "min"))
        df["support_strength_55d"] = (df["close"] - support) / (support + 1e-8) * 100
    if "resistance_strength_55d" not in df.columns:
        resistance = groups["close"].transform(lambda x: safe_rolling(x, 55, "max"))
        df["resistance_strength_55d"] = (resistance - df["close"]) / (resistance + 1e-8) * 100

    # 55d支撑阻力值
    if "support_55d" not in df.columns:
        df["support_55d"] = groups["close"].transform(lambda x: safe_rolling(x, 55, "min"))
    if "resistance_55d" not in df.columns:
        df["resistance_55d"] = groups["close"].transform(lambda x: safe_rolling(x, 55, "max"))

    # 高低点
    if "high_55d" not in df.columns:
        df["high_55d"] = groups["close"].transform(lambda x: safe_rolling(x, 55, "max"))
    if "low_55d" not in df.columns:
        df["low_55d"] = groups["close"].transform(lambda x: safe_rolling(x, 55, "min"))

    # 其他特征
    if "consecutive_new_high" not in df.columns:
        high_10 = groups["close"].transform(lambda x: safe_rolling(x, 10, "max"))
        new_high = (df["close"] >= high_10).astype(int)
        df["consecutive_new_high"] = (
            groups.apply(lambda g: safe_rolling(new_high.loc[g.index], 5, "sum")).values
            if "sample_id" in df.columns
            else safe_rolling(new_high, 5, "sum")
        )

    if "momentum_acceleration" not in df.columns:
        mom = df["close"].pct_change(5)
        df["momentum_acceleration"] = mom.diff()

    if "is_limit_up" not in df.columns:
        df["is_limit_up"] = (df["pct_chg"] >= 9.8).astype(int)

    # OBV相关
    if "obv" not in df.columns:
        obv_sign = np.sign(df["pct_chg"]).fillna(0)
        df["obv"] = (obv_sign * vol).cumsum()
    if "obv_calc" not in df.columns:
        df["obv_calc"] = df["obv"]
    if "obv_ma10" not in df.columns:
        df["obv_ma10"] = safe_rolling(df["obv"], 10, "mean")
    if "obv_trend" not in df.columns:
        df["obv_trend"] = np.sign(df["obv"] - df["obv"].shift(5)).fillna(0)

    # 价格变化相关
    if "price_down_vol_up" not in df.columns:
        df["price_down_vol_up"] = ((df["pct_chg"] < 0) & (vol > vol.shift(1))).astype(int)
    if "price_down_vol_up_count_10d" not in df.columns:
        df["price_down_vol_up_count_10d"] = (
            groups.apply(lambda g: safe_rolling(g["price_down_vol_up"], 10, "sum")).values
            if "sample_id" in df.columns
            else safe_rolling(df["price_down_vol_up"], 10, "sum")
        )
    if "price_up_vol_down" not in df.columns:
        df["price_up_vol_down"] = ((df["pct_chg"] > 0) & (vol < vol.shift(1))).astype(int)
    if "price_up_vol_down_count_10d" not in df.columns:
        df["price_up_vol_down_count_10d"] = (
            groups.apply(lambda g: safe_rolling(g["price_up_vol_down"], 10, "sum")).values
            if "sample_id" in df.columns
            else safe_rolling(df["price_up_vol_down"], 10, "sum")
        )

    # 量价相关性
    for period in [10, 20]:
        col = f"volume_price_corr_{period}d"
        if col not in df.columns:
            df[col] = (
                groups.apply(lambda g: g["close"].rolling(period, min_periods=5).corr(g["vol"])).values
                if "sample_id" in df.columns
                else df["close"].rolling(period, min_periods=5).corr(vol)
            )

    # 量价匹配
    if "volume_price_match" not in df.columns:
        vol_up = vol > vol.shift(1)
        price_up = df["pct_chg"] > 0
        df["volume_price_match"] = (vol_up.values == price_up.values).astype(int)
    if "volume_price_match_sum_10d" not in df.columns:
        df["volume_price_match_sum_10d"] = (
            groups.apply(lambda g: safe_rolling(g["volume_price_match"], 10, "sum")).values
            if "sample_id" in df.columns
            else safe_rolling(df["volume_price_match"], 10, "sum")
        )

    # 量能突破
    if "volume_breakout_count_20d" not in df.columns:
        vol_breakout = (vol > safe_rolling(vol, 20, "mean") * 2).astype(int)
        df["volume_breakout_count_20d"] = (
            groups.apply(lambda g: safe_rolling(vol_breakout.loc[g.index], 20, "sum")).values
            if "sample_id" in df.columns
            else safe_rolling(vol_breakout, 20, "sum")
        )

    if "volume_rsv_20d" not in df.columns:
        vol_low = groups["vol"].transform(lambda x: safe_rolling(x, 20, "min"))
        vol_high = groups["vol"].transform(lambda x: safe_rolling(x, 20, "max"))
        df["volume_rsv_20d"] = np.where(vol_high > vol_low, (vol - vol_low) / (vol_high - vol_low + 1e-8), 0.5)

    if "volume_trend_slope_10d" not in df.columns:
        df["volume_trend_slope_10d"] = vol.diff(10) / (vol.shift(10) + 1e-8)
    if "volume_trend_slope_20d" not in df.columns:
        df["volume_trend_slope_20d"] = vol.diff(20) / (vol.shift(20) + 1e-8)

    # 历史高点
    for period in [10, 20, 55]:
        col = f"prev_high_{period}d"
        if col not in df.columns:
            df[col] = groups["close"].transform(lambda x: x.shift(1).rolling(period, min_periods=1).max())

    # 历史价格相关
    if "price_vs_hist_mean" not in df.columns:
        df["price_vs_hist_mean"] = df["close"] / (
            groups["close"].transform(lambda x: safe_rolling(x, 55, "mean")) + 1e-8
        )
    if "price_vs_hist_high" not in df.columns:
        df["price_vs_hist_high"] = df["close"] / (
            groups["close"].transform(lambda x: safe_rolling(x, 55, "max")) + 1e-8
        )
    if "volatility_vs_hist" not in df.columns:
        vol_current = groups["pct_chg"].transform(lambda x: safe_rolling(x, 10, "std"))
        vol_hist = groups["pct_chg"].transform(lambda x: safe_rolling(x, 55, "std"))
        df["volatility_vs_hist"] = vol_current / (vol_hist + 1e-8)

    log.info(f"特征补充完成，总特征数: {len(df.columns)}")
    return df


def main():
    log.info("=" * 80)
    log.info("重新生成v6负样本并补充特征")
    log.info("=" * 80)

    # 获取v5特征列作为目标
    v5_file = PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v5.csv"
    df_v5 = pd.read_csv(v5_file, nrows=1)
    target_cols = list(df_v5.columns)
    log.info(f"目标特征数: {len(target_cols)}")

    # 加载正样本获取基本信息
    log.info("\n加载正样本信息...")
    pos_df = pd.read_csv(PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v6.csv")

    positive_samples = (
        pos_df.groupby("sample_id")
        .agg({"ts_code": "first", "name": "first", "trade_date": "max", "circ_mv": "first"})
        .reset_index()
    )
    positive_samples["trade_date"] = pd.to_datetime(positive_samples["trade_date"])
    positive_samples["t1_date"] = positive_samples["trade_date"].dt.strftime("%Y%m%d")

    log.info(f"正样本数量: {len(positive_samples)}")

    # 初始化
    dm = DataManager()
    screener = NegativeSampleScreenerV2(dm)

    # 筛选负样本
    log.info("\n筛选负样本...")
    negative_samples = screener.screen_negative_samples(
        positive_samples_df=positive_samples, samples_per_positive=1, random_seed=42, stratified_by_mv=False
    )

    log.info(f"筛选到 {len(negative_samples)} 个负样本")

    # 提取特征
    log.info("\n提取负样本特征（70天）...")
    negative_features = screener.extract_features(negative_samples_df=negative_samples, lookback_days=70)

    log.info(f"原始特征数据: {len(negative_features)} 行, {len(negative_features.columns)} 列")

    # 补充特征
    log.info("\n补充缺失特征...")
    negative_features = add_features_inplace(negative_features)

    # 确保有所有目标列
    for col in target_cols:
        if col not in negative_features.columns:
            negative_features[col] = 0

    # 重排列顺序
    available_cols = [c for c in target_cols if c in negative_features.columns]
    extra_cols = [c for c in negative_features.columns if c not in target_cols]
    negative_features = negative_features[available_cols + extra_cols]

    # 保存
    output_file = PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v6.csv"
    negative_features.to_csv(output_file, index=False)

    log.info(f"\n最终: {len(negative_features)} 行, {len(negative_features.columns)} 列")
    log.success(f"✓ 保存到 {output_file}")


if __name__ == "__main__":
    main()
