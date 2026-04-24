#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为新硬负样本(v5基础数据)计算增强特征，对齐到v5格式

输入: data/training/features/hard_negative_v5_base.csv
输出: data/training/features/hard_negative_feature_data_34d_v5.csv

步骤:
1. 计算基础技术特征 (rolling/均线/动量等)
2. 计算突破特征
3. 添加市场环境特征
4. 计算交互特征
5. 计算v5独有的13个特征
6. 合并市场环境特征(sh_/hs300_)
7. 对齐到v5的206列
"""

import sys
import warnings
import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from src.utils.logger import log

INPUT = "data/training/features/hard_negative_v5_base.csv"
OUTPUT = "data/training/features/hard_negative_feature_data_34d_v5.csv"
MARKET_FEATURES = "data/training/features/market_features.csv"
V5_TARGET_COLS = "data/training/features/v5_target_columns.txt"
DB_PATH = "data/cache/quant_data.db"


# ==================== 1. 基础技术特征 ====================

def calculate_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算基础技术特征（按sample_id分组）"""
    df = df.copy()

    def calc_sample_features(g):
        g = g.sort_values("trade_date").copy()

        if "close" not in g.columns:
            return g

        # 均线
        g["ma5"] = g["close"].rolling(5, min_periods=3).mean()
        g["ma10"] = g["close"].rolling(10, min_periods=5).mean()
        g["ma_34d"] = g["close"].rolling(34, min_periods=10).mean()
        g["ma_55d"] = g["close"].rolling(55, min_periods=20).mean()
        g["ma_8d"] = g["close"].rolling(8, min_periods=3).mean()

        # 波动率
        if "pct_chg" in g.columns:
            g["volatility_8d"] = g["pct_chg"].rolling(8, min_periods=3).std()
            g["volatility_34d"] = g["pct_chg"].rolling(34, min_periods=10).std()
            g["volatility_55d"] = g["pct_chg"].rolling(55, min_periods=20).std()
            g["volatility_vs_hist"] = g["volatility_8d"] / (g["volatility_34d"].rolling(20).mean() + 1e-6)

        # 动量
        g["momentum_5d"] = g["close"].pct_change(5) * 100
        g["momentum_10d"] = g["close"].pct_change(10) * 100
        g["momentum_20d"] = g["close"].pct_change(20) * 100

        # 收益率
        g["return_8d"] = g["close"].pct_change(8) * 100
        g["return_34d"] = g["close"].pct_change(34) * 100
        g["return_55d"] = g["close"].pct_change(55) * 100

        # 高低点
        g["high_8d"] = g["close"].rolling(8, min_periods=3).max()
        g["low_8d"] = g["close"].rolling(8, min_periods=3).min()
        g["high_10d"] = g["close"].rolling(10, min_periods=5).max()
        g["high_20d"] = g["close"].rolling(20, min_periods=10).max()
        g["high_34d"] = g["close"].rolling(34, min_periods=10).max()
        g["low_34d"] = g["close"].rolling(34, min_periods=10).min()
        g["high_55d"] = g["close"].rolling(55, min_periods=20).max()
        g["low_55d"] = g["close"].rolling(55, min_periods=20).min()

        # 前高
        g["prev_high_55d"] = g["high_55d"].shift(1)

        # 价格位置
        g["price_position_8d"] = np.where(
            g["high_8d"] > g["low_8d"], (g["close"] - g["low_8d"]) / (g["high_8d"] - g["low_8d"]), 0.5
        )
        g["price_position_34d"] = np.where(
            g["high_34d"] > g["low_34d"], (g["close"] - g["low_34d"]) / (g["high_34d"] - g["low_34d"]), 0.5
        )
        g["price_position_55d"] = np.where(
            g["high_55d"] > g["low_55d"], (g["close"] - g["low_55d"]) / (g["high_55d"] - g["low_55d"]), 0.5
        )

        # 价格vs均线
        g["price_vs_ma_8d"] = np.where(g["ma_8d"] != 0, (g["close"] - g["ma_8d"]) / g["ma_8d"] * 100, 0)
        g["price_vs_ma_34d"] = np.where(g["ma_34d"] != 0, (g["close"] - g["ma_34d"]) / g["ma_34d"] * 100, 0)
        g["price_vs_ma_55d"] = np.where(g["ma_55d"] != 0, (g["close"] - g["ma_55d"]) / g["ma_55d"] * 100, 0)

        # 趋势斜率
        g["trend_slope_8d"] = g["close"].diff(8) / g["close"].shift(8) * 100
        g["trend_slope_34d"] = g["close"].diff(34) / g["close"].shift(34) * 100
        g["trend_slope_55d"] = g["close"].diff(55) / g["close"].shift(55) * 100

        # KDJ（如无Tushare因子则本地计算）
        if "kdj_k" not in g.columns:
            low_9 = g["close"].rolling(9, min_periods=5).min()
            high_9 = g["close"].rolling(9, min_periods=5).max()
            rsv = np.where(high_9 > low_9, (g["close"] - low_9) / (high_9 - low_9) * 100, 50)
            g["kdj_k"] = pd.Series(rsv).ewm(com=2, adjust=False).mean().values
            g["kdj_d"] = g["kdj_k"].ewm(com=2, adjust=False).mean()
            g["kdj_j"] = 3 * g["kdj_k"] - 2 * g["kdj_d"]

        # 突破信号
        g["breakout_high_10d"] = (g["close"] > g["high_10d"].shift(1)).astype(int)
        g["breakout_high_20d"] = (g["close"] > g["high_20d"].shift(1)).astype(int)
        g["breakout_high_55d"] = (g["close"] > g["high_55d"].shift(1)).astype(int)
        g["breakout_ma5"] = (g["close"] > g["ma5"]).astype(int)
        g["breakout_ma10"] = (g["close"] > g["ma10"]).astype(int)

        # 支撑/阻力
        g["support_10d"] = g["close"].rolling(10, min_periods=5).min()
        g["resistance_10d"] = g["close"].rolling(10, min_periods=5).max()
        g["support_20d"] = g["close"].rolling(20, min_periods=10).min()
        g["resistance_20d"] = g["close"].rolling(20, min_periods=10).max()
        g["support_55d"] = g["close"].rolling(55, min_periods=20).min()
        g["resistance_55d"] = g["close"].rolling(55, min_periods=20).max()

        g["support_strength_10d"] = np.where(g["support_10d"] > 0, (g["close"] - g["support_10d"]) / g["support_10d"] * 100, 0)
        g["resistance_strength_10d"] = np.where(g["resistance_10d"] > 0, (g["resistance_10d"] - g["close"]) / g["resistance_10d"] * 100, 0)
        g["support_strength_20d"] = np.where(g["support_20d"] > 0, (g["close"] - g["support_20d"]) / g["support_20d"] * 100, 0)
        g["resistance_strength_20d"] = np.where(g["resistance_20d"] > 0, (g["resistance_20d"] - g["close"]) / g["resistance_20d"] * 100, 0)
        g["support_strength_55d"] = np.where(g["support_55d"] > 0, (g["close"] - g["support_55d"]) / g["support_55d"] * 100, 0)
        g["resistance_strength_55d"] = np.where(g["resistance_55d"] > 0, (g["resistance_55d"] - g["close"]) / g["resistance_55d"] * 100, 0)

        # OBV
        if "vol" in g.columns:
            g["obv_calc"] = (np.sign(g["close"].diff()) * g["vol"]).cumsum()
            g["obv"] = g["obv_calc"]
            g["obv_ma10"] = g["obv"].rolling(10, min_periods=5).mean()
            g["obv_trend"] = np.where(g["obv"] > g["obv_ma10"], 1, -1)

        # ATR扩展
        if "high" in g.columns and "low" in g.columns:
            tr1 = g["high"] - g["low"]
            tr2 = (g["high"] - g["close"].shift(1)).abs()
            tr3 = (g["low"] - g["close"].shift(1)).abs()
            atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14, min_periods=5).mean()
            g["atr_expansion"] = atr / (atr.rolling(20).mean() + 1e-6)

        # 均线交叉
        g["days_near_ma10"] = ((g["close"] / g["ma10"] - 1).abs() < 0.02).astype(int)
        g["ma10_cross_count"] = ((g["ma5"] > g["ma10"]) & (g["ma5"].shift(1) <= g["ma10"].shift(1))).astype(int).rolling(20).sum()

        # 量价关系
        if "vol" in g.columns:
            g["volume_change"] = g["vol"] / g["vol"].shift(1).fillna(1) - 1
            g["volume_rsv_20d"] = (g["vol"] - g["vol"].rolling(20).min()) / (g["vol"].rolling(20).max() - g["vol"].rolling(20).min() + 1e-6) * 100
            g["volume_trend_slope_10d"] = g["vol"].diff(10) / (g["vol"].shift(10) + 1e-6)
            g["volume_trend_slope_20d"] = g["vol"].diff(20) / (g["vol"].shift(20) + 1e-6)
            g["price_up_vol_down"] = ((g["pct_chg"] > 0) & (g["volume_change"] < -0.1)).astype(int)
            g["price_up_vol_down_count_10d"] = g["price_up_vol_down"].rolling(10).sum()
            g["price_down_vol_up"] = ((g["pct_chg"] < 0) & (g["volume_change"] > 0.1)).astype(int)
            g["price_down_vol_up_count_10d"] = g["price_down_vol_up"].rolling(10).sum()

        # 换手率相关
        if "turnover_rate" in g.columns:
            g["turnover_change_rate"] = g["turnover_rate"] / g["turnover_rate"].shift(1).fillna(1) - 1
            g["turnover_zscore"] = (g["turnover_rate"] - g["turnover_rate"].rolling(20).mean()) / (g["turnover_rate"].rolling(20).std() + 1e-6)
            g["turnover_spike"] = (g["turnover_rate"] > g["turnover_rate"].rolling(20).mean() * 2).astype(int)

        return g

    if "sample_id" in df.columns:
        result = df.groupby("sample_id", group_keys=False).apply(calc_sample_features)
    else:
        result = calc_sample_features(df)

    return result


# ==================== 2. 突破特征 ====================

def calculate_breakout_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算突破强度特征"""
    df = df.copy()

    for window in [10, 20, 55]:
        col_high = f"high_{window}d"
        col_strength = f"breakout_strength_{window}d"
        if col_high in df.columns:
            df[col_strength] = np.where(df[col_high] > 0, (df["close"] - df[col_high]) / df[col_high] * 100, 0)
        else:
            df[col_strength] = 0

    # 突破强度统计
    strength_cols = [c for c in df.columns if c.startswith("breakout_strength_") and c.endswith("d")]
    if strength_cols:
        df["breakout_strength_avg"] = df[strength_cols].mean(axis=1)
        df["breakout_strength_max"] = df[strength_cols].max(axis=1)

    # 放量突破
    if "vol" in df.columns:
        vol_mean = df.groupby("sample_id")["vol"].transform(lambda x: x.rolling(20, min_periods=5).mean())
        df["breakout_volume_strength"] = np.where(vol_mean > 0, df["vol"] / vol_mean, 1)
    elif "volume_ratio" in df.columns:
        df["breakout_volume_strength"] = df["volume_ratio"].fillna(1)
    else:
        df["breakout_volume_strength"] = 1

    # 突破确认
    df["breakout_confirmed_10d"] = np.where(df.get("breakout_strength_10d", 0) > 0, 1, 0)
    df["breakout_confirmed_20d"] = np.where(df.get("breakout_strength_20d", 0) > 0, 1, 0)

    # 多周期共振
    df["breakout_resonance"] = (
        (df.get("breakout_strength_10d", 0) > 0).astype(int)
        + (df.get("breakout_strength_20d", 0) > 0).astype(int)
        + (df.get("breakout_strength_55d", 0) > 0).astype(int)
    )

    return df


# ==================== 3. 市场环境特征 ====================

def get_market_data_from_db(conn, start_date, end_date):
    """从cache DB获取上证指数数据"""
    df = pd.read_sql(
        f"SELECT trade_date, close, pct_chg FROM daily_data "
        f"WHERE ts_code = '000001.SH' AND trade_date BETWEEN '{start_date}' AND '{end_date}' "
        f"ORDER BY trade_date",
        conn
    )
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("trade_date")
    df["market_pct_chg"] = df["pct_chg"]
    df["market_return_34d"] = df["close"].pct_change(34) * 100
    df["market_volatility_34d"] = df["pct_chg"].rolling(34).std()

    # 计算ma5和ma20用于market_trend
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["market_trend"] = np.where(df["ma5"] > df["ma20"], 1, -1)

    df["market_momentum_5d"] = df["close"].pct_change(5) * 100
    df["market_momentum_10d"] = df["close"].pct_change(10) * 100
    df["market_momentum_20d"] = df["close"].pct_change(20) * 100

    df["market_regime"] = np.where(
        df["market_momentum_20d"] > 5, 1, np.where(df["market_momentum_20d"] < -5, -1, 0)
    )

    roll_min = df["close"].rolling(20).min()
    roll_max = df["close"].rolling(20).max()
    df["market_position_20d"] = np.where(roll_max > roll_min, (df["close"] - roll_min) / (roll_max - roll_min), 0.5)

    return df[["trade_date", "market_pct_chg", "market_return_34d", "market_volatility_34d",
               "market_trend", "market_momentum_5d", "market_momentum_10d", "market_momentum_20d",
               "market_regime", "market_position_20d"]]


def add_market_features(df: pd.DataFrame, df_market: pd.DataFrame) -> pd.DataFrame:
    """添加市场环境特征"""
    df = df.copy()

    # 统一trade_date格式
    df["trade_date_key"] = df["trade_date"].astype(str)
    df_market = df_market.copy()
    df_market["trade_date_key"] = df_market["trade_date"].astype(str)

    df = df.merge(df_market, on="trade_date_key", how="left")
    df = df.drop(columns=["trade_date_key"])

    market_cols = ["market_pct_chg", "market_return_34d", "market_volatility_34d",
                   "market_trend", "market_momentum_5d", "market_momentum_10d",
                   "market_momentum_20d", "market_regime", "market_position_20d"]
    for col in market_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    # 超额收益
    if "pct_chg" in df.columns:
        df["excess_return"] = df["pct_chg"] - df["market_pct_chg"]
    else:
        df["excess_return"] = 0

    if "sample_id" in df.columns:
        df["excess_return_cumsum"] = df.groupby("sample_id")["excess_return"].cumsum()
    else:
        df["excess_return_cumsum"] = df["excess_return"].cumsum()

    df["excess_return_consistency"] = np.where(df["excess_return"] > 0, 1, 0)

    return df


# ==================== 4. 交互特征 ====================

def calculate_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算交互特征"""
    df = df.copy()

    # 突破+放量
    breakout_signal = (df.get("breakout_strength_10d", 0) > 0) | (df.get("breakout_strength_20d", 0) > 0)
    volume_signal = df.get("breakout_volume_strength", 1) > 1.5
    df["breakout_with_volume"] = (breakout_signal & volume_signal).astype(int)

    # 动量+市场环境
    momentum = df.get("momentum_20d", df.get("pct_chg", 0))
    market_trend = df.get("market_trend", 0)
    df["momentum_market_interaction"] = momentum * market_trend

    # RSI+KDJ背离
    rsi = df.get("rsi_6", 50)
    kdj_j = df.get("kdj_j", 50)
    df["rsi_kdj_divergence"] = np.abs(rsi - kdj_j)

    # RSI区域
    df["rsi_zone"] = np.where(rsi > 70, 2, np.where(rsi < 30, 0, 1))

    # RSI+KDJ金叉强度
    kdj_k = df.get("kdj_k", 50)
    kdj_d = df.get("kdj_d", 50)
    df["rsi_kdj_golden_cross"] = ((kdj_k > kdj_d) & (kdj_k.shift(1) <= kdj_d.shift(1))).astype(int)
    df["rsi_kdj_strength"] = kdj_k - kdj_d

    # 趋势一致性
    ma5 = df.get("ma5", df.get("close", 0))
    ma10 = df.get("ma10", ma5)
    df["trend_consistency"] = np.where(
        (ma5 > ma10) & (df.get("pct_chg", 0) > 0), 1,
        np.where((ma5 < ma10) & (df.get("pct_chg", 0) < 0), 1, 0)
    )

    # 均线排列分数
    ma34 = df.get("ma_34d", ma10)
    ma55 = df.get("ma_55d", ma34)
    df["ma_alignment_score"] = np.where(
        (ma5 > ma10) & (ma10 > ma34) & (ma34 > ma55), 3,
        np.where((ma5 > ma10) & (ma10 > ma34), 2,
                 np.where(ma5 > ma10, 1, 0))
    )

    # 价格位置平均
    pp8 = df.get("price_position_8d", 0.5)
    pp34 = df.get("price_position_34d", 0.5)
    pp55 = df.get("price_position_55d", 0.5)
    df["price_position_avg"] = (pp8 + pp34 + pp55) / 3

    # 量价背离
    pct_chg = df.get("pct_chg", pd.Series([0] * len(df)))
    if "vol" in df.columns:
        vol_chg = df["vol"] / df["vol"].shift(1).fillna(1) - 1
    else:
        vol_chg = pd.Series([0] * len(df))
    df["volume_price_divergence"] = np.where(
        (pct_chg > 0) & (vol_chg < -0.2), 1, np.where((pct_chg < 0) & (vol_chg > 0.2), -1, 0)
    )
    df["volume_price_divergence_strength"] = df["volume_price_divergence"].abs()
    df["volume_price_confirm"] = ((pct_chg > 0) & (vol_chg > 0)).astype(int)

    # 突破+RSI
    breakout = df.get("breakout_strength_20d", 0)
    df["breakout_rsi_interaction"] = breakout * (100 - rsi) / 100

    # 相对波动率
    vol_34d = df.get("volatility_34d", pct_chg.rolling(34).std())
    market_vol = df.get("market_volatility_34d", 1)
    df["relative_volatility"] = np.where(market_vol > 0, vol_34d / market_vol, 1)

    # 共振+成交量确认
    resonance = df.get("breakout_resonance", 0)
    vol_confirm = (df.get("breakout_volume_strength", 1) > 1.2).astype(int)
    df["resonance_volume_confirm"] = resonance * vol_confirm

    # 高量突破
    df["high_volume_breakout"] = ((df.get("breakout_strength_10d", 0) > 0) & (df.get("breakout_volume_strength", 1) > 1.5)).astype(int)

    # 量价匹配
    df["volume_price_match"] = ((pct_chg > 0) & (vol_chg > 0) | (pct_chg < 0) & (vol_chg < 0)).astype(int)
    if "sample_id" in df.columns:
        df["volume_price_match_sum_10d"] = df.groupby("sample_id")["volume_price_match"].transform(lambda x: x.rolling(10, min_periods=3).sum())

    # Sharpe-like
    ret_34d = df.get("return_34d", 0)
    vol_34 = df.get("volatility_34d", 1)
    df["sharpe_like_34d"] = np.where(vol_34 > 0, ret_34d / vol_34, 0)

    return df


# ==================== 5. 合并市场环境特征(sh_/hs300_) ====================

def merge_market_env_features(df: pd.DataFrame) -> pd.DataFrame:
    """合并sh_/hs300_市场环境特征"""
    df_market = pd.read_csv(MARKET_FEATURES)

    df["trade_date_key"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y%m%d").astype(int)
    df_market["trade_date_key"] = df_market["trade_date"].astype(int)

    market_cols = [c for c in df_market.columns if c not in ["trade_date", "trade_date_key"]]
    df = df.merge(df_market[["trade_date_key"] + market_cols], on="trade_date_key", how="left")
    df = df.drop(columns=["trade_date_key"])

    for col in market_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# ==================== 6. 主流程 ====================

def main():
    log.info("=" * 80)
    log.info("新硬负样本v5特征工程")
    log.info("=" * 80)

    # 读取基础数据
    log.info(f"读取基础数据: {INPUT}")
    df = pd.read_csv(INPUT)
    log.info(f"  记录数: {len(df)}, 样本数: {df['sample_id'].nunique()}, 列数: {len(df.columns)}")

    # 读取目标列
    with open(V5_TARGET_COLS) as f:
        target_cols = [line.strip() for line in f if line.strip()]
    log.info(f"目标列数: {len(target_cols)}")

    # 1. 基础特征
    log.info("\n[1/5] 计算基础技术特征...")
    df = calculate_basic_features(df)
    log.info(f"  当前列数: {len(df.columns)}")

    # 2. 突破特征
    log.info("\n[2/5] 计算突破特征...")
    df = calculate_breakout_features(df)
    log.info(f"  当前列数: {len(df.columns)}")

    # 3. 市场环境特征
    log.info("\n[3/5] 添加市场环境特征...")
    conn = sqlite3.connect(DB_PATH)
    min_date = df["trade_date"].min()
    max_date = df["trade_date"].max()
    df_market = get_market_data_from_db(conn, min_date, max_date)
    conn.close()

    if not df_market.empty:
        df = add_market_features(df, df_market)
        log.info(f"  市场数据: {len(df_market)} 行")
    else:
        log.warning("  市场数据获取失败")
    log.info(f"  当前列数: {len(df.columns)}")

    # 4. 交互特征
    log.info("\n[4/5] 计算交互特征...")
    df = calculate_interaction_features(df)
    log.info(f"  当前列数: {len(df.columns)}")

    # 5. 合并市场环境特征(sh_/hs300_)
    log.info("\n[5/5] 合并市场环境特征...")
    df = merge_market_env_features(df)
    log.info(f"  当前列数: {len(df.columns)}")

    # 对齐到v5目标列
    log.info("\n对齐到v5目标列...")
    result = pd.DataFrame()
    for col in target_cols:
        if col in df.columns:
            result[col] = df[col]
        else:
            log.warning(f"  缺失列(填0): {col}")
            result[col] = 0

    # 确保label存在
    if "label" not in result.columns:
        result["label"] = 0

    log.info(f"最终列数: {len(result.columns)}")
    log.info(f"最终记录数: {len(result)}")

    # 保存
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    log.success(f"\n已保存: {OUTPUT}")


if __name__ == "__main__":
    main()
