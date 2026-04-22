#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v6样本特征工程脚本

为v6版本的正样本、负样本、硬负样本补充所有增强特征：
1. 突破特征增强
2. 市场环境特征
3. 交互特征

基于v5脚本逻辑，处理v6版本文件
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.data.data_manager import DataManager
from src.utils.logger import log


# ==================== 基础技术指标计算 ====================


def calculate_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算基础技术特征（按样本分组）"""
    df = df.copy()

    # 按样本分组计算
    def calc_sample_features(group):
        g = group.sort_values("trade_date").copy()

        # 基础价格特征
        if "close" in g.columns:
            # 波动率
            g["volatility_8d"] = g["pct_chg"].rolling(8, min_periods=3).std() if "pct_chg" in g.columns else 0
            g["volatility_34d"] = g["pct_chg"].rolling(34, min_periods=10).std() if "pct_chg" in g.columns else 0
            g["volatility_55d"] = g["pct_chg"].rolling(55, min_periods=20).std() if "pct_chg" in g.columns else 0

            # 动量
            g["momentum_5d"] = g["close"].pct_change(5) * 100
            g["momentum_10d"] = g["close"].pct_change(10) * 100
            g["momentum_20d"] = g["close"].pct_change(20) * 100

            # 高低点
            g["high_8d"] = g["close"].rolling(8, min_periods=3).max()
            g["low_8d"] = g["close"].rolling(8, min_periods=3).min()
            g["high_34d"] = g["close"].rolling(34, min_periods=10).max()
            g["low_34d"] = g["close"].rolling(34, min_periods=10).min()
            g["high_55d"] = g["close"].rolling(55, min_periods=20).max()
            g["low_55d"] = g["close"].rolling(55, min_periods=20).min()
            g["high_10d"] = g["close"].rolling(10, min_periods=5).max()
            g["high_20d"] = g["close"].rolling(20, min_periods=10).max()

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

            # 均线相关
            if "ma5" not in g.columns:
                g["ma5"] = g["close"].rolling(5, min_periods=3).mean()
            if "ma10" not in g.columns:
                g["ma10"] = g["close"].rolling(10, min_periods=5).mean()
            g["ma_34d"] = g["close"].rolling(34, min_periods=10).mean()
            g["ma_55d"] = g["close"].rolling(55, min_periods=20).mean()

            g["price_vs_ma_8d"] = (
                (g["close"] - g["close"].rolling(8, min_periods=3).mean())
                / g["close"].rolling(8, min_periods=3).mean()
                * 100
            )
            g["price_vs_ma_34d"] = (g["close"] - g["ma_34d"]) / g["ma_34d"] * 100
            g["price_vs_ma_55d"] = (g["close"] - g["ma_55d"]) / g["ma_55d"] * 100

            # 趋势斜率
            g["trend_slope_8d"] = g["close"].diff(8) / g["close"].shift(8) * 100
            g["trend_slope_34d"] = g["close"].diff(34) / g["close"].shift(34) * 100
            g["trend_slope_55d"] = g["close"].diff(55) / g["close"].shift(55) * 100

            # 收益率
            g["return_8d"] = g["close"].pct_change(8) * 100
            g["return_34d"] = g["close"].pct_change(34) * 100
            g["return_55d"] = g["close"].pct_change(55) * 100

        # KDJ指标
        if "close" in g.columns:
            low_9 = g["close"].rolling(9, min_periods=5).min()
            high_9 = g["close"].rolling(9, min_periods=5).max()
            rsv = np.where(high_9 > low_9, (g["close"] - low_9) / (high_9 - low_9) * 100, 50)
            g["kdj_k"] = pd.Series(rsv).ewm(com=2, adjust=False).mean().values
            g["kdj_d"] = pd.Series(g["kdj_k"]).ewm(com=2, adjust=False).mean().values
            g["kdj_j"] = 3 * g["kdj_k"] - 2 * g["kdj_d"]

        # 突破特征
        if "close" in g.columns and "high_10d" in g.columns:
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

        g["support_strength_10d"] = np.where(
            g["support_10d"] > 0, (g["close"] - g["support_10d"]) / g["support_10d"] * 100, 0
        )
        g["resistance_strength_10d"] = np.where(
            g["resistance_10d"] > 0, (g["resistance_10d"] - g["close"]) / g["resistance_10d"] * 100, 0
        )
        g["support_strength_20d"] = np.where(
            g["support_20d"] > 0, (g["close"] - g["support_20d"]) / g["support_20d"] * 100, 0
        )
        g["resistance_strength_20d"] = np.where(
            g["resistance_20d"] > 0, (g["resistance_20d"] - g["close"]) / g["resistance_20d"] * 100, 0
        )

        return g

    if "sample_id" in df.columns:
        result = df.groupby("sample_id", group_keys=False).apply(calc_sample_features)
    else:
        result = calc_sample_features(df)

    return result


# ==================== 突破特征计算 ====================


def calculate_breakout_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算突破强度特征"""
    df = df.copy()

    # 1. 突破强度（连续值）
    if "close" in df.columns and "high_10d" in df.columns:
        df["breakout_strength_10d"] = np.where(
            df["high_10d"] > 0, (df["close"] - df["high_10d"]) / df["high_10d"] * 100, 0
        )
    else:
        df["breakout_strength_10d"] = 0

    if "close" in df.columns and "high_20d" in df.columns:
        df["breakout_strength_20d"] = np.where(
            df["high_20d"] > 0, (df["close"] - df["high_20d"]) / df["high_20d"] * 100, 0
        )
    else:
        df["breakout_strength_20d"] = 0

    if "close" in df.columns and "high_55d" in df.columns:
        df["breakout_strength_55d"] = np.where(
            df["high_55d"] > 0, (df["close"] - df["high_55d"]) / df["high_55d"] * 100, 0
        )
    else:
        df["breakout_strength_55d"] = 0

    # 2. 放量突破强度
    if "vol" in df.columns and "vol_mean" in df.columns:
        df["breakout_volume_strength"] = np.where(df["vol_mean"] > 0, df["vol"] / df["vol_mean"], 1)
    elif "volume_ratio" in df.columns:
        df["breakout_volume_strength"] = df["volume_ratio"].fillna(1)
    else:
        df["breakout_volume_strength"] = 1

    # 3. 突破确认（简化计算）
    df["breakout_confirmed_10d"] = np.where(df["breakout_strength_10d"] > 0, 1, 0)
    df["breakout_confirmed_20d"] = np.where(df["breakout_strength_20d"] > 0, 1, 0)

    # 4. 多周期共振
    df["breakout_resonance"] = (
        (df["breakout_strength_10d"] > 0).astype(int)
        + (df["breakout_strength_20d"] > 0).astype(int)
        + (df["breakout_strength_55d"] > 0).astype(int)
    )

    return df


# ==================== 市场环境特征计算 ====================


def get_market_data(dm: DataManager, start_date: str, end_date: str) -> pd.DataFrame:
    """获取大盘数据（上证指数）"""
    try:
        df_market = dm.get_index_daily("000001.SH", start_date, end_date)
        if df_market.empty:
            return pd.DataFrame()

        df_market = df_market.sort_values("trade_date")
        df_market["market_pct_chg"] = df_market["pct_chg"]
        df_market["market_return_34d"] = df_market["close"].pct_change(34) * 100
        df_market["market_volatility_34d"] = df_market["pct_chg"].rolling(34).std()
        df_market["market_trend"] = (
            np.where(df_market["ma5"] > df_market["ma20"], 1, -1)
            if "ma5" in df_market.columns and "ma20" in df_market.columns
            else 0
        )

        # 动量特征
        df_market["market_momentum_5d"] = df_market["close"].pct_change(5) * 100
        df_market["market_momentum_10d"] = df_market["close"].pct_change(10) * 100
        df_market["market_momentum_20d"] = df_market["close"].pct_change(20) * 100

        # 市场状态
        df_market["market_regime"] = np.where(
            df_market["market_momentum_20d"] > 5, 1, np.where(df_market["market_momentum_20d"] < -5, -1, 0)
        )

        # 市场位置
        roll_min = df_market["close"].rolling(20).min()
        roll_max = df_market["close"].rolling(20).max()
        df_market["market_position_20d"] = np.where(
            roll_max > roll_min, (df_market["close"] - roll_min) / (roll_max - roll_min), 0.5
        )

        return df_market[
            [
                "trade_date",
                "market_pct_chg",
                "market_return_34d",
                "market_volatility_34d",
                "market_trend",
                "market_momentum_5d",
                "market_momentum_10d",
                "market_momentum_20d",
                "market_regime",
                "market_position_20d",
            ]
        ]
    except Exception as e:
        log.warning(f"获取市场数据失败: {e}")
        return pd.DataFrame()


def add_market_features(df: pd.DataFrame, df_market: pd.DataFrame) -> pd.DataFrame:
    """添加市场环境特征"""
    df = df.copy()

    market_cols = [
        "market_pct_chg",
        "market_return_34d",
        "market_volatility_34d",
        "market_trend",
        "market_momentum_5d",
        "market_momentum_10d",
        "market_momentum_20d",
        "market_regime",
        "market_position_20d",
    ]

    # 检查是否已有市场特征
    existing_market_cols = [c for c in market_cols if c in df.columns]
    if len(existing_market_cols) >= 5:
        # 已有市场特征，只补充缺失的
        for col in market_cols:
            if col not in df.columns:
                df[col] = 0
        if "excess_return" not in df.columns:
            df["excess_return"] = df.get("pct_chg", 0) - df.get("market_pct_chg", 0)
        if "excess_return_cumsum" not in df.columns:
            if "sample_id" in df.columns:
                df["excess_return_cumsum"] = df.groupby("sample_id")["excess_return"].cumsum()
            else:
                df["excess_return_cumsum"] = df["excess_return"].cumsum()
        if "excess_return_consistency" not in df.columns:
            df["excess_return_consistency"] = np.where(df["excess_return"] > 0, 1, 0)
        return df

    if df_market.empty:
        # 添加空特征
        for col in market_cols:
            df[col] = 0
        df["excess_return"] = df.get("pct_chg", 0)
        df["excess_return_cumsum"] = 0
        df["excess_return_consistency"] = 0
        return df

    # 合并市场数据
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df_market["trade_date"] = pd.to_datetime(df_market["trade_date"])

    df = df.merge(df_market, on="trade_date", how="left")

    # 填充缺失值
    market_cols = [
        "market_pct_chg",
        "market_return_34d",
        "market_volatility_34d",
        "market_trend",
        "market_momentum_5d",
        "market_momentum_10d",
        "market_momentum_20d",
        "market_regime",
        "market_position_20d",
    ]
    for col in market_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    # 超额收益
    if "pct_chg" in df.columns:
        df["excess_return"] = df["pct_chg"] - df["market_pct_chg"]
    else:
        df["excess_return"] = 0

    # 累计超额收益（按样本分组）
    if "sample_id" in df.columns:
        df["excess_return_cumsum"] = df.groupby("sample_id")["excess_return"].cumsum()
    else:
        df["excess_return_cumsum"] = df["excess_return"].cumsum()

    # 超额收益一致性
    df["excess_return_consistency"] = np.where(df["excess_return"] > 0, 1, 0)

    return df


# ==================== 交互特征计算 ====================


def calculate_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算交互特征"""
    df = df.copy()

    # 1. 突破+放量
    breakout_signal = (df.get("breakout_strength_10d", 0) > 0) | (df.get("breakout_strength_20d", 0) > 0)
    volume_signal = df.get("breakout_volume_strength", 1) > 1.5
    df["breakout_with_volume"] = (breakout_signal & volume_signal).astype(int)

    # 2. 动量+市场环境
    momentum = df.get("momentum_20d", df.get("pct_chg", 0))
    market_trend = df.get("market_trend", 0)
    df["momentum_market_interaction"] = momentum * market_trend

    # 3. RSI+KDJ背离
    rsi = df.get("rsi_6", 50)
    kdj_j = df.get("kdj_j", 50)
    df["rsi_kdj_divergence"] = np.abs(rsi - kdj_j)

    # 4. 趋势一致性
    ma5 = df.get("ma5", df.get("close", 0))
    ma10 = df.get("ma10", ma5)
    df["trend_consistency"] = np.where(
        (ma5 > ma10) & (df.get("pct_chg", 0) > 0), 1, np.where((ma5 < ma10) & (df.get("pct_chg", 0) < 0), 1, 0)
    )

    # 5. 量价背离
    pct_chg = df.get("pct_chg", pd.Series([0] * len(df)))
    if "vol" in df.columns:
        vol_chg = df["vol"] / df["vol"].shift(1).fillna(1) - 1
    else:
        vol_chg = pd.Series([0] * len(df))
    df["volume_price_divergence"] = np.where(
        (pct_chg > 0) & (vol_chg < -0.2), 1, np.where((pct_chg < 0) & (vol_chg > 0.2), -1, 0)
    )

    # 6. 突破+RSI
    breakout = df.get("breakout_strength_20d", 0)
    df["breakout_rsi_interaction"] = breakout * (100 - rsi) / 100

    # 7. 相对波动率
    vol_34d = df.get("volatility_34d", df.get("pct_chg", 0).rolling(34).std())
    market_vol = df.get("market_volatility_34d", 1)
    df["relative_volatility"] = np.where(market_vol > 0, vol_34d / market_vol, 1)

    # 8. 共振+成交量确认
    resonance = df.get("breakout_resonance", 0)
    vol_confirm = (df.get("breakout_volume_strength", 1) > 1.2).astype(int)
    df["resonance_volume_confirm"] = resonance * vol_confirm

    return df


# ==================== 主函数 ====================


def process_file(file_path: Path, dm: DataManager, df_market: pd.DataFrame) -> int:
    """处理单个文件"""
    log.info(f"\n处理文件: {file_path.name}")

    # 加载数据
    df = pd.read_csv(file_path)
    original_cols = len(df.columns)
    log.info(f"  原始特征数: {original_cols}")

    # 0. 计算基础技术特征（如果特征数少于100，说明需要补充）
    if original_cols < 100:
        log.info("  计算基础技术特征...")
        df = calculate_basic_features(df)
        log.info(f"  基础特征后: {len(df.columns)}")

    # 1. 添加突破特征
    df = calculate_breakout_features(df)

    # 2. 添加市场环境特征
    df = add_market_features(df, df_market)

    # 3. 添加交互特征
    df = calculate_interaction_features(df)

    # 保存
    df.to_csv(file_path, index=False)
    final_cols = len(df.columns)
    log.info(f"  最终特征数: {final_cols} (新增 {final_cols - original_cols} 个)")

    return final_cols


def main():
    log.info("=" * 80)
    log.info("v6 样本特征工程")
    log.info("=" * 80)

    # 文件路径
    pos_file = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v6.csv"
    neg_file = PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v6.csv"
    hard_file = PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v6.csv"

    # 检查文件
    for f in [pos_file, neg_file, hard_file]:
        if not f.exists():
            log.error(f"文件不存在: {f}")
            return

    # 初始化数据管理器
    dm = DataManager()

    # 获取市场数据（覆盖所有可能的日期范围）
    log.info("\n获取市场数据...")
    df_market = get_market_data(dm, "19990101", "20261231")
    log.info(f"  市场数据: {len(df_market)} 条")

    # 处理正样本
    log.info("\n" + "=" * 50)
    log.info("[1/3] 处理正样本")
    process_file(pos_file, dm, df_market)

    # 处理负样本
    log.info("\n" + "=" * 50)
    log.info("[2/3] 处理负样本")
    process_file(neg_file, dm, df_market)

    # 处理硬负样本
    log.info("\n" + "=" * 50)
    log.info("[3/3] 处理硬负样本")
    process_file(hard_file, dm, df_market)

    log.info("\n" + "=" * 80)
    log.success("✅ v6 特征工程完成！")
    log.info("=" * 80)
    log.info("\n下一步: python scripts/train_v250_model.py")


if __name__ == "__main__":
    main()
