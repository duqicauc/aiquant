#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.3.3模型预测脚本 - 板块轮动优化版

相比v2.3.2的改进：
1. 添加板块热度分析：识别过热板块（如商业航天连涨1个月）
2. 板块轮动策略：降低过热板块权重，提升启动中和潜力板块权重
3. 市场环境感知：根据大盘走势调整推荐策略
4. 执行效率优化：批量获取板块数据，缓存计算结果
"""

import sys
import json
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.logger import log
from src.data.data_manager import DataManager


def load_model():
    """加载v2.3.0模型"""
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / "v2.3.0" / "model"

    booster = xgb.Booster()
    booster.load_model(str(model_dir / "model.json"))

    with open(model_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    calibrator = joblib.load(str(model_dir / "calibrator.pkl"))

    return booster, feature_names, calibrator


def analyze_market_environment(dm, predict_date):
    """
    分析市场环境

    返回：
    - market_trend: 'strong_up', 'weak_up', 'sideways', 'weak_down', 'strong_down'
    - market_momentum: 市场动量分数 [0-1]
    """
    try:
        # 获取上证指数近30天数据
        end_date = predict_date
        start_date = (datetime.strptime(predict_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")

        df_index = dm.get_daily_data("000001.SH", start_date, end_date)
        if df_index is None or len(df_index) < 20:
            return "sideways", 0.5

        df_index = df_index.sort_values("trade_date").reset_index(drop=True)

        # 计算指标
        returns_5d = df_index["close"].pct_change(5).iloc[-1] * 100
        returns_10d = df_index["close"].pct_change(10).iloc[-1] * 100
        returns_20d = df_index["close"].pct_change(20).iloc[-1] * 100

        # 成交量变化
        vol_ratio = df_index["vol"].iloc[-5:].mean() / df_index["vol"].iloc[-20:-5].mean()

        # 趋势判断
        if returns_5d > 3 and returns_10d > 5:
            trend = "strong_up"
            momentum = 0.8
        elif returns_5d > 0 and returns_10d > 0:
            trend = "weak_up"
            momentum = 0.6
        elif returns_5d < -3 and returns_10d < -5:
            trend = "strong_down"
            momentum = 0.2
        elif returns_5d < 0 and returns_10d < 0:
            trend = "weak_down"
            momentum = 0.4
        else:
            trend = "sideways"
            momentum = 0.5

        # 成交量放大加分
        if vol_ratio > 1.2:
            momentum = min(1.0, momentum + 0.1)

        log.info(
            f"📊 市场环境: {trend}, 动量: {momentum:.2f}, 5日涨幅: {returns_5d:.2f}%, 10日涨幅: {returns_10d:.2f}%, 量比: {vol_ratio:.2f}"
        )

        return trend, momentum

    except Exception as e:
        log.warning(f"市场环境分析失败: {e}")
        return "sideways", 0.5


def analyze_sector_heat(dm, stock_list, predict_date):
    """
    分析板块热度

    返回: dict {行业: 热度分数}
    - 热度分数 > 0.8: 过热板块（降权）
    - 热度分数 0.5-0.8: 启动中板块（加权）
    - 热度分数 < 0.5: 冷门/潜力板块（适当加权）
    """
    try:
        log.info("\n🔥 分析板块热度...")

        # 获取行业分类
        sector_stocks = defaultdict(list)
        for _, row in stock_list.iterrows():
            industry = row.get("industry", "未知")
            if industry and industry != "未知":
                sector_stocks[industry].append(row["ts_code"])

        # 计算每个板块的热度
        sector_heat = {}
        end_date = predict_date
        start_date = (datetime.strptime(predict_date, "%Y%m%d") - timedelta(days=60)).strftime("%Y%m%d")

        for industry, stocks in sector_stocks.items():
            if len(stocks) < 3:  # 板块股票数太少，跳过
                continue

            # 随机抽样最多3只股票来代表该板块（提高效率）
            sample_stocks = np.random.choice(stocks, min(3, len(stocks)), replace=False)

            returns_5d_list = []
            returns_10d_list = []
            returns_20d_list = []
            vol_ratio_list = []

            for ts_code in sample_stocks:
                try:
                    df = dm.get_daily_data(ts_code, start_date, end_date)
                    if df is None or len(df) < 25:
                        continue

                    df = df.sort_values("trade_date").reset_index(drop=True)

                    # 计算收益率
                    ret_5d = df["close"].pct_change(5).iloc[-1] * 100
                    ret_10d = df["close"].pct_change(10).iloc[-1] * 100
                    ret_20d = df["close"].pct_change(20).iloc[-1] * 100

                    # 成交量放大
                    vol_ratio = df["vol"].iloc[-5:].mean() / (df["vol"].iloc[-20:-5].mean() + 1e-8)

                    if pd.notna(ret_5d):
                        returns_5d_list.append(ret_5d)
                    if pd.notna(ret_10d):
                        returns_10d_list.append(ret_10d)
                    if pd.notna(ret_20d):
                        returns_20d_list.append(ret_20d)
                    if pd.notna(vol_ratio):
                        vol_ratio_list.append(vol_ratio)

                except:
                    continue

            if len(returns_20d_list) < 3:
                continue

            # 计算板块平均指标
            avg_ret_5d = np.mean(returns_5d_list)
            avg_ret_10d = np.mean(returns_10d_list)
            avg_ret_20d = np.mean(returns_20d_list)
            avg_vol_ratio = np.mean(vol_ratio_list)

            # 热度评分逻辑：
            # 1. 20日涨幅过大（>30%）且10日仍在涨（>10%）-> 过热（0.9-1.0）
            # 2. 20日涨幅较大（15-30%）且5日仍在涨 -> 强势但接近过热（0.7-0.9）
            # 3. 20日涨幅适中（5-15%）且放量 -> 启动中（0.5-0.7）
            # 4. 20日涨幅小或负 -> 冷门（0.2-0.5）

            heat_score = 0.5  # 基础分

            # 根据20日涨幅调整
            if avg_ret_20d > 30:
                heat_score = 0.95
            elif avg_ret_20d > 20:
                heat_score = 0.85
            elif avg_ret_20d > 10:
                heat_score = 0.70
            elif avg_ret_20d > 5:
                heat_score = 0.55
            else:
                heat_score = 0.40

            # 根据近期趋势微调
            if avg_ret_10d > 15 and avg_ret_20d > 25:
                heat_score = min(1.0, heat_score + 0.05)  # 持续强势，更过热

            if avg_ret_5d > 5 and avg_ret_10d < 3:
                heat_score = max(0.5, heat_score - 0.1)  # 刚启动，降低过热度

            # 成交量放大的板块，如果涨幅不大，是启动信号
            if avg_vol_ratio > 1.3 and avg_ret_20d < 15:
                heat_score = max(0.5, heat_score - 0.1)

            sector_heat[industry] = {
                "heat_score": heat_score,
                "return_5d": avg_ret_5d,
                "return_10d": avg_ret_10d,
                "return_20d": avg_ret_20d,
                "vol_ratio": avg_vol_ratio,
                "stock_count": len(stocks),
            }

        # 输出热度分析结果（Top10过热和Top10冷门）
        sorted_sectors = sorted(sector_heat.items(), key=lambda x: x[1]["heat_score"], reverse=True)

        log.info("\n🔥 过热板块 Top10（降权）:")
        log.info(f"{'行业':<15} {'热度':<8} {'20日涨幅':<10} {'10日涨幅':<10} {'5日涨幅':<10} {'量比':<8}")
        log.info("-" * 75)
        for industry, data in sorted_sectors[:10]:
            log.info(
                f"{industry:<15} {data['heat_score']:<8.2f} {data['return_20d']:>9.2f}% {data['return_10d']:>9.2f}% {data['return_5d']:>9.2f}% {data['vol_ratio']:<8.2f}"
            )

        log.info("\n❄️  冷门/潜力板块 Top10（加权）:")
        log.info(f"{'行业':<15} {'热度':<8} {'20日涨幅':<10} {'10日涨幅':<10} {'5日涨幅':<10} {'量比':<8}")
        log.info("-" * 75)
        for industry, data in sorted_sectors[-10:]:
            log.info(
                f"{industry:<15} {data['heat_score']:<8.2f} {data['return_20d']:>9.2f}% {data['return_10d']:>9.2f}% {data['return_5d']:>9.2f}% {data['vol_ratio']:<8.2f}"
            )

        return sector_heat

    except Exception as e:
        log.error(f"板块热度分析失败: {e}")
        return {}


def calculate_sector_adjustment(industry, sector_heat, market_trend):
    """
    根据板块热度和市场环境计算调整系数

    返回: (调整系数, 原因说明)
    """
    if not sector_heat or industry not in sector_heat:
        return 1.0, ""

    heat_data = sector_heat[industry]
    heat_score = heat_data["heat_score"]

    # 基础调整逻辑
    if heat_score > 0.85:
        # 过热板块，显著降权
        adjustment = 0.70
        reason = f"过热板块({heat_data['return_20d']:.1f}%/20D)"
    elif heat_score > 0.75:
        # 强势板块，轻微降权
        adjustment = 0.85
        reason = f"强势板块({heat_data['return_20d']:.1f}%/20D)"
    elif heat_score > 0.60:
        # 启动中板块，保持或轻微加权
        adjustment = 1.05
        reason = f"启动中({heat_data['return_20d']:.1f}%/20D)"
    elif heat_score < 0.45 and heat_data["vol_ratio"] > 1.2:
        # 冷门但开始放量，可能是轮动机会
        adjustment = 1.10
        reason = f"潜力板块(量比{heat_data['vol_ratio']:.2f})"
    else:
        # 普通板块
        adjustment = 1.0
        reason = ""

    # 市场环境调整
    if market_trend in ["strong_down", "weak_down"]:
        # 市场弱势时，降低对过热板块的追捧，增加对防御性板块的权重
        if heat_score > 0.8:
            adjustment *= 0.9  # 进一步降权过热板块

    return adjustment, reason


def extract_features(df):
    """
    提取特征（与v2.3.2相同）
    """
    df = df.copy()

    # ========== 基础均线 ==========
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma_20d"] = df["close"].rolling(20).mean()

    # ========== MACD ==========
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_dif"] = df["ema12"] - df["ema26"]
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    df["macd"] = 2 * (df["macd_dif"] - df["macd_dea"])

    # ========== RSI ==========
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    df["rsi_6"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    gain12 = delta.where(delta > 0, 0).rolling(12).mean()
    loss12 = (-delta.where(delta < 0, 0)).rolling(12).mean()
    df["rsi_12"] = 100 - (100 / (1 + gain12 / (loss12 + 1e-10)))

    gain24 = delta.where(delta > 0, 0).rolling(24).mean()
    loss24 = (-delta.where(delta < 0, 0)).rolling(24).mean()
    df["rsi_24"] = 100 - (100 / (1 + gain24 / (loss24 + 1e-10)))

    # ========== KDJ ==========
    low_9 = df["low"].rolling(9).min()
    high_9 = df["high"].rolling(9).max()
    rsv = (df["close"] - low_9) / (high_9 - low_9 + 1e-10) * 100
    df["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(com=2, adjust=False).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    # ========== 量比 ==========
    df["volume_ratio"] = df["vol"] / (df["vol"].rolling(5).mean() + 1e-8)

    # ========== 多周期特征 ==========
    for period in [8, 34, 55]:
        df[f"return_{period}d"] = df["close"].pct_change(period) * 100
        df[f"ma_{period}d"] = df["close"].rolling(period).mean()
        df[f"price_vs_ma_{period}d"] = (df["close"] - df[f"ma_{period}d"]) / df[f"ma_{period}d"] * 100
        df[f"volatility_{period}d"] = df["pct_chg"].rolling(period).std()
        df[f"high_{period}d"] = df["high"].rolling(period).max()
        df[f"low_{period}d"] = df["low"].rolling(period).min()
        price_range = df[f"high_{period}d"] - df[f"low_{period}d"]
        df[f"price_position_{period}d"] = (df["close"] - df[f"low_{period}d"]) / (price_range + 1e-10)
        df[f"trend_slope_{period}d"] = (
            df["close"]
            .rolling(period)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else 0, raw=False)
        )

    # ========== 动量 ==========
    df["momentum_5d"] = df["close"].pct_change(5) * 100
    df["momentum_10d"] = df["close"].pct_change(10) * 100
    df["momentum_20d"] = df["close"].pct_change(20) * 100
    df["momentum_acceleration"] = df["momentum_5d"] - df["momentum_5d"].shift(5)

    # ========== 价量关系 ==========
    df["price_change"] = df["close"].diff()
    df["volume_change"] = df["vol"].diff()
    df["volume_price_corr_10d"] = df["close"].rolling(10).corr(df["vol"])
    df["volume_price_corr_20d"] = df["close"].rolling(20).corr(df["vol"])
    df["volume_price_match"] = ((df["price_change"] > 0) & (df["volume_change"] > 0)).astype(int)
    df["volume_price_match_sum_10d"] = df["volume_price_match"].rolling(10).sum()

    # ========== 突破特征 ==========
    for period in [10, 20, 55]:
        df[f"prev_high_{period}d"] = df["high"].rolling(period).max().shift(1)
        df[f"breakout_high_{period}d"] = (df["close"] > df[f"prev_high_{period}d"]).astype(int)
        df[f"resistance_{period}d"] = df["high"].rolling(period).max()
        df[f"support_{period}d"] = df["low"].rolling(period).min()
        df[f"dist_to_resistance_{period}d"] = (df[f"resistance_{period}d"] - df["close"]) / df["close"] * 100
        df[f"dist_to_support_{period}d"] = (df["close"] - df[f"support_{period}d"]) / df["close"] * 100
        df[f"support_strength_{period}d"] = (df["low"] - df[f"support_{period}d"]).abs().rolling(period).mean()
        df[f"resistance_strength_{period}d"] = (df[f"resistance_{period}d"] - df["high"]).abs().rolling(period).mean()

    df["channel_width_20d"] = (df["resistance_20d"] - df["support_20d"]) / df["close"] * 100

    # ========== MA突破 ==========
    df["ma_5d"] = df["close"].rolling(5).mean()
    df["breakout_ma5"] = (df["close"] > df["ma_5d"]).astype(int)
    df["ma_10d"] = df["close"].rolling(10).mean()
    df["breakout_ma10"] = (df["close"] > df["ma_10d"]).astype(int)
    df["breakout_ma20"] = (df["close"] > df["ma_20d"]).astype(int)
    ma_55d = df["close"].rolling(55).mean()
    df["breakout_ma55"] = (df["close"] > ma_55d).astype(int)

    df["breakout_volume_ratio"] = df["vol"] / (df["vol"].rolling(20).mean() + 1e-8)
    df["high_volume_breakout"] = ((df["breakout_high_20d"] == 1) & (df["breakout_volume_ratio"] > 1.5)).astype(int)
    df["consecutive_new_high"] = df["breakout_high_10d"].rolling(5).sum()

    # ========== 成交量趋势 ==========
    df["volume_trend_slope_10d"] = (
        df["vol"].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=False)
    )
    df["volume_trend_slope_20d"] = (
        df["vol"].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0, raw=False)
    )
    df["volume_breakout_count_20d"] = (df["vol"] > df["vol"].rolling(20).mean() * 1.5).rolling(20).sum()

    # ========== 量价背离 ==========
    df["price_up_vol_down"] = ((df["price_change"] > 0) & (df["volume_change"] < 0)).astype(int)
    df["price_up_vol_down_count_10d"] = df["price_up_vol_down"].rolling(10).sum()
    df["price_down_vol_up"] = ((df["price_change"] < 0) & (df["volume_change"] > 0)).astype(int)
    df["price_down_vol_up_count_10d"] = df["price_down_vol_up"].rolling(10).sum()

    # ========== OBV ==========
    df["obv"] = (np.sign(df["close"].diff()) * df["vol"]).fillna(0).cumsum()
    df["obv_calc"] = df["obv"]
    df["obv_ma10"] = df["obv"].rolling(10).mean()
    df["obv_trend"] = (df["obv"] > df["obv_ma10"]).astype(int)

    # ========== 成交量RSV ==========
    vol_low_20 = df["vol"].rolling(20).min()
    vol_high_20 = df["vol"].rolling(20).max()
    df["volume_rsv_20d"] = (df["vol"] - vol_low_20) / (vol_high_20 - vol_low_20 + 1e-10) * 100

    # ========== 乖离率 ==========
    df["bias_short"] = (df["close"] - df["ma5"]) / df["ma5"] * 100
    df["bias_mid"] = (df["close"] - df["ma10"]) / df["ma10"] * 100
    df["bias_long"] = (df["close"] - df["ma_20d"]) / df["ma_20d"] * 100

    # ========== EMA ==========
    df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_60"] = df["close"].ewm(span=60, adjust=False).mean()

    # ========== 量比 ==========
    df["vol_ma5_ratio"] = df["vol"] / (df["vol"].rolling(5).mean() + 1e-8)
    df["vol_ma20_ratio"] = df["vol"] / (df["vol"].rolling(20).mean() + 1e-8)

    # ========== 涨停 ==========
    df["is_limit_up"] = (df["pct_chg"] >= 9.8).astype(int)

    # ========== 历史位置 ==========
    df["price_vs_hist_mean"] = (df["close"] - df["close"].rolling(34).mean()) / df["close"].rolling(34).mean() * 100
    df["price_vs_hist_high"] = (df["close"] - df["close"].rolling(34).max()) / df["close"].rolling(34).max() * 100
    df["volatility_vs_hist"] = df["pct_chg"].rolling(10).std() / (df["pct_chg"].rolling(34).std() + 1e-8)

    # ========== 市场相关（占位） ==========
    df["market_pct_chg"] = 0
    df["market_return_34d"] = 0
    df["market_volatility_34d"] = 0
    df["market_trend"] = 0
    df["excess_return"] = df["pct_chg"]
    df["excess_return_cumsum"] = df["pct_chg"].rolling(34).sum()

    # ========== 风险特征 ==========
    for period in [10, 20, 55]:
        rolling_max = df["close"].rolling(period, min_periods=1).max()
        drawdown = (df["close"] - rolling_max) / rolling_max * 100
        df[f"max_drawdown_{period}d"] = drawdown.rolling(period, min_periods=1).min()

    # ATR
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - prev_close)
    tr3 = abs(df["low"] - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["atr_14"] = true_range.rolling(14, min_periods=1).mean()
    df["atr_ratio_14"] = df["atr_14"] / df["close"] * 100
    atr_mean = df["atr_14"].rolling(55, min_periods=14).mean()
    df["atr_expansion"] = df["atr_14"] / (atr_mean + 1e-10)

    # 距高点天数
    for period in [20, 55]:
        rolling_high = df["close"].rolling(period, min_periods=1).max()
        is_at_high = df["close"] == rolling_high
        days_list = []
        days_since_high = 0
        for is_high in is_at_high:
            if is_high:
                days_since_high = 0
            else:
                days_since_high += 1
            days_list.append(days_since_high)
        df[f"days_from_high_{period}d"] = days_list

    # 恢复比例
    rolling_low_20 = df["close"].rolling(20, min_periods=1).min()
    rolling_high_20 = df["close"].rolling(20, min_periods=1).max()
    price_range = rolling_high_20 - rolling_low_20
    df["recovery_ratio_20d"] = (df["close"] - rolling_low_20) / (price_range + 1e-10)

    # ========== 收益预测特征 ==========
    df["momentum_strength"] = df["momentum_5d"] * 0.3 + df["momentum_10d"] * 0.4 + df["momentum_20d"] * 0.3

    breakout_count = (
        df["breakout_high_10d"].astype(int)
        + df["breakout_high_20d"].astype(int)
        + df["breakout_high_55d"].astype(int)
        + df["breakout_ma5"].astype(int)
        + df["breakout_ma10"].astype(int)
        + df["breakout_ma20"].astype(int)
        + df["breakout_ma55"].astype(int)
    )
    df["breakout_strength"] = breakout_count / 7.0

    vol_ma20 = df["vol"].rolling(20, min_periods=1).mean()
    df["volume_expansion_ratio"] = df["vol"] / (vol_ma20 + 1e-8)
    df["volume_expansion_ratio"] = df["volume_expansion_ratio"].clip(upper=10.0)

    high_20 = df["high"].rolling(20, min_periods=1).max()
    low_20 = df["low"].rolling(20, min_periods=1).min()
    price_range_20 = high_20 - low_20
    df["price_position_score"] = (df["close"] - low_20) / (price_range_20 + 1e-10)

    momentum_norm = (df["momentum_strength"] / 50.0).clip(0, 1)
    volume_norm = (df["volume_expansion_ratio"] / 2.0).clip(0, 1)
    price_vol_match = df["volume_price_match_sum_10d"] / 10.0

    df["expected_return_score"] = (
        momentum_norm * 0.3
        + df["breakout_strength"] * 0.25
        + volume_norm * 0.2
        + df["price_position_score"] * 0.15
        + price_vol_match * 0.1
    )

    # ========== v2.3.2新增：连续涨停天数 ==========
    df["consecutive_limit_up"] = df["is_limit_up"].rolling(3, min_periods=1).sum()

    return df


def calculate_v233_score(
    cal_prob, expected_return_score, pct_chg, rsi_6, amount, consecutive_limit_up, industry, sector_heat, market_trend
):
    """
    v2.3.3评分公式（在v2.3.2基础上增加板块调整）

    改进：
    1. 保留v2.3.2的所有惩罚机制
    2. 新增板块热度调整：过热板块降权，启动中/潜力板块加权
    """
    # v2.3.2基础评分
    base_score = 0.6 * cal_prob + 0.4 * expected_return_score

    penalty = 1.0
    penalty_reasons = []

    # v2.3.2的惩罚机制
    if pct_chg > 15:
        penalty *= 0.5
        penalty_reasons.append(f"追高惩罚(涨幅{pct_chg:.1f}%)")
    elif pct_chg > 10:
        penalty *= 0.8
        penalty_reasons.append(f"轻度追高(涨幅{pct_chg:.1f}%)")

    if pct_chg >= 9.8 and cal_prob < 0.8:
        penalty *= 0.7
        penalty_reasons.append(f"涨停低概率({cal_prob:.2f})")

    if rsi_6 > 95:
        penalty *= 0.8
        penalty_reasons.append(f"RSI过热({rsi_6:.1f})")
    elif rsi_6 > 90:
        penalty *= 0.9
        penalty_reasons.append(f"RSI偏高({rsi_6:.1f})")

    if consecutive_limit_up >= 3:
        penalty *= 0.6
        penalty_reasons.append(f"连续涨停({consecutive_limit_up}天)")
    elif consecutive_limit_up >= 2:
        penalty *= 0.8
        penalty_reasons.append(f"连续涨停({consecutive_limit_up}天)")

    # v2.3.3新增：板块调整
    sector_adjustment, sector_reason = calculate_sector_adjustment(industry, sector_heat, market_trend)
    if sector_reason:
        penalty_reasons.append(sector_reason)

    final_score = base_score * penalty * sector_adjustment

    return final_score, penalty * sector_adjustment, penalty_reasons


def process_single_stock(
    dm, ts_code, name, industry, predict_date, feature_names, booster, calibrator, sector_heat, market_trend
):
    """处理单只股票"""
    try:
        end_date = predict_date
        start_date = (datetime.strptime(predict_date, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")

        df = dm.get_daily_data(ts_code, start_date, end_date)
        if df is None or len(df) < 60:
            return None

        df = df.sort_values("trade_date").reset_index(drop=True)

        # 提取特征
        df = extract_features(df)
        last_row = df.iloc[-1]

        # 构建特征向量
        feature_vector = []
        for fn in feature_names:
            val = last_row.get(fn, 0)
            if pd.isna(val) or not np.isfinite(val):
                val = 0
            feature_vector.append(float(val))

        # 预测
        dmatrix = xgb.DMatrix([feature_vector], feature_names=feature_names)
        raw_prob = float(booster.predict(dmatrix)[0])
        cal_prob = float(calibrator.predict([raw_prob])[0])

        # 获取关键指标
        expected_return_score = last_row.get("expected_return_score", 0.5)
        if pd.isna(expected_return_score) or not np.isfinite(expected_return_score):
            expected_return_score = 0.5
        expected_return_norm = float(np.clip(expected_return_score, 0, 1))

        pct_chg = float(last_row.get("pct_chg", 0))
        rsi_6 = float(last_row.get("rsi_6", 50))
        amount = float(last_row.get("amount", 0))
        consecutive_limit_up = float(last_row.get("consecutive_limit_up", 0))

        # v2.3.3评分（加入板块因子）
        final_score, penalty, penalty_reasons = calculate_v233_score(
            cal_prob,
            expected_return_norm,
            pct_chg,
            rsi_6,
            amount,
            consecutive_limit_up,
            industry,
            sector_heat,
            market_trend,
        )

        return {
            "ts_code": ts_code,
            "name": name,
            "industry": industry,
            "close": float(last_row["close"]),
            "pct_chg": pct_chg,
            "amount": amount,
            "raw_probability": raw_prob,
            "calibrated_probability": cal_prob,
            "expected_return_score": expected_return_score,
            "final_score": final_score,
            "penalty": penalty,
            "penalty_reasons": "|".join(penalty_reasons) if penalty_reasons else "",
            "return_34d": float(last_row.get("return_34d", 0)),
            "rsi_6": rsi_6,
            "max_drawdown_20d": float(last_row.get("max_drawdown_20d", 0)),
            "atr_ratio_14": float(last_row.get("atr_ratio_14", 0)),
            "momentum_strength": float(last_row.get("momentum_strength", 0)),
            "breakout_strength": float(last_row.get("breakout_strength", 0)),
            "volume_expansion_ratio": float(last_row.get("volume_expansion_ratio", 1.0)),
            "consecutive_limit_up": consecutive_limit_up,
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="v2.3.3模型预测 - 板块轮动优化版")
    parser.add_argument("--date", type=str, default="20260113", help="预测日期 (YYYYMMDD)")
    parser.add_argument("--min-amount", type=float, default=30000, help="最小成交额（千元），默认3000万")
    parser.add_argument("--enable-sector", action="store_true", help="启用板块轮动分析（会增加执行时间）")
    args = parser.parse_args()

    predict_date = args.date
    min_amount = args.min_amount

    log.info("=" * 80)
    log.info(f"v2.3.3模型预测 - 板块轮动优化版 - {predict_date}")
    log.info("=" * 80)
    log.info(f"最小成交额要求: {min_amount/1000:.0f}百万元")

    # 初始化
    dm = DataManager()

    # 加载模型
    log.info("\n📦 加载v2.3.0模型...")
    booster, feature_names, calibrator = load_model()
    log.success(f"✓ 模型加载成功: {len(feature_names)} 特征")

    # 获取股票列表
    stock_list = dm.get_stock_list()
    valid = stock_list[
        ~stock_list["name"].str.contains("ST|退", na=False)
        & ~stock_list["ts_code"].str.startswith("688")
        & ~stock_list["ts_code"].str.startswith("8")
    ].copy()
    log.info(f"📊 有效股票: {len(valid)} 只")

    # 分析市场环境
    market_trend, market_momentum = analyze_market_environment(dm, predict_date)

    # 分析板块热度（可选功能）
    sector_heat = {}
    if args.enable_sector:
        log.info("\n⚠️  启用板块轮动分析（这将增加5-10分钟执行时间）...")
        sector_heat = analyze_sector_heat(dm, valid, predict_date)
    else:
        log.info("\n⚡ 跳过板块轮动分析（使用 --enable-sector 启用）")

    # 批量处理
    log.info("\n🚀 开始预测...")
    results = []
    total = len(valid)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for idx, row in valid.iterrows():
            industry = row.get("industry", "未知")
            future = executor.submit(
                process_single_stock,
                dm,
                row["ts_code"],
                row["name"],
                industry,
                predict_date,
                feature_names,
                booster,
                calibrator,
                sector_heat,
                market_trend,
            )
            futures[future] = (row["ts_code"], row["name"])

        completed = 0
        error_count = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 500 == 0 or completed == total:
                log.info(
                    f"进度: {completed}/{total} ({completed/total*100:.1f}%) | 成功: {len(results)}, 失败: {error_count}"
                )

            result = future.result()
            if result:
                results.append(result)
            else:
                error_count += 1

    if not results:
        log.error("没有预测结果")
        return

    # 转换为DataFrame
    df_results = pd.DataFrame(results)

    # 流动性过滤
    before_filter = len(df_results)
    df_results = df_results[df_results["amount"] >= min_amount]
    log.info(f"流动性过滤: {before_filter} -> {len(df_results)} (过滤掉成交额<{min_amount/1000:.0f}百万的股票)")

    # 按final_score排序
    df_results = df_results.sort_values("final_score", ascending=False).reset_index(drop=True)

    # Top10
    df_top10 = df_results.head(10)

    log.success(f"\n✓ 预测完成: {len(df_results)} 只股票（流动性过滤后）")

    # 显示Top10
    log.info("\n" + "=" * 120)
    log.info("🏆 v2.3.3 Top10 推荐（板块轮动优化版）")
    log.info("=" * 120)
    log.info(
        f"\n{'排名':<4} {'代码':<12} {'名称':<10} {'行业':<12} {'综合评分':<10} {'校准概率':<10} {'当日涨幅':<10} {'调整系数':<10} {'调整原因':<40}"
    )
    log.info("-" * 125)

    for i, (_, row) in enumerate(df_top10.iterrows(), 1):
        penalty_str = f"{row['penalty']:.2f}" if row["penalty"] != 1.0 else "无"
        reasons = row["penalty_reasons"] if row["penalty_reasons"] else "-"
        log.info(
            f"{i:<4} {row['ts_code']:<12} {row['name']:<10} {row['industry']:<12} "
            f"{row['final_score']:<10.4f} {row['calibrated_probability']:<10.4f} "
            f"{row['pct_chg']:>+9.2f}% {penalty_str:<10} {reasons:<40}"
        )

    # 保存结果
    output_dir = PROJECT_ROOT / "data" / "prediction" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"v2.3.3_top10_{predict_date}.csv"
    df_top10.to_csv(output_file, index=False, encoding="utf-8-sig")
    log.success(f"\n💾 Top10结果已保存: {output_file}")

    # 保存完整结果
    full_output_file = output_dir / f"v2.3.3_full_{predict_date}.csv"
    df_results.to_csv(full_output_file, index=False, encoding="utf-8-sig")
    log.info(f"💾 完整结果已保存: {full_output_file}")

    # 统计对比
    log.info("\n" + "=" * 80)
    log.info("📊 v2.3.3 板块分布分析")
    log.info("=" * 80)

    sector_distribution = df_top10["industry"].value_counts()
    log.info("\nTop10板块分布:")
    for industry, count in sector_distribution.items():
        log.info(f"  {industry}: {count}只")

    penalized = df_top10[df_top10["penalty"] < 1.0]
    chase_high = df_top10[df_top10["pct_chg"] > 9]

    log.info("\nTop10统计:")
    log.info(f"  - 被调整的股票: {len(penalized)}/10")
    log.info(f"  - 当日涨幅>9%的股票: {len(chase_high)}/10")
    log.info(f"  - 平均当日涨幅: {df_top10['pct_chg'].mean():.2f}%")
    log.info(f"  - 平均校准概率: {df_top10['calibrated_probability'].mean():.4f}")
    log.info(f"  - 市场环境: {market_trend} (动量: {market_momentum:.2f})")


if __name__ == "__main__":
    main()
