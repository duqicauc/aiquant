#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.3.2模型预测脚本 - 统一版

整合三种预测模式：
1. 基础模式：v2.3.2追高控制，RSI过热惩罚
2. 主板模式：仅沪深主板，可选RSI区间筛选
3. 右侧模式：板块热度加成，轻度追高惩罚，不做RSI惩罚

用法示例：
# 基础预测
python predict_v232_unified.py --date 20260122

# 主板+基本面
python predict_v232_unified.py --date 20260122 --mainboard --fundamental

# 主板+RSI健康筛选
python predict_v232_unified.py --date 20260122 --mainboard --rsi-filter 40 70

# 右侧方案（板块加成）
python predict_v232_unified.py --date 20260122 --mainboard --right-side

# 完整方案（主板+基本面+右侧）
python predict_v232_unified.py --date 20260122 --mainboard --fundamental --right-side
"""

import sys
import json
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.logger import log
from src.data.data_manager import DataManager
from src.models.screening.fundamental_screener import FundamentalScreener


# ============================================================
# 热门板块配置（右侧模式使用）
# ============================================================
HOT_SECTORS = {
    # 电力/能源
    "特高压": 1.2,
    "电力": 1.15,
    "电气设备": 1.15,
    "储能": 1.15,
    "光伏": 1.1,
    "新能源": 1.1,
    "油气": 1.1,
    "天然气": 1.1,
    # 科技/AI
    "人工智能": 1.15,
    "AI": 1.15,
    "机器人": 1.15,
    "算力": 1.1,
    "芯片": 1.1,
    "半导体": 1.1,
    "消费电子": 1.1,
    "存储": 1.1,
    # 汽车
    "汽车": 1.1,
    "汽车配件": 1.1,
    "锂电池": 1.1,
    # 军工
    "军工": 1.15,
    "航天航空": 1.15,
    "国防军工": 1.15,
    "航天": 1.15,
    # 金融
    "券商": 1.1,
    "保险": 1.05,
    "银行": 1.05,
    # 资源
    "有色金属": 1.1,
    "稀土": 1.1,
    "黄金": 1.1,
    "贵金属": 1.1,
    # 机械
    "机械": 1.05,
    "机械基件": 1.1,
    "专用机械": 1.05,
}


def is_main_board(ts_code):
    """
    判断是否为沪深主板股票

    沪深主板：
    - 上海主板：600xxx, 601xxx, 603xxx, 605xxx
    - 深圳主板：000xxx, 001xxx, 002xxx, 003xxx

    过滤掉：
    - 创业板：300xxx
    - 科创板：688xxx
    - 北交所：8xxxxx, 4xxxxx, 920xxx
    """
    code = ts_code.split(".")[0]

    # 上海主板
    if code.startswith(("600", "601", "603", "605")):
        return True

    # 深圳主板
    if code.startswith(("000", "001", "002", "003")):
        return True

    return False


def load_model():
    """加载v2.3.0模型"""
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / "v2.3.0" / "model"

    booster = xgb.Booster()
    booster.load_model(str(model_dir / "model.json"))

    with open(model_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    calibrator = joblib.load(str(model_dir / "calibrator.pkl"))

    return booster, feature_names, calibrator


def extract_features(df):
    """提取特征"""
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

    # ========== 连续涨停天数 ==========
    df["consecutive_limit_up"] = df["is_limit_up"].rolling(3, min_periods=1).sum()

    return df


def calculate_score(cal_prob, expected_return_score, pct_chg, rsi_6, consecutive_limit_up, right_side=False):
    """
    统一评分函数

    Args:
        right_side: True=右侧模式（轻度追高惩罚，不做RSI惩罚）
                   False=标准模式（严格追高惩罚，RSI过热惩罚）
    """
    # 基础评分：0.6*校准概率 + 0.4*预期收益
    base_score = 0.6 * cal_prob + 0.4 * expected_return_score

    penalty = 1.0
    penalty_reasons = []

    if right_side:
        # ===== 右侧模式：轻度惩罚，不惩罚RSI =====
        # 1. 当日涨停惩罚（轻度）
        if pct_chg >= 9.8:
            penalty *= 0.85
            penalty_reasons.append(f"涨停({pct_chg:.1f}%)")
        elif pct_chg > 7:
            penalty *= 0.95
            penalty_reasons.append(f"大涨({pct_chg:.1f}%)")

        # 2. 连续涨停惩罚
        if consecutive_limit_up >= 3:
            penalty *= 0.7
            penalty_reasons.append(f"连板{int(consecutive_limit_up)}天")
        elif consecutive_limit_up >= 2:
            penalty *= 0.85
            penalty_reasons.append(f"连板{int(consecutive_limit_up)}天")
    else:
        # ===== 标准模式：严格惩罚 =====
        # 1. 追高惩罚：当日涨幅>15%
        if pct_chg > 15:
            penalty *= 0.5
            penalty_reasons.append(f"追高惩罚(涨幅{pct_chg:.1f}%)")
        elif pct_chg > 10:
            penalty *= 0.8
            penalty_reasons.append(f"轻度追高(涨幅{pct_chg:.1f}%)")

        # 2. 涨停低概率惩罚
        if pct_chg >= 9.8 and cal_prob < 0.8:
            penalty *= 0.7
            penalty_reasons.append(f"涨停低概率({cal_prob:.2f})")

        # 3. RSI过热惩罚
        if rsi_6 > 95:
            penalty *= 0.8
            penalty_reasons.append(f"RSI过热({rsi_6:.1f})")
        elif rsi_6 > 90:
            penalty *= 0.9
            penalty_reasons.append(f"RSI偏高({rsi_6:.1f})")

        # 4. 连续涨停惩罚
        if consecutive_limit_up >= 3:
            penalty *= 0.6
            penalty_reasons.append(f"连续涨停({int(consecutive_limit_up)}天)")
        elif consecutive_limit_up >= 2:
            penalty *= 0.8
            penalty_reasons.append(f"连续涨停({int(consecutive_limit_up)}天)")

    final_score = base_score * penalty

    return final_score, base_score, penalty, penalty_reasons


def get_sector_boost(industry, hot_sectors):
    """获取板块热度加成"""
    if pd.isna(industry):
        return 1.0, False

    for sector, boost in hot_sectors.items():
        if sector in industry:
            return boost, True

    return 1.0, False


def process_single_stock(dm, ts_code, name, predict_date, feature_names, booster, calibrator, right_side=False):
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

        # 评分
        final_score, base_score, penalty, penalty_reasons = calculate_score(
            cal_prob, expected_return_norm, pct_chg, rsi_6, consecutive_limit_up, right_side
        )

        return {
            "ts_code": ts_code,
            "name": name,
            "close": float(last_row["close"]),
            "pct_chg": pct_chg,
            "amount": amount,
            "raw_probability": raw_prob,
            "calibrated_probability": cal_prob,
            "expected_return_score": expected_return_score,
            "base_score": base_score,
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
    parser = argparse.ArgumentParser(
        description="v2.3.2模型预测 - 统一版（支持基础/主板/右侧模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例：
  # 基础预测
  python predict_v232_unified.py --date 20260122

  # 主板+基本面
  python predict_v232_unified.py --date 20260122 --mainboard --fundamental

  # 主板+RSI健康筛选（40-70）
  python predict_v232_unified.py --date 20260122 --mainboard --rsi-filter 40 70

  # 右侧方案（板块加成，不惩罚RSI）
  python predict_v232_unified.py --date 20260122 --mainboard --right-side

  # 完整方案（主板+基本面+右侧）
  python predict_v232_unified.py --date 20260122 --mainboard --fundamental --right-side
        """,
    )

    # 基础参数
    parser.add_argument("--date", type=str, required=True, help="预测日期 (YYYYMMDD)")
    parser.add_argument("--min-amount", type=float, default=30000, help="最小成交额（千元），默认3000万")
    parser.add_argument("--top-n", type=int, default=50, help="从Top N中筛选，默认50")
    parser.add_argument("--output-n", type=int, default=10, help="输出股票数量，默认10")

    # 筛选模式
    parser.add_argument("--mainboard", action="store_true", help="仅沪深主板（过滤创业板/科创板/北交所）")
    parser.add_argument("--fundamental", action="store_true", help="启用基本面筛选")
    parser.add_argument("--right-side", action="store_true", help="右侧模式（板块加成，轻度追高惩罚，不惩罚RSI）")
    parser.add_argument(
        "--rsi-filter",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="RSI区间筛选，如 --rsi-filter 40 70（与--right-side互斥）",
    )

    # 板块配置
    parser.add_argument("--hot-sectors", type=str, default="", help="自定义热门板块，逗号分隔，如：特高压,电力,机器人")

    args = parser.parse_args()

    # 参数校验
    if args.right_side and args.rsi_filter:
        log.warning("--right-side 和 --rsi-filter 不能同时使用，右侧模式下忽略RSI筛选")
        args.rsi_filter = None

    predict_date = args.date
    min_amount = args.min_amount
    top_n = args.top_n
    output_n = args.output_n

    # 构建模式描述
    mode_parts = []
    if args.mainboard:
        mode_parts.append("主板")
    if args.fundamental:
        mode_parts.append("基本面")
    if args.right_side:
        mode_parts.append("右侧")
    if args.rsi_filter:
        mode_parts.append(f"RSI{args.rsi_filter[0]:.0f}-{args.rsi_filter[1]:.0f}")

    mode_str = "+".join(mode_parts) if mode_parts else "基础"

    # 合并热门板块
    hot_sectors = HOT_SECTORS.copy()
    if args.hot_sectors:
        custom_sectors = [s.strip() for s in args.hot_sectors.split(",")]
        for sector in custom_sectors:
            hot_sectors[sector] = 1.2
        log.info(f"自定义热门板块: {custom_sectors}")

    log.info("=" * 80)
    log.info(f"v2.3.2模型预测 - 统一版 - {predict_date}")
    log.info(f"模式: {mode_str}")
    log.info("=" * 80)
    log.info("执行流程：")
    log.info("  1. 全市场评分（模型对所有股票打分）")
    log.info("  2. 评分后筛选（在排名基础上应用筛选条件）")
    log.info("\n筛选条件（评分后应用）：")
    if args.mainboard:
        log.info("  - 仅沪深主板（过滤创业板、科创板、北交所）")
    if args.fundamental:
        log.info("  - 基本面筛选（市值10-100亿，ROE>5%等）")
    if args.right_side:
        log.info("  - 右侧模式（板块加成，轻度追高惩罚，不惩罚RSI）")
    if args.rsi_filter:
        log.info(f"  - RSI区间: {args.rsi_filter[0]:.0f}-{args.rsi_filter[1]:.0f}")
    log.info(f"  - 最小成交额: {min_amount/1000:.0f}百万元")
    log.info(f"  - 从Top{top_n}中输出Top{output_n}")

    # 初始化
    dm = DataManager()

    # 加载模型
    log.info("\n📦 加载v2.3.0模型...")
    booster, feature_names, calibrator = load_model()
    log.success(f"✓ 模型加载成功: {len(feature_names)} 特征")

    # 获取股票列表
    stock_list = dm.get_stock_list()

    # 仅过滤ST和北交所（对全市场评分）
    valid = stock_list[
        ~stock_list["name"].str.contains("ST|退", na=False)
        & ~stock_list["ts_code"].str.startswith("8")
        & ~stock_list["ts_code"].str.startswith("4")  # 北交所
        & ~stock_list["ts_code"].str.startswith("920")  # 北交所  # 北交所
    ].copy()
    log.info(f"📊 全市场股票（过滤ST/北交所）: {len(valid)} 只")

    # 基本面筛选器（预先准备，评分后使用）
    fundamental_screener = None
    fundamental_passed_codes = None
    if args.fundamental:
        log.info("\n准备基本面筛选器（将在评分后应用）...")
        fundamental_screener = FundamentalScreener(
            dm,
            config={
                "enabled": True,
                "market_cap_min": 100000,
                "market_cap_max": 1000000,
                "revenue_min": 5e8,
                "net_profit_min": 5000000,
                "roe_min": 5,
                "roa_min": 2,
            },
        )
        # 预先获取通过基本面筛选的股票代码（异步处理）
        log.info("正在获取基本面数据...")
        fundamental_passed = fundamental_screener.filter_stocks(valid.copy(), predict_date)
        fundamental_passed_codes = set(fundamental_passed["ts_code"].tolist())
        log.info(f"基本面筛选通过: {len(fundamental_passed_codes)} 只")

    # 批量处理 - 对全市场评分
    log.info(f"\n🚀 开始对全市场 {len(valid)} 只股票评分...")
    results = []
    total = len(valid)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for idx, row in valid.iterrows():
            future = executor.submit(
                process_single_stock,
                dm,
                row["ts_code"],
                row["name"],
                predict_date,
                feature_names,
                booster,
                calibrator,
                args.right_side,
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
    df_all = pd.DataFrame(results)
    log.info(f"全市场评分完成: {len(df_all)} 只股票")

    # 添加板块信息
    industry_map = stock_list.set_index("ts_code")["industry"].to_dict()
    df_all["industry"] = df_all["ts_code"].map(industry_map)

    # 添加主板标记
    df_all["is_main_board"] = df_all["ts_code"].apply(is_main_board)

    # 右侧模式：计算板块加成
    if args.right_side:
        df_all["sector_boost"] = df_all["industry"].apply(lambda x: get_sector_boost(x, hot_sectors)[0])
        df_all["is_hot_sector"] = df_all["industry"].apply(lambda x: get_sector_boost(x, hot_sectors)[1])
        # 综合评分 = 基础评分 * 板块加成
        df_all["final_score"] = df_all["base_score"] * df_all["sector_boost"]
    else:
        df_all["sector_boost"] = 1.0
        df_all["is_hot_sector"] = False

    # 按final_score排序（全市场排名）
    df_all = df_all.sort_values("final_score", ascending=False).reset_index(drop=True)
    df_all["market_rank"] = range(1, len(df_all) + 1)

    # ============================================================
    # 评分后筛选（在全市场排名基础上筛选）
    # ============================================================
    df_results = df_all.copy()

    # 1. 流动性过滤
    before_filter = len(df_results)
    df_results = df_results[df_results["amount"] >= min_amount]
    log.info("\n📊 筛选过程（评分后）:")
    log.info(f"  流动性过滤 (>{min_amount/1000:.0f}百万): {before_filter} -> {len(df_results)}")

    # 2. 主板筛选
    if args.mainboard:
        before_filter = len(df_results)
        df_results = df_results[df_results["is_main_board"]].copy()
        log.info(f"  主板筛选: {before_filter} -> {len(df_results)}")

    # 3. 基本面筛选
    if args.fundamental and fundamental_passed_codes:
        before_filter = len(df_results)
        df_results = df_results[df_results["ts_code"].isin(fundamental_passed_codes)].copy()
        log.info(f"  基本面筛选: {before_filter} -> {len(df_results)}")

    # 重新排序并添加筛选后排名
    df_results = df_results.sort_values("final_score", ascending=False).reset_index(drop=True)
    df_results["filtered_rank"] = range(1, len(df_results) + 1)

    # 取Top N
    df_top_n = df_results.head(top_n).copy()
    df_top_n["rank"] = range(1, len(df_top_n) + 1)

    # 4. RSI筛选（从Top N中筛选）
    if args.rsi_filter:
        rsi_min, rsi_max = args.rsi_filter
        df_rsi_filtered = df_top_n[(df_top_n["rsi_6"] >= rsi_min) & (df_top_n["rsi_6"] <= rsi_max)].copy()
        log.info(f"  RSI筛选 ({rsi_min:.0f}-{rsi_max:.0f}): {len(df_top_n)} -> {len(df_rsi_filtered)}")
        df_output = df_rsi_filtered.head(output_n)
    else:
        df_output = df_top_n.head(output_n)

    log.success("\n✓ 预测完成")

    # 显示结果
    log.info("\n" + "=" * 130)
    log.info(f"🏆 v2.3.2 Top{output_n}（{mode_str}模式）")
    log.info("=" * 130)

    if args.right_side:
        log.info(
            f"\n{'排名':<4} {'全市场':<8} {'代码':<12} {'名称':<10} {'板块':<12} {'热门':<6} {'综合分':<10} {'基础分':<10} {'RSI':<8} {'涨幅':<10}"
        )
        log.info("-" * 140)
        for _, row in df_output.iterrows():
            hot_mark = "🔥" if row["is_hot_sector"] else "-"
            industry = row.get("industry", "未知") or "未知"
            market_rank = row.get("market_rank", "-")
            log.info(
                f"{row['rank']:<4} #{market_rank:<7} {row['ts_code']:<12} {row['name']:<10} "
                f"{industry:<12} {hot_mark:<6} {row['final_score']:<10.4f} "
                f"{row['base_score']:<10.4f} {row['rsi_6']:<8.1f} {row['pct_chg']:>+8.2f}%"
            )
    else:
        log.info(
            f"\n{'排名':<4} {'全市场':<8} {'代码':<12} {'名称':<10} {'板块':<12} {'综合分':<10} {'RSI':<8} {'涨幅':<10} {'惩罚':<10}"
        )
        log.info("-" * 120)
        for _, row in df_output.iterrows():
            industry = row.get("industry", "未知") or "未知"
            penalty_str = f"{row['penalty']:.2f}" if row["penalty"] < 1.0 else "无"
            market_rank = row.get("market_rank", "-")
            log.info(
                f"{row['rank']:<4} #{market_rank:<7} {row['ts_code']:<12} {row['name']:<10} "
                f"{industry:<12} {row['final_score']:<10.4f} "
                f"{row['rsi_6']:<8.1f} {row['pct_chg']:>+8.2f}% {penalty_str:<10}"
            )

    # 热门板块统计（右侧模式）
    if args.right_side:
        hot_stocks = df_top_n[df_top_n["is_hot_sector"]]
        if len(hot_stocks) > 0:
            log.info("\n" + "=" * 80)
            log.info(f"🔥 热门板块股票（共{len(hot_stocks)}只）")
            log.info("=" * 80)

            industry_counts = Counter(hot_stocks["industry"].dropna())
            for industry, count in industry_counts.most_common(10):
                stocks = hot_stocks[hot_stocks["industry"] == industry]["name"].tolist()
                boost = 1.0
                for sector, b in hot_sectors.items():
                    if sector in industry:
                        boost = b
                        break
                log.info(f"  {industry} (加成{boost:.0%}): {', '.join(stocks[:5])}")

    # 保存结果
    output_dir = PROJECT_ROOT / "data" / "prediction" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 保存全市场完整评分结果（最重要，评分后筛选的基础）
    full_market_file = output_dir / f"v232_full_market_{predict_date}.csv"
    df_all.to_csv(full_market_file, index=False, encoding="utf-8-sig")
    log.info(f"\n💾 全市场评分结果: {full_market_file} ({len(df_all)} 只)")

    # 2. 保存筛选后的Top N结果
    filename_parts = ["v232", "unified"]
    if args.mainboard:
        filename_parts.append("mainboard")
    if args.fundamental:
        filename_parts.append("fundamental")
    if args.right_side:
        filename_parts.append("rightside")
    if args.rsi_filter:
        filename_parts.append(f"rsi{int(args.rsi_filter[0])}-{int(args.rsi_filter[1])}")
    filename_parts.append(predict_date)

    output_file = output_dir / f"{'_'.join(filename_parts)}.csv"
    df_output.to_csv(output_file, index=False, encoding="utf-8-sig")
    log.success(f"💾 筛选结果: {output_file} ({len(df_output)} 只)")

    # 3. 保存筛选后的完整结果（流动性+主板+基本面筛选后）
    if args.mainboard or args.fundamental:
        filtered_file = output_dir / f"v232_filtered_{predict_date}.csv"
        df_results.to_csv(filtered_file, index=False, encoding="utf-8-sig")
        log.info(f"💾 筛选后完整结果: {filtered_file} ({len(df_results)} 只)")

    # 统计
    log.info("\n" + "=" * 80)
    log.info("📊 统计信息")
    log.info("=" * 80)
    log.info(f"全市场股票: {len(df_all)}")
    log.info(f"筛选后股票: {len(df_results)}")
    log.info(f"输出股票平均综合分: {df_output['final_score'].mean():.4f}")
    log.info(f"输出股票平均RSI: {df_output['rsi_6'].mean():.1f}")
    if args.right_side:
        log.info(f"Top{top_n}中热门板块股票: {len(df_top_n[df_top_n['is_hot_sector']])}")

    # 显示筛选效果
    if args.mainboard or args.fundamental:
        log.info("\n💡 筛选效果:")
        log.info(
            f"  全市场Top10中，筛选后保留: {len(df_all.head(10)[df_all.head(10)['ts_code'].isin(df_results['ts_code'])])}/10"
        )
        log.info(
            f"  全市场Top50中，筛选后保留: {len(df_all.head(50)[df_all.head(50)['ts_code'].isin(df_results['ts_code'])])}/50"
        )


if __name__ == "__main__":
    main()
