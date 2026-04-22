#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.4.0模型预测脚本 - 基于140特征完整版

特点：
1. 使用v2.4.0模型（140个特征，含5个反追龙头特征）
2. 实时计算所有特征
3. 概率校准输出

使用方法：
  python scripts/predict_v240.py --date 20251212 --top 10
"""

import sys
import json
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.logger import log
from src.data.data_manager import DataManager

VERSION = "v2.4.0"


def load_model():
    """加载v2.4.0模型"""
    model_dir = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / VERSION / "model"

    if not model_dir.exists():
        log.error(f"模型目录不存在: {model_dir}")
        return None, None, None

    booster = xgb.Booster()
    booster.load_model(str(model_dir / "model.json"))

    with open(model_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    calibrator = joblib.load(str(model_dir / "calibrator.pkl"))

    log.success(f"✓ 模型已加载: {VERSION} ({len(feature_names)}个特征)")
    return booster, feature_names, calibrator


def extract_features(df):
    """
    提取特征（v2.4.0版本，包含140个特征）

    输入: 单只股票的日线数据DataFrame
    输出: 包含所有特征的DataFrame
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
    # 最大回撤
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

    # ========== v2.4.0新增：反追龙头特征 ==========
    # 1. price_range_pct: 34天振幅百分比
    if "high_34d" in df.columns and "low_34d" in df.columns:
        df["price_range_pct"] = np.where(df["low_34d"] > 0, (df["high_34d"] - df["low_34d"]) / df["low_34d"] * 100, 0)

    # 2. close_vs_ma10_std: 使用 bias_short 的绝对值作为近似
    df["close_vs_ma10_std"] = df["bias_short"].abs()

    # 3. days_near_ma10: 使用 price_position_34d 推导（范围0-100）
    position_norm = df["price_position_34d"] / 100.0  # 归一化到0-1
    df["days_near_ma10"] = (1 - (position_norm - 0.5).abs() * 2) * 34
    df["days_near_ma10"] = df["days_near_ma10"].clip(0, 34)

    # 4. volume_shrink_ratio: 后半段vs前半段的成交量比
    df["volume_shrink_ratio"] = np.where(df["vol_ma20_ratio"] > 0, df["vol_ma5_ratio"] / df["vol_ma20_ratio"], 1)

    # 5. ma10_cross_count: 波动率高 + 乖离率小 = 频繁穿越
    volatility_mean = df["volatility_34d"].mean()
    if volatility_mean > 0:
        volatility_norm = df["volatility_34d"] / volatility_mean
    else:
        volatility_norm = 1
    bias_small = (df["bias_short"].abs() < 3).astype(float)
    df["ma10_cross_count"] = (volatility_norm * bias_small * 10).clip(0, 34)

    return df


def process_single_stock(dm, ts_code, name, predict_date, feature_names, booster, calibrator):
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

        # 获取T1前涨幅（用于评估反追龙头效果）
        return_34d = last_row.get("return_34d", 0)
        if pd.isna(return_34d) or not np.isfinite(return_34d):
            return_34d = 0

        return {
            "ts_code": ts_code,
            "name": name,
            "raw_probability": round(raw_prob, 4),
            "calibrated_probability": round(cal_prob, 4),
            "return_34d": round(return_34d, 2),
            "close": round(last_row.get("close", 0), 2),
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=f"{VERSION}模型预测")
    parser.add_argument("--date", type=str, required=True, help="预测日期(YYYYMMDD)")
    parser.add_argument("--top", type=int, default=10, help="输出Top N股票")
    args = parser.parse_args()

    predict_date = args.date
    top_n = args.top

    log.info("=" * 80)
    log.info(f"{VERSION} 模型预测 - 140特征完整版")
    log.info("=" * 80)
    log.info(f"预测日期: {predict_date}")
    log.info(f"输出数量: Top {top_n}")
    log.info("")

    # 1. 加载模型
    booster, feature_names, calibrator = load_model()
    if booster is None:
        return

    # 2. 初始化数据管理器
    log.info("初始化数据管理器...")
    dm = DataManager()

    # 3. 获取股票列表
    log.info("获取股票列表...")
    stock_list = dm.get_stock_list()

    # 过滤
    valid = stock_list[~stock_list["name"].str.contains("ST|退", na=False) & ~stock_list["ts_code"].str.endswith(".BJ")]
    log.info(f"有效股票: {len(valid)} 只")

    # 4. 并行预测
    log.info("\n开始预测...")
    results = []
    total = len(valid)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for idx, row in valid.iterrows():
            future = executor.submit(
                process_single_stock, dm, row["ts_code"], row["name"], predict_date, feature_names, booster, calibrator
            )
            futures[future] = (row["ts_code"], row["name"])

        completed = 0
        error_count = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 500 == 0 or completed == total:
                log.info(f"进度: {completed}/{total} ({completed/total*100:.1f}%) | 成功: {len(results)}")

            result = future.result()
            if result:
                results.append(result)
            else:
                error_count += 1

    if not results:
        log.error("没有预测结果")
        return

    # 5. 排序并输出
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("calibrated_probability", ascending=False).reset_index(drop=True)

    # Top N
    df_top = df_results.head(top_n)

    log.success(f"\n✓ 预测完成: {len(results)} 只股票")

    # 显示Top N
    log.info("\n" + "=" * 80)
    log.info(f"🏆 {VERSION} Top{top_n} 推荐")
    log.info("=" * 80)
    log.info(f"\n{'排名':<4} {'代码':<12} {'名称':<10} {'校准概率':<10} {'原始概率':<10} {'34日涨幅':<10}")
    log.info("-" * 70)

    for i, (_, row) in enumerate(df_top.iterrows(), 1):
        log.info(
            f"{i:<4} {row['ts_code']:<12} {row['name']:<10} "
            f"{row['calibrated_probability']:<10.4f} {row['raw_probability']:<10.4f} "
            f"{row['return_34d']:>+9.1f}%"
        )

    # 统计Top N的T1前涨幅
    avg_return_34d = df_top["return_34d"].mean()
    low_position_count = (df_top["return_34d"] <= 20).sum()
    log.info("-" * 70)
    log.info(f"Top{top_n}平均T1前涨幅: {avg_return_34d:.1f}%")
    log.info(f"低位启动(<=20%)占比: {low_position_count}/{top_n} ({low_position_count/top_n*100:.0f}%)")

    # 6. 保存结果
    output_dir = PROJECT_ROOT / "data" / "prediction" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{VERSION}_top{top_n}_{predict_date}.csv"
    df_top.to_csv(output_file, index=False, encoding="utf-8-sig")
    log.success(f"\n✓ 结果已保存: {output_file}")

    # 保存完整结果
    full_output = output_dir / f"{VERSION}_full_{predict_date}.csv"
    df_results.to_csv(full_output, index=False, encoding="utf-8-sig")
    log.info(f"✓ 完整结果: {full_output}")


if __name__ == "__main__":
    main()
