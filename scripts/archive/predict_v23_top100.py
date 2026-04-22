#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.3.0模型预测脚本

预测指定日期收盘后的股票评分，输出Top100
特征处理逻辑与训练/评估完全一致
"""

import sys
import json
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timedelta

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


def extract_features(df):
    """
    提取特征（与训练时完全一致）

    输入: 单只股票的日线数据DataFrame
    输出: 包含所有特征的最后一行
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

    return df


def calculate_risk_score(row):
    """计算风险系数"""
    risk_score = 1.0
    reasons = []

    # 34日涨幅
    ret = row.get("return_34d", 0)
    if pd.notna(ret):
        if ret > 80:
            risk_score *= 0.3
            reasons.append(f"涨幅{ret:.0f}%")
        elif ret > 60:
            risk_score *= 0.5
            reasons.append(f"涨幅{ret:.0f}%")
        elif ret > 40:
            risk_score *= 0.7
            reasons.append(f"涨幅{ret:.0f}%")

    # 最大回撤
    dd = row.get("max_drawdown_20d", 0)
    if pd.notna(dd) and dd < -30:
        risk_score *= 0.7
        reasons.append(f"回撤{dd:.0f}%")

    # ATR
    atr = row.get("atr_ratio_14", 0)
    if pd.notna(atr) and atr > 10:
        risk_score *= 0.7
        reasons.append(f"ATR{atr:.1f}%")

    return risk_score, "; ".join(reasons) if reasons else ""


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
            if pd.isna(val):
                val = 0
            feature_vector.append(float(val))

        # 预测
        dmatrix = xgb.DMatrix([feature_vector], feature_names=feature_names)
        raw_prob = float(booster.predict(dmatrix)[0])
        cal_prob = float(calibrator.predict([raw_prob])[0])

        # 风险评分
        risk_score, risk_reasons = calculate_risk_score(last_row)

        return {
            "ts_code": ts_code,
            "name": name,
            "close": last_row["close"],
            "pct_chg": last_row.get("pct_chg", 0),
            "raw_probability": raw_prob,
            "calibrated_probability": cal_prob,
            "risk_score": risk_score,
            "final_score": cal_prob * risk_score,
            "return_34d": last_row.get("return_34d", 0),
            "rsi_6": last_row.get("rsi_6", 0),
            "max_drawdown_20d": last_row.get("max_drawdown_20d", 0),
            "atr_ratio_14": last_row.get("atr_ratio_14", 0),
            "risk_reasons": risk_reasons,
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="v2.3.0模型预测")
    parser.add_argument("--date", type=str, default="20251231", help="预测日期 (YYYYMMDD)")
    parser.add_argument("--top-n", type=int, default=100, help="输出Top N股票数量")
    args = parser.parse_args()

    predict_date = args.date
    top_n = args.top_n

    log.info("=" * 80)
    log.info(f"v2.3.0模型预测 - {predict_date}")
    log.info("=" * 80)

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

    # 批量预测
    log.info("\n🔄 开始预测...")
    results = []
    total = len(valid)
    processed = 0
    success = 0

    start_time = datetime.now()

    for idx, (_, row) in enumerate(valid.iterrows()):
        ts_code = row["ts_code"]
        name = row["name"]

        result = process_single_stock(dm, ts_code, name, predict_date, feature_names, booster, calibrator)

        processed += 1
        if result is not None:
            results.append(result)
            success += 1

        # 进度显示
        if processed % 200 == 0 or processed == total:
            elapsed = (datetime.now() - start_time).total_seconds()
            speed = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / speed / 60 if speed > 0 else 0
            log.info(
                f"📈 进度: {processed}/{total} ({processed/total*100:.1f}%) | 成功: {success} | 速度: {speed:.1f}只/秒 | 预计剩余: {eta:.1f}分钟"
            )

    log.success(f"\n✓ 预测完成: {success}/{total} 只股票")

    # 创建DataFrame并排序
    df_pred = pd.DataFrame(results)
    df_pred = df_pred.sort_values("final_score", ascending=False)

    # 输出Top N
    df_top = df_pred.head(top_n)

    log.info("\n" + "=" * 80)
    log.info(f"📋 Top {top_n} 股票预测结果")
    log.info("=" * 80)

    # 统计信息
    log.info("\n📊 统计信息:")
    log.info(
        f"  校准概率 - 最高: {df_top['calibrated_probability'].max():.4f}, 平均: {df_top['calibrated_probability'].mean():.4f}"
    )
    log.info(f"  风险评分 - 最高: {df_top['risk_score'].max():.2f}, 平均: {df_top['risk_score'].mean():.2f}")
    log.info(f"  最终得分 - 最高: {df_top['final_score'].max():.4f}, 平均: {df_top['final_score'].mean():.4f}")

    # 按概率分层统计
    log.info(f"\n📈 Top {top_n} 概率分布:")
    prob_bins = [(0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6), (0.0, 0.5)]
    for low, high in prob_bins:
        count = ((df_top["calibrated_probability"] >= low) & (df_top["calibrated_probability"] < high)).sum()
        if count > 0:
            log.info(f"  概率[{low:.1f}, {high:.1f}): {count} 只")

    # 显示Top 20
    log.info("\n🏆 Top 20 股票:")
    log.info("-" * 100)
    log.info(
        f"{'排名':>4} | {'代码':>10} | {'名称':>8} | {'收盘价':>8} | {'校准概率':>8} | {'风险评分':>8} | {'最终得分':>8} | {'34日涨幅':>8}"
    )
    log.info("-" * 100)

    for i, (_, row) in enumerate(df_top.head(20).iterrows()):
        log.info(
            f"{i+1:>4} | {row['ts_code']:>10} | {row['name']:>8} | {row['close']:>8.2f} | {row['calibrated_probability']:>8.4f} | {row['risk_score']:>8.2f} | {row['final_score']:>8.4f} | {row['return_34d']:>7.1f}%"
        )

    # 保存结果
    output_dir = PROJECT_ROOT / "data" / "prediction" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存Top N
    output_file = output_dir / f"v2.3.0_top{top_n}_{predict_date}.csv"
    df_top.to_csv(output_file, index=False, encoding="utf-8-sig")
    log.success(f"\n💾 Top {top_n} 已保存: {output_file}")

    # 保存全量预测结果
    full_file = output_dir / f"v2.3.0_full_predictions_{predict_date}.csv"
    df_pred.to_csv(full_file, index=False, encoding="utf-8-sig")
    log.success(f"💾 全量预测已保存: {full_file}")

    log.info("\n" + "=" * 80)
    log.info("✅ 预测任务完成!")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
