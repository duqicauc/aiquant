#!/usr/bin/env python3
"""
v3.0.0 批量快速预测（基于 ArcticDB）

核心优化：
1. 从 ArcticDB 一次性读取全日期范围数据
2. 多只股票合并后一次性调用 FeatureEngineer.compute_all_features
3. 避免逐只股票、逐日期的重复 API 调用

Usage:
    python scripts/predict_v3_fast.py --start 20260105 --end 20260508
"""
import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.arctic_provider import ArcticDataProvider
from src.features.feature_engineer import FeatureEngineer
from src.features.multits_flattener import flatten_multits
from src.utils.logger import log

MODEL_DIR = PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / "v3.0.0"
META_PATH = MODEL_DIR / "feature_cols.json"
MODEL_PATH = MODEL_DIR / "xgb_flat_final.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "prediction" / "v3.0.0"

LOOKBACK_DAYS = 34

# v27 特征列（与训练时一致）
V27_FEATURES = [
    "ma10",
    "price_position_55d",
    "return_55d",
    "support_20d",
    "breakout_ma10",
    "resistance_55d",
    "price_vs_ma_55d",
    "low_34d",
    "trend_slope_34d",
    "price_vs_ma_34d",
    "vol",
    "dist_to_support_20d",
    "volume_trend_slope_10d",
    "obv_calc",
    "breakout_high_55d",
    "total_mv",
    "ma_8d",
    "high_volume_breakout",
    "support_strength_10d",
    "macd_dea",
    "ma_34d",
    "volume_ratio",
    "turnover_rate",
    "volume_rsv_20d",
    "breakout_ma5",
    "volume_trend_slope_20d",
    "momentum_10d",
    "volume_change",
    "volume_price_corr_10d",
    "close",
    "price_down_vol_up_count_10d",
    "price_down_vol_up",
    "price_vs_ma_8d",
    "low_55d",
    "support_55d",
    "resistance_20d",
    "volume_price_match_sum_10d",
    "volume_price_corr_20d",
    "breakout_ma55",
    "high",
    "trend_slope_8d",
    "volume_breakout_count_20d",
    "breakout_high_10d",
    "high_8d",
    "low_8d",
    "open",
    "change",
    "resistance_strength_10d",
    "price_position_34d",
    "pct_chg",
    "high_34d",
    "rsi_12",
    "macd",
    "low",
    "volatility_34d",
    "trend_slope_55d",
    "momentum_5d",
    "return_8d",
    "dist_to_support_55d",
    "obv_ma10",
    "breakout_ma20",
    "dist_to_resistance_55d",
    "obv_trend",
    "momentum_20d",
    "ma5",
    "support_strength_20d",
    "return_34d",
    "channel_width_20d",
    "resistance_10d",
    "circ_mv",
    "price_change",
    "high_55d",
    "consecutive_new_high",
    "volume_price_match",
    "price_position_8d",
    "price_up_vol_down_count_10d",
    "support_10d",
    "resistance_strength_20d",
    "ma_10d",
    "dist_to_resistance_20d",
    "volatility_55d",
    "ma_5d",
    "momentum_acceleration",
    "ma_55d",
    "support_strength_55d",
    "price_up_vol_down",
    "dist_to_resistance_10d",
    "amount",
    "rsi_6",
    "pre_close",
    "ma_20d",
    "breakout_volume_ratio",
    "breakout_high_20d",
    "dist_to_support_10d",
    "macd_dif",
    "rsi_24",
    "volatility_8d",
    "resistance_strength_55d",
    "price_vs_hist_mean",
    "price_vs_hist_high",
    "volatility_vs_hist",
    "turnover_rate_f",
    "bias_short",
    "bias_mid",
    "bias_long",
    "ema_5",
    "ema_10",
    "ema_20",
    "ema_60",
    "obv",
    "vol_ma5_ratio",
    "vol_ma20_ratio",
    "is_limit_up",
    "max_drawdown_10d",
    "max_drawdown_20d",
    "max_drawdown_55d",
    "atr_14",
    "atr_ratio_14",
    "atr_expansion",
    "days_from_high_20d",
    "days_from_high_55d",
    "recovery_ratio_20d",
    "price_range_pct",
    "close_vs_ma10_std",
    "days_near_ma10",
    "volume_shrink_ratio",
    "ma10_cross_count",
    "kdj_d",
    "kdj_j",
    "kdj_k",
    "prev_high_20d",
    "prev_high_55d",
    "prev_high_10d",
    "breakout_with_volume",
    "momentum_market_interaction",
    "rsi_kdj_divergence",
    "trend_consistency",
    "volume_price_divergence",
    "breakout_rsi_interaction",
    "relative_volatility",
    "resonance_volume_confirm",
    "market_pct_chg",
    "market_return_34d",
    "market_volatility_34d",
    "market_trend",
    "market_momentum_5d",
    "market_momentum_10d",
    "market_momentum_20d",
    "market_regime",
    "market_position_20d",
    "excess_return",
    "excess_return_cumsum",
    "excess_return_consistency",
    "breakout_strength_10d",
    "breakout_strength_20d",
    "breakout_strength_55d",
    "breakout_volume_strength",
    "breakout_confirmed_10d",
    "breakout_confirmed_20d",
    "breakout_resonance",
    "turnover_zscore",
    "turnover_change_rate",
    "turnover_spike",
    "rsi_kdj_golden_cross",
    "rsi_kdj_strength",
    "rsi_zone",
    "volume_price_divergence_strength",
    "volume_price_confirm",
    "breakout_strength_avg",
    "breakout_strength_max",
    "ma_alignment_score",
    "price_position_avg",
    "sharpe_like_34d",
]

META_COLS = ["sample_id", "ts_code", "name", "trade_date", "days_to_t1", "label"]


def load_model() -> xgb.Booster:
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    return model


def load_meta() -> dict:
    with open(META_PATH, "r") as f:
        return json.load(f)


def predict_range_fast(start_date: str, end_date: str):
    """基于 ArcticDB 的批量快速预测"""
    log.info(f"{'='*60}")
    log.info(f"V3.0.0 快速预测: {start_date} ~ {end_date}")
    log.info(f"{'='*60}")

    model = load_model()
    meta = load_meta()
    flat_cols = meta["feature_cols"]
    expected_days = meta["expected_days"]
    engineer = FeatureEngineer()
    provider = ArcticDataProvider()

    # 1. 读取全量数据（扩大日期范围以覆盖 lookback）
    start_dt = pd.to_datetime(start_date) - timedelta(days=LOOKBACK_DAYS + 20)
    end_dt = pd.to_datetime(end_date)

    log.info(f"从 ArcticDB 读取数据: {start_dt.strftime('%Y%m%d')} ~ {end_dt.strftime('%Y%m%d')}")
    df_raw = provider.read_daily_combined(start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d"))
    if df_raw.empty:
        log.error("ArcticDB 数据为空")
        return {}

    df_raw["trade_date"] = pd.to_datetime(df_raw["trade_date"])
    log.info(
        f"  读取到 {len(df_raw)} 行, {df_raw['ts_code'].nunique()} 只股票, {df_raw['trade_date'].nunique()} 个交易日"
    )

    # 2. 读取市场环境数据（上证指数）
    df_market = provider.read_market_index(start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d"))
    if not df_market.empty:
        if "trade_date" not in df_market.columns:
            df_market = df_market.reset_index()
        df_market["trade_date"] = pd.to_datetime(df_market["trade_date"])
        log.info(f"  上证指数数据: {len(df_market)} 行")
    else:
        log.warning("  上证指数数据为空")
        df_market = pd.DataFrame()

    # 3. 获取预测日期列表
    pred_dates = sorted(df_raw[df_raw["trade_date"] >= pd.to_datetime(start_date)]["trade_date"].unique())
    log.info(f"预测日期数: {len(pred_dates)}")

    # 4. 逐预测日期处理，但每日期批量计算所有股票
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for i, pred_date in enumerate(pred_dates, 1):
        log.info(f"[{i}/{len(pred_dates)}] 预测 {pred_date.strftime('%Y%m%d')}...")

        # 提取 lookback 窗口数据
        lookback_end = pred_date - timedelta(days=1)
        lookback_start = pred_date - timedelta(days=LOOKBACK_DAYS + 10)
        df_window = df_raw[(df_raw["trade_date"] >= lookback_start) & (df_raw["trade_date"] <= lookback_end)].copy()

        if df_window.empty:
            log.warning(f"  {pred_date.strftime('%Y%m%d')}: lookback 数据为空")
            continue

        # 每只股票取最近 LOOKBACK_DAYS 个交易日
        grouped = df_window.groupby("ts_code")
        all_stock_slices = []
        valid_stocks = []

        for ts_code, group in grouped:
            group = group.sort_values("trade_date").tail(LOOKBACK_DAYS)
            if len(group) < LOOKBACK_DAYS * 0.7:
                continue
            group["__stock__"] = ts_code  # 标记股票，用于后续拆分
            all_stock_slices.append(group)
            valid_stocks.append(ts_code)

        if not all_stock_slices:
            log.warning(f"  {pred_date.strftime('%Y%m%d')}: 无有效股票")
            continue

        log.info(f"  有效股票: {len(valid_stocks)}")

        # 合并所有股票的切片为一个大数据框
        df_batch = pd.concat(all_stock_slices, ignore_index=True)
        df_batch = df_batch.drop(columns=["__stock__"], errors="ignore")

        # 市场环境数据：取同日期范围的上证指数
        if not df_market.empty:
            mkt = df_market[
                (df_market["trade_date"] >= df_window["trade_date"].min())
                & (df_market["trade_date"] <= df_window["trade_date"].max())
            ].copy()
        else:
            mkt = pd.DataFrame()

        # 一次性批量计算特征！
        try:
            df_feat_all = engineer.compute_all_features(df_batch, mkt)
        except Exception as e:
            log.error(f"  批量特征计算失败: {e}")
            continue

        # 按股票拆分，添加元数据
        sample_features = []
        for ts_code in valid_stocks:
            df_stock = df_feat_all[df_feat_all["ts_code"] == ts_code].copy()
            if len(df_stock) < LOOKBACK_DAYS * 0.7:
                continue
            df_stock = df_stock.sort_values("trade_date").tail(LOOKBACK_DAYS)
            df_stock["sample_id"] = ts_code
            df_stock["name"] = ""
            df_stock["days_to_t1"] = range(-len(df_stock), 0)
            df_stock["label"] = 0
            df_stock["trade_date"] = pred_date  # T1 日期
            sample_features.append(df_stock)

        if not sample_features:
            log.warning(f"  {pred_date.strftime('%Y%m%d')}: 拆分后无有效股票")
            continue

        # 合并所有股票的多行数据
        df_all_samples = pd.concat(sample_features, ignore_index=True)

        # 过滤 v27 特征
        keep_cols = [c for c in META_COLS + V27_FEATURES if c in df_all_samples.columns]
        df_all_samples = df_all_samples[keep_cols].copy()

        # 展平
        feature_cols = [c for c in df_all_samples.columns if c not in set(META_COLS)]
        df_flat = flatten_multits(df_all_samples, feature_cols, expected_days)

        if df_flat.empty:
            log.warning(f"  {pred_date.strftime('%Y%m%d')}: 展平结果为空")
            continue

        # 对齐特征并预测
        aligned = pd.DataFrame(index=df_flat.index)
        for col in flat_cols:
            aligned[col] = df_flat[col] if col in df_flat.columns else 0.0

        dmatrix = xgb.DMatrix(aligned.values, feature_names=flat_cols)
        probs = model.predict(dmatrix)

        result = pd.DataFrame(
            {
                "ts_code": df_flat["ts_code"].values,
                "trade_date": pred_date.strftime("%Y%m%d"),
                "prob": probs,
            }
        )
        result = result.sort_values("prob", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)

        # 保存
        date_str = pred_date.strftime("%Y%m%d")
        result.to_csv(OUTPUT_DIR / f"predictions_{date_str}_all.csv", index=False)
        result.head(100).to_csv(OUTPUT_DIR / f"predictions_{date_str}_top100.csv", index=False)
        result.head(50).to_csv(OUTPUT_DIR / f"predictions_{date_str}_top50.csv", index=False)

        all_results[date_str] = result
        log.info(f"  预测完成: {len(result)} 只股票")

    log.success(f"全部预测完成: {len(all_results)} 个交易日")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="v3.0.0 快速批量预测")
    parser.add_argument("--start", required=True, help="开始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="结束日期 YYYYMMDD")
    args = parser.parse_args()

    predict_range_fast(args.start, args.end)


if __name__ == "__main__":
    main()
