#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试 highhn 模型"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.catboost_predictor import CatBoostPredictor
from src.backtest.backtester_realistic import RealisticBacktester

# 生成预测
print("=== 生成 highhn 2024Q4 预测 ===")
predictor = CatBoostPredictor("v2.9.2-catboost-conservative-highhn")

pred_dir = PROJECT_ROOT / "data" / "prediction" / "v292_highhn_2024q4"

# 直接用本地数据（复用已有特征）
import sqlite3
import pandas as pd
from src.features.feature_engineer import FeatureEngineer
from src.data.tushare_data_provider import TushareDataProvider

DB_PATH = PROJECT_ROOT / "data" / "cache" / "quant_data.db"
FACTOR_CACHE = PROJECT_ROOT / "data" / "cache" / "stk_factor_pro"

conn = sqlite3.connect(DB_PATH)
df_daily = pd.read_sql_query("SELECT * FROM daily_data WHERE trade_date >= '20240630' AND trade_date <= '20241231'", conn)
df_daily["trade_date"] = pd.to_datetime(df_daily["trade_date"])
df_basic = pd.read_sql_query("SELECT * FROM daily_basic WHERE trade_date >= '20240630' AND trade_date <= '20241231'", conn)
df_basic["trade_date"] = pd.to_datetime(df_basic["trade_date"])
conn.close()

merge_cols = [c for c in df_basic.columns if c not in df_daily.columns or c in ["ts_code", "trade_date"]]
df = pd.merge(df_daily, df_basic[merge_cols], on=["ts_code", "trade_date"], how="left")

all_factors = []
for date_str in sorted(df["trade_date"].dt.strftime("%Y%m%d").unique()):
    pkl_path = FACTOR_CACHE / f"{date_str}.pkl"
    if pkl_path.exists():
        try:
            df_factor = pd.read_pickle(pkl_path)
            if df_factor is not None and not df_factor.empty:
                rename_map = {"macd_dif_qfq": "macd_dif", "macd_dea_qfq": "macd_dea", "macd_qfq": "macd", "rsi_qfq_6": "rsi_6", "rsi_qfq_12": "rsi_12", "rsi_qfq_24": "rsi_24", "kdj_k_qfq": "kdj_k", "kdj_d_qfq": "kdj_d", "kdj_qfq": "kdj_j", "obv_qfq": "obv", "ema_qfq_5": "ema_5", "ema_qfq_10": "ema_10", "ema_qfq_20": "ema_20", "ema_qfq_60": "ema_60", "bias1_qfq": "bias_short", "bias2_qfq": "bias_mid", "bias3_qfq": "bias_long", "ma_qfq_5": "ma5", "ma_qfq_10": "ma10", "ma_qfq_20": "ma_20d", "atr_qfq": "atr"}
                df_factor = df_factor.rename(columns=rename_map)
                df_factor["trade_date"] = pd.to_datetime(date_str) if "trade_date" not in df_factor.columns else pd.to_datetime(df_factor["trade_date"])
                factor_cols = [c for c in df_factor.columns if c not in df.columns or c in ["ts_code", "trade_date"]]
                all_factors.append(df_factor[factor_cols])
        except: pass

if all_factors:
    df_factors = pd.concat(all_factors, ignore_index=True)
    factor_cols = [c for c in df_factors.columns if c not in df.columns or c in ["ts_code", "trade_date"]]
    df = pd.merge(df, df_factors[factor_cols], on=["ts_code", "trade_date"], how="left")

df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
df["sample_id"] = df.groupby("ts_code").ngroup()
df["name"] = ""
df["days_to_t1"] = 0

provider = TushareDataProvider()
df_market = provider.fetch_market_index("20240630", "20241231")

fe = FeatureEngineer()
df_features = fe.compute_all_features(df, df_market)

pred_dir.mkdir(parents=True, exist_ok=True)
for date_str in sorted(df_features[df_features["trade_date"] >= pd.to_datetime("20241001")]["trade_date"].dt.strftime("%Y%m%d").unique()):
    pred_dt = pd.to_datetime(date_str)
    df_pred = df_features[df_features["trade_date"] == pred_dt].copy()
    if df_pred.empty: continue
    df_pred["prob"] = predictor.predict(df_pred)
    X = df_pred[[c for c in predictor.feature_names if c in df_pred.columns]].fillna(0).astype(float)
    df_pred["prob_raw"] = predictor.model.predict_proba(X)[:, 1]
    df_pred = df_pred.sort_values("prob_raw", ascending=False).reset_index(drop=True)
    df_pred["rank"] = range(1, len(df_pred) + 1)
    predictor.save_results(df_pred, date_str, pred_dir)

# Realistic 回测
print("\n=== highhn realistic 回测 ===")
bt = RealisticBacktester(prediction_dir=str(pred_dir), initial_capital=10_000_000, per_stock_amount=300_000, top_n_buy=10, stop_loss_pct=4.0, trailing_stop_pct=5.0, trailing_stop_activation=5.0, enable_sector_filter=False, ma_window=5, ma_consecutive_days=2, buy_slippage_bps=15.0, sell_slippage_bps=20.0, commission_rate=0.00025, min_commission=5.0, stamp_duty_rate=0.001, min_amount=10_000)
result = bt.run(start_date="20241001", end_date="20241231")
if result: print(f"realistic: {result['total_return']:+.2f}%")

# Integrated 回测
print("\n=== highhn integrated 回测 ===")
bt2 = RealisticBacktester(prediction_dir=str(pred_dir), initial_capital=10_000_000, per_stock_amount=300_000, top_n_buy=10, stop_loss_pct=4.0, trailing_stop_pct=5.0, trailing_stop_activation=5.0, enable_sector_filter=True, ma_window=5, ma_consecutive_days=2, buy_slippage_bps=15.0, sell_slippage_bps=20.0, commission_rate=0.00025, min_commission=5.0, stamp_duty_rate=0.001, min_amount=10_000)
result2 = bt2.run(start_date="20241001", end_date="20241231")
if result2: print(f"integrated: {result2['total_return']:+.2f}%")
