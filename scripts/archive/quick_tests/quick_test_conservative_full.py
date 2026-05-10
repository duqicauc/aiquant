#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""保守版模型全季度回测验证（从本地数据库）"""

import sys
import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.catboost_predictor import CatBoostPredictor
from src.backtest.backtester_realistic import RealisticBacktester
from src.data.tushare_data_provider import TushareDataProvider
from src.features.feature_engineer import FeatureEngineer
from src.utils.logger import log

DB_PATH = PROJECT_ROOT / "data" / "cache" / "quant_data.db"
FACTOR_CACHE = PROJECT_ROOT / "data" / "cache" / "stk_factor_pro"


def load_local_data(start_date: str, end_date: str):
    conn = sqlite3.connect(DB_PATH)
    df_daily = pd.read_sql_query(
        f"SELECT * FROM daily_data WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'",
        conn,
    )
    df_daily["trade_date"] = pd.to_datetime(df_daily["trade_date"])
    df_basic = pd.read_sql_query(
        f"SELECT * FROM daily_basic WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'",
        conn,
    )
    df_basic["trade_date"] = pd.to_datetime(df_basic["trade_date"])
    conn.close()

    merge_cols = [c for c in df_basic.columns if c not in df_daily.columns or c in ["ts_code", "trade_date"]]
    df = pd.merge(df_daily, df_basic[merge_cols], on=["ts_code", "trade_date"], how="left")

    all_factors = []
    trade_dates = sorted(df["trade_date"].dt.strftime("%Y%m%d").unique())
    for date_str in trade_dates:
        pkl_path = FACTOR_CACHE / f"{date_str}.pkl"
        if pkl_path.exists():
            try:
                df_factor = pd.read_pickle(pkl_path)
                if df_factor is not None and not df_factor.empty:
                    rename_map = {
                        "macd_dif_qfq": "macd_dif", "macd_dea_qfq": "macd_dea", "macd_qfq": "macd",
                        "rsi_qfq_6": "rsi_6", "rsi_qfq_12": "rsi_12", "rsi_qfq_24": "rsi_24",
                        "kdj_k_qfq": "kdj_k", "kdj_d_qfq": "kdj_d", "kdj_qfq": "kdj_j",
                        "obv_qfq": "obv", "ema_qfq_5": "ema_5", "ema_qfq_10": "ema_10",
                        "ema_qfq_20": "ema_20", "ema_qfq_60": "ema_60",
                        "bias1_qfq": "bias_short", "bias2_qfq": "bias_mid", "bias3_qfq": "bias_long",
                        "ma_qfq_5": "ma5", "ma_qfq_10": "ma10", "ma_qfq_20": "ma_20d", "atr_qfq": "atr",
                    }
                    df_factor = df_factor.rename(columns=rename_map)
                    if "trade_date" in df_factor.columns:
                        df_factor["trade_date"] = pd.to_datetime(df_factor["trade_date"])
                    else:
                        df_factor["trade_date"] = pd.to_datetime(date_str)
                    factor_cols = [c for c in df_factor.columns if c not in df.columns or c in ["ts_code", "trade_date"]]
                    all_factors.append(df_factor[factor_cols])
            except Exception:
                pass

    if all_factors:
        df_factors = pd.concat(all_factors, ignore_index=True)
        factor_cols = [c for c in df_factors.columns if c not in df.columns or c in ["ts_code", "trade_date"]]
        df = pd.merge(df, df_factors[factor_cols], on=["ts_code", "trade_date"], how="left")

    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return df


def generate_predictions(data_start: str, data_end: str, pred_start: str, pred_end: str, output_dir: Path):
    df_raw = load_local_data(data_start, data_end)
    df_raw = df_raw.copy()
    df_raw["sample_id"] = df_raw.groupby("ts_code").ngroup()
    df_raw["name"] = ""
    df_raw["days_to_t1"] = 0

    provider = TushareDataProvider()
    df_market = provider.fetch_market_index(data_start, data_end)

    fe = FeatureEngineer()
    df_features = fe.compute_all_features(df_raw, df_market)

    predictor = CatBoostPredictor("v2.9.2-catboost-conservative")
    output_dir.mkdir(parents=True, exist_ok=True)

    mask = (df_features["trade_date"] >= pd.to_datetime(pred_start)) & (df_features["trade_date"] <= pd.to_datetime(pred_end))
    trade_dates = sorted(df_features[mask]["trade_date"].dt.strftime("%Y%m%d").unique())

    for date_str in trade_dates:
        pred_dt = pd.to_datetime(date_str)
        df_pred = df_features[df_features["trade_date"] == pred_dt].copy()
        if df_pred.empty:
            continue
        df_pred["prob"] = predictor.predict(df_pred)
        X = df_pred[[c for c in predictor.feature_names if c in df_pred.columns]].fillna(0).astype(float)
        df_pred["prob_raw"] = predictor.model.predict_proba(X)[:, 1]
        df_pred = df_pred.sort_values("prob_raw", ascending=False).reset_index(drop=True)
        df_pred["rank"] = range(1, len(df_pred) + 1)
        predictor.save_results(df_pred, date_str, output_dir)

    return trade_dates


def run_backtest(pred_dir: str, start_date: str, end_date: str):
    bt = RealisticBacktester(
        prediction_dir=pred_dir,
        initial_capital=10_000_000,
        per_stock_amount=300_000,
        top_n_buy=10,
        stop_loss_pct=4.0,
        trailing_stop_pct=5.0,
        trailing_stop_activation=5.0,
        enable_sector_filter=False,
        ma_window=5, ma_consecutive_days=2,
        buy_slippage_bps=15.0, sell_slippage_bps=20.0,
        commission_rate=0.00025, min_commission=5.0,
        stamp_duty_rate=0.001, min_amount=10_000,
    )
    return bt.run(start_date=start_date, end_date=end_date)


def main():
    quarters = [
        ("2024Q4", "20240630", "20241231", "20241001", "20241231"),
        ("2025Q1", "20241001", "20250331", "20250101", "20250331"),
        ("2026Q1", "20251001", "20260331", "20260101", "20260331"),
    ]

    for name, data_start, data_end, pred_start, pred_end in quarters:
        print(f"\n{'='*60}")
        print(f"  {name} 回测")
        print(f"  数据: {data_start} ~ {data_end}, 预测: {pred_start} ~ {pred_end}")
        print(f"{'='*60}")

        pred_dir = PROJECT_ROOT / "data" / "prediction" / f"v292_conservative_{name.lower()}"

        if not any(pred_dir.glob("predictions_*.csv")):
            print(f"生成预测中...")
            generate_predictions(data_start, data_end, pred_start, pred_end, pred_dir)
        else:
            print(f"使用已有预测: {pred_dir}")

        result = run_backtest(str(pred_dir), pred_start, pred_end)
        if result:
            txns = result.get("transactions", pd.DataFrame())
            sell_txns = txns[txns.get("action", "").str.contains("卖出", na=False)] if not txns.empty else pd.DataFrame()
            win_rate = (sell_txns["pnl"] > 0).mean() * 100 if not sell_txns.empty and "pnl" in sell_txns.columns else 0
            print(f"\n  ✅ 收益率: {result['total_return']:+.2f}%")
            print(f"     交易: {len(sell_txns)}次卖出, 胜率: {win_rate:.1f}%")
        else:
            print("  ❌ 回测失败")


if __name__ == "__main__":
    main()
