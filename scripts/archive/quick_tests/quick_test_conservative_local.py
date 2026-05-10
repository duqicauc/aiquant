#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从本地数据库读取数据，快速测试保守版模型"""

import sys
import pickle
import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np

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
    """从本地数据库读取数据"""
    conn = sqlite3.connect(DB_PATH)

    # 读取行情数据
    df_daily = pd.read_sql_query(
        f"SELECT * FROM daily_data WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'",
        conn,
    )
    df_daily["trade_date"] = pd.to_datetime(df_daily["trade_date"])

    # 读取指标数据
    df_basic = pd.read_sql_query(
        f"SELECT * FROM daily_basic WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'",
        conn,
    )
    df_basic["trade_date"] = pd.to_datetime(df_basic["trade_date"])

    conn.close()

    # 合并
    merge_cols = [c for c in df_basic.columns if c not in df_daily.columns or c in ["ts_code", "trade_date"]]
    df = pd.merge(df_daily, df_basic[merge_cols], on=["ts_code", "trade_date"], how="left")

    # 读取技术因子
    all_factors = []
    trade_dates = sorted(df["trade_date"].dt.strftime("%Y%m%d").unique())
    for date_str in trade_dates:
        pkl_path = FACTOR_CACHE / f"{date_str}.pkl"
        if pkl_path.exists():
            try:
                df_factor = pd.read_pickle(pkl_path)
                if df_factor is not None and not df_factor.empty:
                    # 重命名列
                    rename_map = {
                        "macd_dif_qfq": "macd_dif",
                        "macd_dea_qfq": "macd_dea",
                        "macd_qfq": "macd",
                        "rsi_qfq_6": "rsi_6",
                        "rsi_qfq_12": "rsi_12",
                        "rsi_qfq_24": "rsi_24",
                        "kdj_k_qfq": "kdj_k",
                        "kdj_d_qfq": "kdj_d",
                        "kdj_qfq": "kdj_j",
                        "obv_qfq": "obv",
                        "ema_qfq_5": "ema_5",
                        "ema_qfq_10": "ema_10",
                        "ema_qfq_20": "ema_20",
                        "ema_qfq_60": "ema_60",
                        "bias1_qfq": "bias_short",
                        "bias2_qfq": "bias_mid",
                        "bias3_qfq": "bias_long",
                        "ma_qfq_5": "ma5",
                        "ma_qfq_10": "ma10",
                        "ma_qfq_20": "ma_20d",
                        "atr_qfq": "atr",
                    }
                    df_factor = df_factor.rename(columns=rename_map)
                    if "trade_date" in df_factor.columns:
                        df_factor["trade_date"] = pd.to_datetime(df_factor["trade_date"])
                    else:
                        df_factor["trade_date"] = pd.to_datetime(date_str)
                    factor_cols = [c for c in df_factor.columns if c not in df.columns or c in ["ts_code", "trade_date"]]
                    all_factors.append(df_factor[factor_cols])
            except Exception as e:
                log.warning(f"读取 {date_str} 因子缓存失败: {e}")

    if all_factors:
        df_factors = pd.concat(all_factors, ignore_index=True)
        factor_cols = [c for c in df_factors.columns if c not in df.columns or c in ["ts_code", "trade_date"]]
        df = pd.merge(df, df_factors[factor_cols], on=["ts_code", "trade_date"], how="left")

    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    log.success(f"本地数据加载完成: {len(df)} 行, {len(df.columns)} 列, {df['ts_code'].nunique()} 只股票")
    return df


def main():
    # 1. 加载本地数据
    print("=== 加载本地数据 ===")
    df_raw = load_local_data("20240630", "20241231")

    # 2. 准备特征工程
    df_raw = df_raw.copy()
    df_raw["sample_id"] = df_raw.groupby("ts_code").ngroup()
    df_raw["name"] = ""
    df_raw["days_to_t1"] = 0

    # 3. 获取市场环境数据（从 Tushare，少量请求）
    print("\n=== 获取市场环境数据 ===")
    provider = TushareDataProvider()
    df_market = provider.fetch_market_index("20240630", "20241231")

    # 4. 计算特征
    print("\n=== 计算特征 ===")
    fe = FeatureEngineer()
    df_features = fe.compute_all_features(df_raw, df_market)

    # 5. 预测
    print("\n=== 保守版预测 ===")
    predictor = CatBoostPredictor("v2.9.2-catboost-conservative")

    pred_dir = PROJECT_ROOT / "data" / "prediction" / "v292_conservative_stk_factor"
    pred_dir.mkdir(parents=True, exist_ok=True)

    trade_dates = sorted(df_features[df_features["trade_date"] >= pd.to_datetime("20241001")]["trade_date"].dt.strftime("%Y%m%d").unique())
    print(f"预测 {len(trade_dates)} 个交易日")

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

        predictor.save_results(df_pred, date_str, pred_dir)

        if date_str == "20241008":
            top10 = df_pred.head(10)
            print(f"\n{date_str} Top10 prob_raw: mean={top10['prob_raw'].mean():.6f}, std={top10['prob_raw'].std():.6f}")
            print(f"{date_str} Top10 prob_cal: mean={top10['prob'].mean():.6f}, std={top10['prob'].std():.6f}")
            print(f"Top10 市值均值: {top10['total_mv'].mean():.1f}亿")

    print(f"\n预测完成，保存到 {pred_dir}")

    # 6. 跑 realistic 回测
    print("\n=== 保守版 realistic 回测 (2024Q4) ===")
    bt = RealisticBacktester(
        prediction_dir=str(pred_dir),
        initial_capital=10_000_000,
        per_stock_amount=300_000,
        top_n_buy=10,
        stop_loss_pct=4.0,
        trailing_stop_pct=5.0,
        trailing_stop_activation=5.0,
        enable_sector_filter=False,
        ma_window=5,
        ma_consecutive_days=2,
        buy_slippage_bps=15.0,
        sell_slippage_bps=20.0,
        commission_rate=0.00025,
        min_commission=5.0,
        stamp_duty_rate=0.001,
        min_amount=10_000,
    )

    result = bt.run(start_date="20241001", end_date="20241231")
    if result:
        print(f"\n结果: 初始资金={result['initial_capital']:,.0f}, 最终资金={result['final_capital']:,.0f}, 收益率={result['total_return']:.2f}%")
        print(f"交易次数: {result['total_trades']}, 胜率: {result.get('win_rate', 0):.1f}%")


if __name__ == "__main__":
    main()
