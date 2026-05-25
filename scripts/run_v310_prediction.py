#!/usr/bin/env python3
"""
v3.1.0 预测入口 — 补充数据并生成最新交易日预测

Usage:
    cd /home/ubuntu/aiquant && venv/bin/python scripts/run_v310_prediction.py
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.models.breakout_predictor import BreakoutPredictor
from src.models.bounce_predictor import BouncePredictor
from src.data.tushare_data_provider import TushareDataProvider
from src.utils.logger import log

PREDICTION_DIR = project_root / "data" / "prediction" / "v3.1.0"
OUTPUT_DIR = project_root / "data" / "prediction" / "v3.1.0_daily"


def get_last_trade_date() -> str:
    """获取最新交易日（考虑休市）"""
    provider = TushareDataProvider()
    today = datetime.now().strftime("%Y%m%d")
    # 获取过去 10 个交易日
    dates = provider.get_trade_dates(
        (datetime.now() - timedelta(days=30)).strftime("%Y%m%d"),
        today
    )
    if not dates:
        return today
    return dates[-1]


def run_v310_prediction(trade_date: str, top_k: int = 50):
    """运行 v3.1.0 双模型预测，输出兼容格式"""
    log.info(f"运行 v3.1.0 预测: {trade_date}")

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Breakout 预测
    bo = BreakoutPredictor()
    df_bo = bo.predict_date(trade_date, top_k=None)
    log.info(f"Breakout 预测完成: {len(df_bo)} 只股票")

    # Bounce 预测
    bu = BouncePredictor()
    df_bu = bu.predict_date(trade_date, top_k=None)
    log.info(f"Bounce 预测完成: {len(df_bu)} 只股票")

    if df_bo.empty and df_bu.empty:
        log.error("两个模型预测均为空")
        return

    # 等权融合
    half_k = top_k // 2
    bo_top = df_bo.head(half_k)[["ts_code", "prob_cal"]].copy() if not df_bo.empty else pd.DataFrame()
    bu_top = df_bu.head(half_k)[["ts_code", "prob_cal"]].copy() if not df_bu.empty else pd.DataFrame()

    combined = pd.concat([bo_top, bu_top], ignore_index=True)
    combined = combined.sort_values("prob_cal", ascending=False).drop_duplicates("ts_code")

    if len(combined) < top_k:
        remaining_bo = df_bo[~df_bo["ts_code"].isin(combined["ts_code"])] if not df_bo.empty else pd.DataFrame()
        remaining_bu = df_bu[~df_bu["ts_code"].isin(combined["ts_code"])] if not df_bu.empty else pd.DataFrame()
        remaining = pd.concat([
            remaining_bo[["ts_code", "prob_cal"]],
            remaining_bu[["ts_code", "prob_cal"]]
        ]).sort_values("prob_cal", ascending=False)
        n_fill = top_k - len(combined)
        combined = pd.concat([combined, remaining.head(n_fill)], ignore_index=True)

    combined = combined.sort_values("prob_cal", ascending=False).head(top_k).reset_index(drop=True)
    combined["rank"] = range(1, len(combined) + 1)
    combined["trade_date"] = trade_date
    combined = combined.rename(columns={"prob_cal": "prob"})

    # 保存 top_k 格式（兼容 v3.0.0的 predictions_YYYYMMDD_top50.csv）
    top_file = OUTPUT_DIR / f"predictions_{trade_date}_top{top_k}.csv"
    combined[["ts_code", "trade_date", "prob", "rank"]].to_csv(top_file, index=False)
    log.success(f"保存 top{top_k}: {top_file}")

    # 保存 all 格式（兼容 v3.0.0的 predictions_YYYYMMDD_all.csv）
    # 使用 Breakout 的全市场结果作为 all 基础（因为它覆盖全市场）
    all_df = df_bo.copy() if not df_bo.empty else df_bu.copy()
    if not all_df.empty:
        all_df = all_df.rename(columns={"prob_cal": "prob"})
        all_file = OUTPUT_DIR / f"predictions_{trade_date}_all.csv"
        all_df[["ts_code", "trade_date", "prob", "rank"]].to_csv(all_file, index=False)
        log.success(f"保存 all: {all_file}")

    # 保存 v3.1.0 原始格式
    if not df_bo.empty:
        df_bo.head(top_k).to_csv(PREDICTION_DIR / f"breakout_{trade_date}_top{top_k}.csv", index=False)
    if not df_bu.empty:
        df_bu.head(top_k).to_csv(PREDICTION_DIR / f"bounce_{trade_date}_top{top_k}.csv", index=False)
    combined.to_csv(PREDICTION_DIR / f"fused_equal_{trade_date}_top{top_k}.csv", index=False)

    log.success(f"v3.1.0 预测完成: {trade_date}")


if __name__ == "__main__":
    trade_date = get_last_trade_date()
    log.info(f"目标交易日: {trade_date}")
    run_v310_prediction(trade_date, top_k=50)
