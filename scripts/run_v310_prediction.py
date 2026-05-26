#!/usr/bin/env python3
"""
v3.1.0 预测入口 — Breakout-only 生产输出

说明:
- 主输出(predictions_*.csv) 仅使用 Breakout 模型结果
- Bounce 模型结果单独存档到 v3.1.0/bounce_*.csv（研究用，不进入 daily）
- 原因: Bounce 2024回测负收益(夏普0.03)，不独立上线；等权融合会污染主输出

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

    # Breakout 预测（生产主模型）
    bo = BreakoutPredictor()
    df_bo = bo.predict_date(trade_date, top_k=None)
    log.info(f"Breakout 预测完成: {len(df_bo)} 只股票")

    # Bounce 预测（研究用，不进入主输出）
    bu = BouncePredictor()
    df_bu = bu.predict_date(trade_date, top_k=None)
    log.info(f"Bounce 预测完成: {len(df_bu)} 只股票")

    if df_bo.empty:
        log.error("Breakout 预测为空，无法生成主输出")
        return

    # ── 主输出: Breakout-only ──
    main_top = df_bo.head(top_k).copy()
    main_top = main_top.sort_values("prob_cal", ascending=False).reset_index(drop=True)
    main_top["rank"] = range(1, len(main_top) + 1)
    main_top["trade_date"] = trade_date
    main_top = main_top.rename(columns={"prob_cal": "prob"})

    # 保存 top_k 格式（兼容 v3.0.0的 predictions_YYYYMMDD_top50.csv）
    top_file = OUTPUT_DIR / f"predictions_{trade_date}_top{top_k}.csv"
    main_top[["ts_code", "trade_date", "prob", "rank"]].to_csv(top_file, index=False)
    log.success(f"保存 top{top_k}: {top_file}")

    # 保存 all 格式（兼容 v3.0.0的 predictions_YYYYMMDD_all.csv）
    all_df = df_bo.copy()
    all_df = all_df.rename(columns={"prob_cal": "prob"})
    all_file = OUTPUT_DIR / f"predictions_{trade_date}_all.csv"
    all_df[["ts_code", "trade_date", "prob", "rank"]].to_csv(all_file, index=False)
    log.success(f"保存 all: {all_file}")

    # ── 存档: 各模型原始结果 ──
    df_bo.head(top_k).to_csv(PREDICTION_DIR / f"breakout_{trade_date}_top{top_k}.csv", index=False)
    if not df_bu.empty:
        df_bu.head(top_k).to_csv(PREDICTION_DIR / f"bounce_{trade_date}_top{top_k}.csv", index=False)
    # 保留 fused 文件（空文件标记，避免旧融合结果被误用）
    fused_mark = pd.DataFrame({"note": ["Breakout-only since 2026-05-26. Bounce not fused."]})
    fused_mark.to_csv(PREDICTION_DIR / f"fused_equal_{trade_date}_top{top_k}.csv", index=False)

    log.success(f"v3.1.0 预测完成: {trade_date}")


if __name__ == "__main__":
    trade_date = get_last_trade_date()
    log.info(f"目标交易日: {trade_date}")
    run_v310_prediction(trade_date, top_k=50)
