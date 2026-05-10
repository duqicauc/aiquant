#!/usr/bin/env python3
"""
补全 ArcticDB 2026年 Q1-Q2 数据

从 Tushare 获取缺失的 daily_basic 和 stk_factor_pro 数据，写入 ArcticDB。
OHLCV 数据已较完整，主要补全 factors 和 basic。

Usage:
    python scripts/fill_arctic_data_2026q1.py
"""
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import tushare as ts
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.arctic_provider import ArcticDataProvider
from src.data.tushare_data_provider import STK_FACTOR_RENAME
from src.utils.logger import log

load_dotenv()
TOKEN = os.getenv("TUSHARE_TOKEN")
if TOKEN:
    ts.set_token(TOKEN)
PRO = ts.pro_api(TOKEN)

START_DATE = "20260105"
END_DATE = "20260508"


def get_trade_dates() -> list:
    """获取交易日列表"""
    df = PRO.trade_cal(start_date=START_DATE, end_date=END_DATE, is_open=1)
    return sorted(df["cal_date"].tolist())


def fetch_and_append_factors(provider: ArcticDataProvider, trade_date: str):
    """获取单日 factors 并追加到 ArcticDB"""
    try:
        df = PRO.stk_factor_pro(trade_date=trade_date)
        if df is None or df.empty:
            log.warning(f"  {trade_date}: stk_factor_pro 无数据")
            return 0

        rename_map = {k: v for k, v in STK_FACTOR_RENAME.items() if k in df.columns}
        cols = ["ts_code", "trade_date"] + list(rename_map.keys())
        df = df[[c for c in cols if c in df.columns]].copy()
        df = df.rename(columns=rename_map)
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        provider.append_daily_factors(df)
        log.info(f"  {trade_date}: factors {len(df)} 行写入 ArcticDB")
        return len(df)
    except Exception as e:
        log.error(f"  {trade_date}: factors 获取失败: {e}")
        return 0


def fetch_and_append_basic(provider: ArcticDataProvider, trade_date: str):
    """获取单日 daily_basic 并追加到 ArcticDB"""
    try:
        df = PRO.daily_basic(trade_date=trade_date)
        if df is None or df.empty:
            log.warning(f"  {trade_date}: daily_basic 无数据")
            return 0

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        keep = ["ts_code", "trade_date", "turnover_rate", "turnover_rate_f",
                "volume_ratio", "total_mv", "circ_mv", "pe", "pb"]
        df = df[[c for c in keep if c in df.columns]].copy()

        provider.append_daily_basic(df)
        log.info(f"  {trade_date}: basic {len(df)} 行写入 ArcticDB")
        return len(df)
    except Exception as e:
        log.error(f"  {trade_date}: basic 获取失败: {e}")
        return 0


def main():
    log.info(f"{'='*60}")
    log.info(f"ArcticDB 数据补全: {START_DATE} ~ {END_DATE}")
    log.info(f"{'='*60}")

    provider = ArcticDataProvider()
    trade_dates = get_trade_dates()
    log.info(f"共 {len(trade_dates)} 个交易日需要补全")

    total_factors = 0
    total_basic = 0

    for i, d in enumerate(trade_dates, 1):
        log.info(f"[{i}/{len(trade_dates)}] 处理 {d}...")
        total_factors += fetch_and_append_factors(provider, d)
        total_basic += fetch_and_append_basic(provider, d)
        time.sleep(0.3)  # Tushare API 限速保护

    log.success(f"补全完成! factors: {total_factors} 行, basic: {total_basic} 行")


if __name__ == "__main__":
    main()
