#!/usr/bin/env python3
"""补全 ArcticDB daily/factors 缺失数据（Batch 1 全因子）"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tushare_data_provider import TushareDataProvider, STK_FACTOR_RENAME
from src.data.arctic_provider import ArcticDataProvider
from src.utils.logger import log

ARCTIC_URI = os.getenv("ARCTICDB_URI", "lmdb:///Users/javaadu/Documents/GitHub/aiquant/data/cache/quant_data.arctic")
DEFAULT_START_DATE = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
DEFAULT_END_DATE = datetime.now().strftime("%Y%m%d")
FACTOR_COLS = ["ts_code", "trade_date"] + list(STK_FACTOR_RENAME.values())
BATCH_DAYS = 30  # 累积 30 天数据后一次性写入


def delete_factors_symbol(arctic):
    """删除旧的 factors symbol（用于强制重建）"""
    lib = arctic.get_library("daily")
    try:
        lib.delete("factors")
        log.info("已删除旧的 daily/factors")
    except Exception as e:
        log.info(f"删除 daily/factors 失败（可能不存在）: {e}")


def write_batch(arctic, batch_df: pd.DataFrame, is_first: bool) -> int:
    """将累积的 batch 数据写入 ArcticDB"""
    if batch_df.empty:
        return 0

    batch_df = batch_df.copy()
    batch_df["trade_date"] = pd.to_datetime(batch_df["trade_date"])
    batch_df = batch_df.sort_values(["trade_date", "ts_code"])
    batch_df.set_index("trade_date", inplace=True)

    lib = arctic.get_library("daily")
    if is_first:
        lib.write("factors", batch_df)
        log.info(f"首次写入 factors: {len(batch_df)} 行, {len(batch_df.columns)} 列")
    else:
        lib.append("factors", batch_df)
        log.info(f"追加 factors: {len(batch_df)} 行")

    return len(batch_df)


def main():
    parser = argparse.ArgumentParser(description="补全 ArcticDB daily/factors 缺失数据")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="结束日期 YYYYMMDD")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不写入")
    parser.add_argument("--force", action="store_true", help="强制重建，删除旧数据后重新获取")
    parser.add_argument("--sleep", type=int, default=6, help="每次请求间隔秒数")
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date

    provider = TushareDataProvider()
    arctic = ArcticDataProvider(uri=ARCTIC_URI)

    trade_dates = provider.get_trade_dates(start_date, end_date)
    log.info(f"共有 {len(trade_dates)} 个交易日需要补全: {trade_dates[0]} ~ {trade_dates[-1]}")

    if args.force and not args.dry_run:
        delete_factors_symbol(arctic)

    total = len(trade_dates)
    batch_data = []
    is_first = True
    total_written = 0

    for i, date in enumerate(trade_dates, 1):
        try:
            df = provider.fetch_stk_factor_pro(date)
            if df.empty:
                log.warning(f"[{i}/{total}] {date} stk_factor_pro: 返回空数据")
                continue

            # 确保列存在
            for c in FACTOR_COLS:
                if c not in df.columns:
                    df[c] = None

            batch_data.append(df[FACTOR_COLS])

            if args.dry_run:
                log.info(f"[{i}/{total}] {date} factors: {len(df)} 条 (预览模式)")
            else:
                log.info(f"[{i}/{total}] {date} factors: {len(df)} 条 (已累积 {len(batch_data)} 天)")
        except Exception as e:
            log.error(f"[{i}/{total}] {date} factors 失败: {e}")

        # 每 BATCH_DAYS 天写入一次，或最后一批写入
        should_write = len(batch_data) >= BATCH_DAYS or (i == total and batch_data)

        if should_write and not args.dry_run:
            batch_df = pd.concat(batch_data, ignore_index=True)
            n = write_batch(arctic, batch_df, is_first)
            total_written += n
            batch_data = []
            is_first = False

        if i < total and args.sleep > 0:
            time.sleep(args.sleep)

    log.info(f"=" * 50)
    log.info(f"总计写入: {total_written:,} 行")
    log.success("完成")


if __name__ == "__main__":
    main()
