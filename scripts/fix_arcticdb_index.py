#!/usr/bin/env python3
"""
修复 ArcticDB 中 factors / basic / index_daily 的索引问题

问题：factors 和 basic 写入时没有将 trade_date 设为索引，
      导致 read_daily_factors / read_daily_basic 的 date_range filter 失败。

修复：读取 -> 设置 trade_date 为 DatetimeIndex -> 重新写入

Usage:
    python scripts/fix_arcticdb_index.py
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.arctic_provider import ArcticDataProvider
from src.utils.logger import log


def fix_daily_symbol(provider: ArcticDataProvider, symbol: str):
    """修复 daily library 中指定 symbol 的索引"""
    lib = provider.get_library("daily")

    log.info(f"修复 daily/{symbol}...")
    try:
        df = lib.read(symbol).data
        log.info(f"  读取到 {len(df)} 行, 索引类型: {type(df.index).__name__}")
    except Exception as e:
        log.error(f"  读取失败: {e}")
        return

    if isinstance(df.index, pd.DatetimeIndex):
        log.info(f"  索引已经是 DatetimeIndex，无需修复")
        return

    if "trade_date" not in df.columns:
        log.error(f"  缺少 trade_date 列，无法修复")
        return

    # 设置 trade_date 为 DatetimeIndex
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date").sort_index()

    # 重新写入（覆盖）
    lib.write(symbol, df)
    log.success(f"  已修复并重新写入: {len(df)} 行, 索引: {type(df.index).__name__}")


def fix_market_index(provider: ArcticDataProvider):
    """修复 market/index_daily"""
    lib = provider.get_library("market")

    log.info("修复 market/index_daily...")
    try:
        df = lib.read("index_daily").data
        log.info(f"  读取到 {len(df)} 行")
    except Exception as e:
        log.error(f"  读取失败: {e}")
        return

    if "trade_date" not in df.columns:
        log.error(f"  缺少 trade_date 列")
        return

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date").sort_index()
    lib.write("index_daily", df)
    log.success(f"  已修复: {len(df)} 行")


def main():
    log.info(f"{'='*60}")
    log.info("ArcticDB 索引修复")
    log.info(f"{'='*60}")

    provider = ArcticDataProvider()

    fix_daily_symbol(provider, "factors")
    fix_daily_symbol(provider, "basic")
    fix_market_index(provider)

    log.success("全部修复完成!")


if __name__ == "__main__":
    main()
