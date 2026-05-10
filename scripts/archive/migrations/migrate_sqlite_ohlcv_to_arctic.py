"""
将 SQLite daily_data 完整历史迁移到 ArcticDB daily/ohlcv。
策略：一次性读取全部数据 → 格式转换 → 覆盖写入 ArcticDB。
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.arctic_provider import ArcticDataProvider
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("migrate_ohlcv")


def migrate_all():
    db_path = PROJECT_ROOT / "data" / "cache" / "quant_data.db"
    conn = sqlite3.connect(db_path)

    log.info("开始从 SQLite 读取全部 daily_data...")
    df = pd.read_sql_query(
        "SELECT ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount FROM daily_data ORDER BY trade_date, ts_code",
        conn,
        parse_dates=["trade_date"],
    )
    conn.close()

    log.info(f"SQLite 读取完成: {len(df)} 行, {df['trade_date'].nunique()} 个交易日, {df['ts_code'].nunique()} 只股票")

    df = df.set_index("trade_date").sort_index()

    provider = ArcticDataProvider()
    lib = provider.get_library("daily")

    # 删除现有 ohlcv，全量重建
    if "ohlcv" in lib.list_symbols():
        lib.delete("ohlcv")
        log.info("已删除 ArcticDB 现有 ohlcv symbol")

    log.info("正在写入 ArcticDB...")
    lib.write("ohlcv", df)
    log.info(f"已写入 ArcticDB: {len(df)} 行")

    # 验证
    df_check = provider.read_daily_ohlcv("19990101", "20261231")
    log.info(
        f"ArcticDB ohlcv 验证: {len(df_check)} 行, {df_check.index.nunique()} 个交易日, "
        f"{df_check['ts_code'].nunique()} 只股票, {df_check.index.min()} ~ {df_check.index.max()}"
    )


if __name__ == "__main__":
    migrate_all()
