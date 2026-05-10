"""
将 SQLite daily_basic / stk_factor 完整历史迁移到 ArcticDB。
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
log = logging.getLogger("migrate_basic_factor")


def migrate_table(sqlite_table: str, arctic_symbol: str, columns: list):
    db_path = PROJECT_ROOT / "data" / "cache" / "quant_data.db"
    conn = sqlite3.connect(db_path)

    cols_sql = ",".join(columns)
    log.info(f"开始从 SQLite 读取 {sqlite_table}...")
    df = pd.read_sql_query(
        f"SELECT {cols_sql} FROM {sqlite_table} ORDER BY trade_date, ts_code",
        conn,
        parse_dates=["trade_date"],
    )
    conn.close()

    log.info(
        f"SQLite 读取完成: {len(df)} 行, {df['trade_date'].nunique()} 个交易日, "
        f"{df['ts_code'].nunique()} 只股票"
    )

    df = df.set_index("trade_date").sort_index()

    provider = ArcticDataProvider()
    lib = provider.get_library("daily")

    # 删除现有 symbol，全量重建
    if arctic_symbol in lib.list_symbols():
        lib.delete(arctic_symbol)
        log.info(f"已删除 ArcticDB 现有 {arctic_symbol} symbol")

    log.info(f"正在写入 ArcticDB {arctic_symbol}...")
    lib.write(arctic_symbol, df)
    log.info(f"已写入 ArcticDB: {len(df)} 行")

    # 验证
    if arctic_symbol == "basic":
        df_check = provider.read_daily_basic("19990101", "20261231")
    else:
        df_check = provider.read_daily_factors("19990101", "20261231")

    log.info(
        f"ArcticDB {arctic_symbol} 验证: {len(df_check)} 行, {df_check.index.nunique()} 个交易日, "
        f"{df_check['ts_code'].nunique()} 只股票, {df_check.index.min()} ~ {df_check.index.max()}"
    )


def main():
    migrate_table(
        "daily_basic",
        "basic",
        ["ts_code", "trade_date", "turnover_rate", "volume_ratio", "total_mv", "circ_mv", "pe", "pb"],
    )
    migrate_table(
        "stk_factor",
        "factors",
        ["ts_code", "trade_date", "macd_dif", "macd_dea", "macd", "kdj_k", "kdj_d", "kdj_j", "rsi_6", "rsi_12", "rsi_24"],
    )


if __name__ == "__main__":
    main()
