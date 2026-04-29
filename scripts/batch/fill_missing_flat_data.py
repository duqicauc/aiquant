#!/usr/bin/env python3
"""补全 quant_data.db flat 表缺失数据"""
import argparse
import os, sqlite3, sys, time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tushare_data_provider import TushareDataProvider
from src.data.arctic_provider import ArcticDataProvider
from src.utils.logger import log

DB_PATH = PROJECT_ROOT / "data" / "cache" / "quant_data.db"
ARCTIC_URI = os.getenv("ARCTICDB_URI", f"lmdb://{PROJECT_ROOT / 'data' / 'cache' / 'quant_data.arctic'}")
# 默认日期范围（可通过命令行覆盖）
DEFAULT_START_DATE = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
DEFAULT_END_DATE = datetime.now().strftime("%Y%m%d")
CHUNK_SIZE = 80  # SQLite 默认 max_variables=999, 12列 -> ~83行

DAILY_COLS = ["ts_code","trade_date","open","high","low","close","pre_close","change","pct_chg","vol","amount","update_time"]
BASIC_COLS = ["ts_code","trade_date","turnover_rate","volume_ratio","total_mv","circ_mv","pe","pb","update_time"]
FACTOR_COLS = ["ts_code","trade_date","macd_dif","macd_dea","macd","kdj_k","kdj_d","kdj_j","rsi_6","rsi_12","rsi_24","update_time"]
# ArcticDB 用全因子列（从 tushare_data_provider 动态获取）
from src.data.tushare_data_provider import STK_FACTOR_RENAME
ARCTIC_FACTOR_COLS = ["ts_code", "trade_date"] + list(STK_FACTOR_RENAME.values())

def get_existing_dates(conn, table, start_date: str):
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT trade_date FROM {table} WHERE trade_date >= ?", (start_date,))
    return {row[0] for row in cursor.fetchall()}

def chunked_insert(conn, df, table, cols):
    if df.empty: return 0
    placeholders = ",".join(["?"] * len(cols))
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
    rows = df[cols].values.tolist()
    total = 0
    for i in range(0, len(rows), CHUNK_SIZE):
        chunk = rows[i:i+CHUNK_SIZE]
        conn.executemany(sql, chunk)
        total += len(chunk)
    return total

def write_daily_data(conn, df, now):
    if df.empty: return 0
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
    df["update_time"] = now
    for c in DAILY_COLS:
        if c not in df.columns: df[c] = None
    return chunked_insert(conn, df, "daily_data", DAILY_COLS)

def write_daily_basic(conn, df, now):
    if df.empty: return 0
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
    df["update_time"] = now
    for c in BASIC_COLS:
        if c not in df.columns: df[c] = None
    return chunked_insert(conn, df, "daily_basic", BASIC_COLS)

def write_stk_factor(conn, df, now):
    if df.empty: return 0
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
    df["update_time"] = now
    for c in FACTOR_COLS:
        if c not in df.columns: df[c] = None
    return chunked_insert(conn, df, "stk_factor", FACTOR_COLS)

def write_to_arctic(arctic, df_daily, df_basic, df_factor):
    """同步写入 ArcticDB"""
    try:
        if not df_daily.empty:
            df = df_daily.copy()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values(["trade_date", "ts_code"])
            df.set_index("trade_date", inplace=True)
            arctic.append_daily_ohlcv(df)
    except Exception as e:
        log.warning(f"ArcticDB daily ohlcv 写入失败: {e}")

    try:
        if not df_basic.empty:
            df = df_basic.copy()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values(["trade_date", "ts_code"])
            df.set_index("trade_date", inplace=True)
            arctic.append_daily_basic(df)
    except Exception as e:
        log.warning(f"ArcticDB daily basic 写入失败: {e}")

    try:
        if not df_factor.empty:
            df = df_factor.copy()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.sort_values(["trade_date", "ts_code"])
            df.set_index("trade_date", inplace=True)
            arctic.append_daily_factors(df)
    except Exception as e:
        log.warning(f"ArcticDB daily factors 写入失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="补全 quant_data.db flat 表缺失数据（同步写入 ArcticDB）")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="结束日期 YYYYMMDD")
    parser.add_argument("--no-arctic", action="store_true", help="不同步写入 ArcticDB")
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date

    provider = TushareDataProvider()
    arctic = None if args.no_arctic else ArcticDataProvider(uri=ARCTIC_URI)
    trade_dates = provider.get_trade_dates(start_date, end_date)
    log.info(f"共有 {len(trade_dates)} 个交易日需要补全: {trade_dates[0]} ~ {trade_dates[-1]}")
    with sqlite3.connect(DB_PATH) as conn:
        existing = {
            "daily_data": get_existing_dates(conn, "daily_data", start_date),
            "daily_basic": get_existing_dates(conn, "daily_basic", start_date),
            "stk_factor": get_existing_dates(conn, "stk_factor", start_date),
        }
        for tbl, dates in existing.items():
            log.info(f"  {tbl} 已有 {len(dates)} 天")
        total = len(trade_dates)
        for i, date in enumerate(trade_dates, 1):
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df_daily, df_basic, df_factor = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            # daily_data
            if date not in existing["daily_data"]:
                try:
                    df_daily = provider.fetch_daily(date)
                    n = write_daily_data(conn, df_daily, now)
                    log.info(f"[{i}/{total}] {date} daily_data: {n} 条")
                except Exception as e:
                    log.error(f"[{i}/{total}] {date} daily_data 失败: {e}")
            else:
                log.info(f"[{i}/{total}] {date} daily_data: 跳过")

            # daily_basic
            if date not in existing["daily_basic"]:
                try:
                    df_basic = provider.fetch_daily_basic(date)
                    n = write_daily_basic(conn, df_basic, now)
                    log.info(f"[{i}/{total}] {date} daily_basic: {n} 条")
                except Exception as e:
                    log.error(f"[{i}/{total}] {date} daily_basic 失败: {e}")
            else:
                log.info(f"[{i}/{total}] {date} daily_basic: 跳过")

            # stk_factor
            if date not in existing["stk_factor"]:
                try:
                    df_factor = provider.fetch_stk_factor_pro(date)
                    n = write_stk_factor(conn, df_factor, now)
                    log.info(f"[{i}/{total}] {date} stk_factor: {n} 条")
                except Exception as e:
                    log.error(f"[{i}/{total}] {date} stk_factor 失败: {e}")
            else:
                log.info(f"[{i}/{total}] {date} stk_factor: 跳过")

            conn.commit()

            # 同步写入 ArcticDB
            if arctic:
                write_to_arctic(arctic, df_daily, df_basic, df_factor)

            if i < total:
                time.sleep(6)
        log.info("=" * 50)
        log.info("验证结果:")
        for tbl in ["daily_data", "daily_basic", "stk_factor"]:
            c = conn.cursor()
            c.execute(f"SELECT MAX(trade_date), COUNT(*) FROM {tbl} WHERE trade_date >= ?", (start_date,))
            md, cnt = c.fetchone()
            log.info(f"  {tbl}: max_date={md}, count={cnt}")
    log.success("完成")

if __name__ == "__main__":
    main()
