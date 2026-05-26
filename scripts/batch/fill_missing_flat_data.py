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

def get_existing_dates(conn, table, start_date: str, arctic: ArcticDataProvider = None):
    """获取已存在的日期。优先查 ArcticDB，失败回退 SQLite。"""
    if arctic is not None:
        try:
            if table == "daily_data":
                df = arctic.read_daily_ohlcv(start_date, "20991231")
            elif table == "daily_basic":
                df = arctic.read_daily_basic(start_date, "20991231")
            elif table == "stk_factor":
                df = arctic.read_daily_factors(start_date, "20991231")
            else:
                df = pd.DataFrame()
            if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                return set(df.index.strftime("%Y%m%d"))
            return set()
        except Exception:
            pass
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT DISTINCT trade_date FROM {table} WHERE trade_date >= ?", (start_date,))
        return {row[0] for row in cursor.fetchall()}
    except sqlite3.OperationalError:
        return set()

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
    parser = argparse.ArgumentParser(description="补全行情数据（daily_data 写入 ArcticDB，basic/factor 写入 SQLite + ArcticDB）")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="结束日期 YYYYMMDD")
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date

    provider = TushareDataProvider()
    arctic = ArcticDataProvider(uri=ARCTIC_URI)
    trade_dates = provider.get_trade_dates(start_date, end_date)
    log.info(f"共有 {len(trade_dates)} 个交易日需要补全: {trade_dates[0]} ~ {trade_dates[-1]}")
    with sqlite3.connect(DB_PATH) as conn:
        existing = {
            "daily_data": get_existing_dates(conn, "daily_data", start_date, arctic),
            "daily_basic": get_existing_dates(conn, "daily_basic", start_date),
            "stk_factor": get_existing_dates(conn, "stk_factor", start_date),
        }
        for tbl, dates in existing.items():
            log.info(f"  {tbl} 已有 {len(dates)} 天")
        total = len(trade_dates)
        for i, date in enumerate(trade_dates, 1):
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df_daily, df_basic, df_factor = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            # daily_data → 仅写入 ArcticDB（不再写 SQLite）
            if date not in existing["daily_data"]:
                try:
                    df_daily = provider.fetch_daily(date)
                    log.info(f"[{i}/{total}] {date} daily_data: 从 Tushare 获取 {len(df_daily)} 条 → ArcticDB")
                except Exception as e:
                    log.error(f"[{i}/{total}] {date} daily_data 失败: {e}")
            else:
                log.info(f"[{i}/{total}] {date} daily_data: 跳过（ArcticDB 已存在）")

            # ── 第一步：统一从 Tushare fetch（不受 SQLite 影响）───────────────
            if date not in existing["daily_basic"]:
                try:
                    df_basic = provider.fetch_daily_basic(date)
                except Exception as e:
                    log.warning(f"[{i}/{total}] {date} daily_basic fetch 失败，跳过: {e}")
                    df_basic = pd.DataFrame()
            else:
                df_basic = pd.DataFrame()   # 已有，跳过 fetch

            if date not in existing["stk_factor"]:
                try:
                    df_factor = provider.fetch_stk_factor_pro(date)
                except Exception as e:
                    log.warning(f"[{i}/{total}] {date} stk_factor fetch 失败，跳过: {e}")
                    df_factor = pd.DataFrame()
            else:
                df_factor = pd.DataFrame()  # 已有，跳过 fetch

            # ── 第二步：写 SQLite（失败不影响 ArcticDB）─────────────────────
            if not df_basic.empty:
                try:
                    n = write_daily_basic(conn, df_basic, now)
                    log.info(f"[{i}/{total}] {date} daily_basic: {n} 条 → SQLite ✓")
                except Exception as e:
                    log.warning(f"[{i}/{total}] {date} daily_basic → SQLite 失败（表可能不存在）: {e}")
            elif date in existing["daily_basic"]:
                log.info(f"[{i}/{total}] {date} daily_basic: 跳过（已有）")

            if not df_factor.empty:
                try:
                    n = write_stk_factor(conn, df_factor, now)
                    log.info(f"[{i}/{total}] {date} stk_factor: {n} 条 → SQLite ✓")
                except Exception as e:
                    log.warning(f"[{i}/{total}] {date} stk_factor → SQLite 失败（表可能不存在）: {e}")
            elif date in existing["stk_factor"]:
                log.info(f"[{i}/{total}] {date} stk_factor: 跳过（已有）")

            conn.commit()

            # ── 第三步：统一写入 ArcticDB（与 SQLite 成败无关）─────────────
            write_to_arctic(arctic, df_daily, df_basic, df_factor)
            if not df_daily.empty:
                log.info(f"[{i}/{total}] {date} daily_data: {len(df_daily)} 条 → ArcticDB ✓")
            if not df_basic.empty:
                log.info(f"[{i}/{total}] {date} daily_basic: {len(df_basic)} 条 → ArcticDB ✓")
            if not df_factor.empty:
                log.info(f"[{i}/{total}] {date} stk_factor: {len(df_factor)} 条 → ArcticDB ✓")

            if i < total:
                time.sleep(6)
        log.info("=" * 50)
        log.info("验证结果:")
        # ArcticDB daily_data
        try:
            df_ohlcv = arctic.read_daily_ohlcv(start_date, "20991231")
            log.info(f"  ArcticDB daily/ohlcv: {len(df_ohlcv)} 行, {df_ohlcv.index.nunique()} 天")
        except Exception as e:
            log.warning(f"  ArcticDB daily/ohlcv 验证失败: {e}")
        for tbl in ["daily_basic", "stk_factor"]:
            try:
                c = conn.cursor()
                c.execute(f"SELECT MAX(trade_date), COUNT(*) FROM {tbl} WHERE trade_date >= ?", (start_date,))
                md, cnt = c.fetchone()
                log.info(f"  {tbl}: max_date={md}, count={cnt}")
            except sqlite3.OperationalError:
                log.warning(f"  {tbl}: SQLite 表不存在，跳过验证（项目已迁移到 ArcticDB）")
    log.success("完成")

if __name__ == "__main__":
    main()
