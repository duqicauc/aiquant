#!/usr/bin/env python3
"""每日 ArcticDB vs SQLite 数据一致性验证"""
import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.arctic_provider import ArcticDataProvider
from src.utils.logger import log

DB_PATH = PROJECT_ROOT / "data" / "cache" / "quant_data.db"
ARCTIC_URI = f"lmdb://{PROJECT_ROOT / 'data' / 'cache' / 'quant_data.arctic'}"

TABLES = {
    "daily_data": {
        "arctic_lib": "daily",
        "arctic_sym": "ohlcv",
        "sqlite_table": "daily_data",
        "key_cols": ["ts_code", "trade_date"],
        "value_cols": ["open", "high", "low", "close", "vol", "amount", "pct_chg"],
        "tolerance": 1e-6,
    },
    "daily_basic": {
        "arctic_lib": "daily",
        "arctic_sym": "basic",
        "sqlite_table": "daily_basic",
        "key_cols": ["ts_code", "trade_date"],
        "value_cols": ["turnover_rate", "volume_ratio", "pe", "pb", "total_mv", "circ_mv"],
        "tolerance": 1e-6,
    },
    "stk_factor": {
        "arctic_lib": "daily",
        "arctic_sym": "factors",
        "sqlite_table": "stk_factor",
        "key_cols": ["ts_code", "trade_date"],
        "value_cols": ["macd_dif", "macd", "rsi_6", "kdj_k", "kdj_j"],
        "tolerance": 0.5,  # stk_factor_pro 与旧 stk_factor 接口计算可能存在差异
    },
}


def read_sqlite_day(table, date):
    conn = sqlite3.connect(str(DB_PATH))
    query = f"SELECT * FROM {table} WHERE trade_date = ?"
    df = pd.read_sql_query(query, conn, params=(date,))
    conn.close()
    if not df.empty and "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def read_arctic_day(arctic, lib_name, sym, date):
    lib = arctic.get_library(lib_name)
    try:
        df = lib.read(sym, date_range=(pd.to_datetime(date), pd.to_datetime(date))).data
        if not df.empty and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        return df
    except Exception as e:
        log.warning(f"读取 ArcticDB {lib_name}/{sym} {date} 失败: {e}")
        return pd.DataFrame()


def compare_table(arctic, cfg, date, tolerance=1e-6):
    df_sqlite = read_sqlite_day(cfg["sqlite_table"], date)
    df_arctic = read_arctic_day(arctic, cfg["arctic_lib"], cfg["arctic_sym"], date)

    table_name = cfg["sqlite_table"]
    key_cols = cfg["key_cols"]
    value_cols = [c for c in cfg["value_cols"] if c in df_sqlite.columns and c in df_arctic.columns]

    if df_sqlite.empty and df_arctic.empty:
        return {"status": "both_empty", "pass": True}
    if df_sqlite.empty:
        return {"status": "sqlite_empty", "pass": False, "arctic_rows": len(df_arctic)}
    if df_arctic.empty:
        return {"status": "arctic_empty", "pass": False, "sqlite_rows": len(df_sqlite)}

    # 标准化 key
    for col in key_cols:
        if col in df_sqlite.columns:
            df_sqlite[col] = df_sqlite[col].astype(str)
        if col in df_arctic.columns:
            df_arctic[col] = df_arctic[col].astype(str)

    merged = pd.merge(
        df_sqlite[key_cols + value_cols],
        df_arctic[key_cols + value_cols],
        on=key_cols,
        suffixes=("_sqlite", "_arctic"),
        how="outer",
        indicator=True,
    )

    matched = merged[merged["_merge"] == "both"]
    only_sqlite = merged[merged["_merge"] == "left_only"]
    only_arctic = merged[merged["_merge"] == "right_only"]

    max_err = 0.0
    for col in value_cols:
        s_col = f"{col}_sqlite"
        a_col = f"{col}_arctic"
        if s_col not in matched.columns or a_col not in matched.columns:
            continue
        s = pd.to_numeric(matched[s_col], errors="coerce")
        a = pd.to_numeric(matched[a_col], errors="coerce")
        valid = s.notna() & a.notna()
        if valid.sum() == 0:
            continue
        diff = np.abs(s[valid] - a[valid])
        base = np.maximum(np.abs(s[valid]), np.abs(a[valid]))
        rel_err = (diff / np.maximum(base, 1e-12)).max()
        max_err = max(max_err, rel_err)

    # factor 表允许 ArcticDB 数据更完整（only_arctic > 0 可接受）
    # 但 SQLite 中不应该有 ArcticDB 缺失的数据（only_sqlite 必须为 0）
    # 且已匹配数据的误差必须在容忍度内
    passed = max_err <= tolerance and len(only_sqlite) == 0

    return {
        "status": "compared",
        "pass": passed,
        "sqlite_rows": len(df_sqlite),
        "arctic_rows": len(df_arctic),
        "matched": len(matched),
        "only_sqlite": len(only_sqlite),
        "only_arctic": len(only_arctic),
        "max_rel_err": float(max_err),
    }


def main():
    parser = argparse.ArgumentParser(description="每日 ArcticDB vs SQLite 数据一致性验证")
    parser.add_argument("--date", help="验证日期 YYYYMMDD（默认昨天）")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="最大相对误差容忍度")
    parser.add_argument("--notify", action="store_true", help="失败时输出告警信息")
    args = parser.parse_args()

    date = args.date or (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    arctic = ArcticDataProvider(uri=ARCTIC_URI)

    log.info(f"=" * 60)
    log.info(f"每日验证: {date}")
    log.info(f"=" * 60)

    all_pass = True
    for name, cfg in TABLES.items():
        tol = cfg.get("tolerance", args.tolerance)
        result = compare_table(arctic, cfg, date, tol)
        if result["status"] == "compared":
            status = "✅ PASS" if result["pass"] else "❌ FAIL"
            log.info(
                f"{name}: {status} | SQLite={result['sqlite_rows']} Arctic={result['arctic_rows']} "
                f"matched={result['matched']} only_sqlite={result['only_sqlite']} "
                f"only_arctic={result['only_arctic']} max_err={result['max_rel_err']:.2e} (tol={tol:.0e})"
            )
            if not result["pass"]:
                all_pass = False
        else:
            log.warning(f"{name}: {result['status']}")
            all_pass = False

    log.info(f"=" * 60)
    if all_pass:
        log.success(f"验证通过: {date}")
        sys.exit(0)
    else:
        log.error(f"验证失败: {date}")
        if args.notify:
            print(f"[ALERT] ArcticDB validation failed for {date}")
        sys.exit(1)


if __name__ == "__main__":
    main()
