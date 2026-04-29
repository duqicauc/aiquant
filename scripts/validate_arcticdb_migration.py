#!/usr/bin/env python3
"""验证 ArcticDB 迁移数据与 SQLite 的一致性"""
import argparse
import random
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.arctic_provider import ArcticDataProvider
from src.utils.logger import log

DB_PATH = PROJECT_ROOT / "data" / "cache" / "quant_data.db"
ARCTIC_URI = f"lmdb://{PROJECT_ROOT / 'data' / 'cache' / 'quant_data.arctic'}"

# 各表的核心数值列（用于对比）
TABLE_CONFIG = {
    "daily_data": {
        "symbol": "ohlcv",
        "lib": "daily",
        "key_cols": ["ts_code", "trade_date"],
        "value_cols": ["open", "high", "low", "close", "vol", "amount", "pct_chg"],
        "date_col": "trade_date",
    },
    "daily_basic": {
        "symbol": "basic",
        "lib": "daily",
        "key_cols": ["ts_code", "trade_date"],
        "value_cols": ["turnover_rate", "volume_ratio", "pe", "pb", "total_mv", "circ_mv"],
        "date_col": "trade_date",
    },
    "stk_factor": {
        "symbol": "factors",
        "lib": "daily",
        "key_cols": ["ts_code", "trade_date"],
        "value_cols": ["macd_dif", "macd_dea", "macd", "kdj_k", "kdj_d", "kdj_j", "rsi_6", "rsi_12", "rsi_24"],
        "date_col": "trade_date",
    },
    "weekly_data": {
        "symbol": "ohlcv",
        "lib": "weekly",
        "key_cols": ["ts_code", "trade_date"],
        "value_cols": ["open", "high", "low", "close", "vol", "amount", "pct_chg"],
        "date_col": "trade_date",
    },
}


def get_sqlite_dates(conn, table, limit=100):
    """获取 SQLite 中的日期列表"""
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT {TABLE_CONFIG[table]['date_col']} FROM {table} ORDER BY 1")
    dates = [row[0] for row in cursor.fetchall()]
    return dates


def sample_dates(dates, n=10):
    """随机采样日期"""
    if len(dates) <= n:
        return dates
    # 确保包含首尾
    samples = [dates[0], dates[-1]]
    remaining = dates[1:-1]
    if len(remaining) >= n - 2:
        samples.extend(random.sample(remaining, n - 2))
    else:
        samples.extend(remaining)
    return sorted(samples)


def read_sqlite_day(conn, table, date):
    """读取 SQLite 单日数据"""
    cfg = TABLE_CONFIG[table]
    date_str = date if isinstance(date, str) else date.strftime("%Y%m%d")
    query = f"SELECT * FROM {table} WHERE {cfg['date_col']} = ?"
    df = pd.read_sql_query(query, conn, params=(date_str,))
    if not df.empty and cfg["date_col"] in df.columns:
        df[cfg["date_col"]] = pd.to_datetime(df[cfg["date_col"]])
    return df


def read_arctic_day(arctic, table, date):
    """读取 ArcticDB 单日数据"""
    cfg = TABLE_CONFIG[table]
    lib = arctic.get_library(cfg["lib"])
    try:
        df = lib.read(cfg["symbol"], date_range=(pd.to_datetime(date), pd.to_datetime(date))).data
        if not df.empty:
            df = df.reset_index()
            if "index" in df.columns:
                df.rename(columns={"index": cfg["date_col"]}, inplace=True)
        return df
    except Exception as e:
        log.warning(f"读取 ArcticDB {cfg['lib']}/{cfg['symbol']} {date} 失败: {e}")
        return pd.DataFrame()


def compare_dataframes(df_sqlite, df_arctic, table):
    """对比两个 DataFrame 的数据一致性"""
    cfg = TABLE_CONFIG[table]
    key_cols = cfg["key_cols"]
    value_cols = [c for c in cfg["value_cols"] if c in df_sqlite.columns and c in df_arctic.columns]

    if df_sqlite.empty and df_arctic.empty:
        return {"status": "both_empty", "sqlite_rows": 0, "arctic_rows": 0}

    if df_sqlite.empty:
        return {"status": "sqlite_empty", "sqlite_rows": 0, "arctic_rows": len(df_arctic)}

    if df_arctic.empty:
        return {"status": "arctic_empty", "sqlite_rows": len(df_sqlite), "arctic_rows": 0}

    # 标准化 key 列类型
    for col in key_cols:
        if col in df_sqlite.columns:
            df_sqlite[col] = df_sqlite[col].astype(str)
        if col in df_arctic.columns:
            df_arctic[col] = df_arctic[col].astype(str)

    # 合并对比
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

    col_errors = {}
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
        rel_err = diff / np.maximum(base, 1e-12)
        max_rel_err = rel_err.max()
        mean_rel_err = rel_err.mean()
        col_errors[col] = {
            "max_rel_err": float(max_rel_err),
            "mean_rel_err": float(mean_rel_err),
            "match_count": int(valid.sum()),
        }

    return {
        "status": "compared",
        "sqlite_rows": len(df_sqlite),
        "arctic_rows": len(df_arctic),
        "matched_rows": len(matched),
        "only_sqlite": len(only_sqlite),
        "only_arctic": len(only_arctic),
        "column_errors": col_errors,
    }


def validate_table(arctic, conn, table, sample_days=5):
    """验证单个表"""
    log.info(f"\n{'='*60}")
    log.info(f"验证表: {table}")

    dates = get_sqlite_dates(conn, table)
    if not dates:
        log.warning(f"SQLite {table} 无数据")
        return None

    log.info(f"SQLite 总天数: {len(dates)}, 范围: {dates[0]} ~ {dates[-1]}")

    sample = sample_dates(dates, sample_days)
    log.info(f"采样日期: {sample}")

    results = []
    for date in sample:
        df_sqlite = read_sqlite_day(conn, table, date)
        df_arctic = read_arctic_day(arctic, table, date)
        result = compare_dataframes(df_sqlite, df_arctic, table)
        result["date"] = str(date)
        results.append(result)

        status = result["status"]
        if status == "compared":
            max_errs = {k: f"{v['max_rel_err']:.2e}" for k, v in result["column_errors"].items()}
            log.info(
                f"  {date}: SQLite={result['sqlite_rows']} Arctic={result['arctic_rows']} "
                f"matched={result['matched_rows']} only_sqlite={result['only_sqlite']} "
                f"only_arctic={result['only_arctic']} max_err={max_errs}"
            )
        else:
            log.warning(f"  {date}: {status}")

    return results


def main():
    parser = argparse.ArgumentParser(description="验证 ArcticDB 迁移数据一致性")
    parser.add_argument("--tables", default="daily_data,daily_basic,stk_factor,weekly_data",
                        help="逗号分隔的表名")
    parser.add_argument("--sample-days", type=int, default=5, help="每个表随机采样天数")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="最大相对误差容忍度")
    args = parser.parse_args()

    tables = [t.strip() for t in args.tables.split(",")]
    arctic = ArcticDataProvider(uri=ARCTIC_URI)
    conn = sqlite3.connect(DB_PATH)

    all_results = {}
    failed = False

    for table in tables:
        if table not in TABLE_CONFIG:
            log.warning(f"未知表: {table}，跳过")
            continue
        results = validate_table(arctic, conn, table, args.sample_days)
        if results:
            all_results[table] = results
            for r in results:
                if r["status"] == "compared":
                    for col, err in r["column_errors"].items():
                        if err["max_rel_err"] > args.tolerance:
                            log.error(
                                f"  ❌ {table} {r['date']} {col}: max_rel_err={err['max_rel_err']:.2e} > {args.tolerance}"
                            )
                            failed = True

    conn.close()

    log.info(f"\n{'='*60}")
    if failed:
        log.error("验证失败: 发现超出容忍度的数据差异")
        sys.exit(1)
    else:
        log.success("验证通过: 所有采样数据在容忍度内一致")


if __name__ == "__main__":
    main()
