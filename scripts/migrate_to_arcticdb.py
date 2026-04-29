#!/usr/bin/env python3
"""
SQLite → ArcticDB 迁移脚本（按日期范围分批版）

解决 ArcticDB append 要求时间单调递增的问题：
- 不再用 LIMIT/OFFSET（SQLite 默认按 rowid 排序，时间可能乱序）
- 改为按日期范围分批读取，每批内部时间连续

Usage:
    python scripts/migrate_to_arcticdb.py
"""

import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.arctic_provider import ArcticDataProvider
from src.utils.logger import log

DB_PATH = PROJECT_ROOT / "data" / "cache" / "quant_data.db"
CHECKPOINT_FILE = PROJECT_ROOT / "data" / "cache" / ".migrate_checkpoint.json"
BATCH_DAYS = 30  # 每批读取 30 天的数据


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(cp: dict):
    def _convert(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj
    cp_clean = _convert(cp)
    CHECKPOINT_FILE.write_text(json.dumps(cp_clean, indent=2, ensure_ascii=False), encoding="utf-8")


def get_date_range(conn: sqlite3.Connection, table: str, date_col: str = "trade_date"):
    """获取表的最小和最大日期"""
    cursor = conn.cursor()
    cursor.execute(f"SELECT MIN({date_col}), MAX({date_col}) FROM {table}")
    min_date, max_date = cursor.fetchone()
    return min_date, max_date


def get_unique_dates(conn: sqlite3.Connection, table: str, date_col: str = "trade_date"):
    """获取所有唯一日期（已排序）"""
    df = pd.read_sql_query(f"SELECT DISTINCT {date_col} FROM {table} ORDER BY {date_col}", conn)
    return df[date_col].tolist()


def migrate_table_by_date_range(
    provider: ArcticDataProvider,
    table: str,
    library: str,
    symbol: str,
    date_col: str = "trade_date",
    batch_days: int = BATCH_DAYS,
):
    """按日期范围分批迁移表"""
    cp = load_checkpoint()
    key = f"{library}/{symbol}"
    if cp.get(key) == "done":
        log.info(f"[跳过] {key} 已迁移完成")
        return

    log.info(f"[开始] 迁移 {table} → {key}")
    conn = sqlite3.connect(str(DB_PATH))

    # 获取所有唯一日期
    dates = get_unique_dates(conn, table, date_col)
    total_dates = len(dates)
    total_rows = pd.read_sql_query(f"SELECT COUNT(*) FROM {table}", conn).iloc[0, 0]
    log.info(f"  总日期: {total_dates}, 总行数: {total_rows:,}")

    # 已迁移到的日期索引
    start_idx = cp.get(key, {}).get("date_index", 0)
    if start_idx > 0:
        log.info(f"  断点续传，从第 {start_idx} 个日期 ({dates[start_idx]}) 开始")

    total_migrated = cp.get(key, {}).get("rows", 0)
    is_first = start_idx == 0

    while start_idx < total_dates:
        t0 = datetime.now()
        batch_start = dates[start_idx]
        batch_end = dates[min(start_idx + batch_days - 1, total_dates - 1)]

        df = pd.read_sql_query(
            f"SELECT * FROM {table} WHERE {date_col} >= ? AND {date_col} <= ?",
            conn,
            params=(batch_start, batch_end),
        )
        if df.empty:
            start_idx += batch_days
            continue

        # 日期列处理
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], format="mixed", errors="coerce")
            df = df.set_index(date_col)
            df = df.sort_index()

        total_migrated += len(df)

        # 写入 ArcticDB
        if is_first:
            if library == "daily" and symbol == "ohlcv":
                provider.write_daily_ohlcv(df)
            elif library == "daily" and symbol == "factors":
                provider.write_daily_factors(df)
            elif library == "daily" and symbol == "basic":
                provider.write_daily_basic(df)
            elif library == "weekly" and symbol == "ohlcv":
                provider.write_weekly_ohlcv(df)
            else:
                lib = provider.get_library(library)
                lib.write(symbol, df)
            is_first = False
        else:
            if library == "daily" and symbol == "ohlcv":
                provider.append_daily_ohlcv(df)
            elif library == "daily" and symbol == "factors":
                provider.append_daily_factors(df)
            elif library == "daily" and symbol == "basic":
                provider.append_daily_basic(df)
            elif library == "weekly" and symbol == "ohlcv":
                provider.append_weekly_ohlcv(df)
            else:
                lib = provider.get_library(library)
                lib.append(symbol, df)

        start_idx += batch_days
        elapsed = (datetime.now() - t0).total_seconds()
        log.info(
            f"  [{start_idx}/{total_dates}] {batch_start}~{batch_end} "
            f"{len(df):,} 行 {elapsed:.1f}s | 累计 {total_migrated:,} 行"
        )

        # 保存断点
        cp[key] = {"date_index": start_idx, "rows": total_migrated}
        save_checkpoint(cp)

    # 标记完成
    cp[key] = "done"
    save_checkpoint(cp)
    conn.close()
    log.success(f"[完成] {table} → {key} ({total_migrated:,} 行)")


def migrate_reference(provider: ArcticDataProvider):
    """迁移参考数据"""
    conn = sqlite3.connect(str(DB_PATH))
    cp = load_checkpoint()

    # stock_basic
    if not cp.get("reference/stock_basic") == "done":
        log.info("[开始] 迁移 stock_basic")
        try:
            df = pd.read_sql_query("SELECT * FROM stock_basic", conn)
            if not df.empty:
                provider.write_stock_basic(df)
                log.success(f"[完成] stock_basic: {len(df)} 条")
            cp["reference/stock_basic"] = "done"
            save_checkpoint(cp)
        except Exception as e:
            log.warning(f"stock_basic 迁移失败: {e}")

    # trade_cal
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trade_cal'")
    if cursor.fetchone():
        if not cp.get("reference/trade_cal") == "done":
            log.info("[开始] 迁移 trade_cal")
            try:
                df = pd.read_sql_query("SELECT * FROM trade_cal", conn)
                if not df.empty:
                    provider.write_trade_cal(df)
                    log.success(f"[完成] trade_cal: {len(df)} 条")
                cp["reference/trade_cal"] = "done"
                save_checkpoint(cp)
            except Exception as e:
                log.warning(f"trade_cal 迁移失败: {e}")

    conn.close()


def cleanup_arcticdb(provider: ArcticDataProvider):
    """清理已迁移的错误数据，准备重新迁移"""
    log.info("清理 ArcticDB 中的旧数据...")
    for lib_name in provider.list_libraries():
        lib = provider.get_library(lib_name)
        for sym in lib.list_symbols():
            try:
                lib.delete(sym)
                log.info(f"  已删除 {lib_name}/{sym}")
            except Exception as e:
                log.warning(f"  删除 {lib_name}/{sym} 失败: {e}")


def main():
    log.info("=" * 80)
    log.info("SQLite → ArcticDB 迁移启动（按日期范围分批版）")
    log.info("=" * 80)

    provider = ArcticDataProvider()

    # 如果 checkpoint 不存在，清理旧数据
    if not CHECKPOINT_FILE.exists():
        cleanup_arcticdb(provider)

    # 1. daily_data → daily/ohlcv
    migrate_table_by_date_range(provider, "daily_data", "daily", "ohlcv")

    # 2. stk_factor → daily/factors
    migrate_table_by_date_range(provider, "stk_factor", "daily", "factors")

    # 3. daily_basic → daily/basic
    migrate_table_by_date_range(provider, "daily_basic", "daily", "basic")

    # 4. weekly_data → weekly/ohlcv
    migrate_table_by_date_range(provider, "weekly_data", "weekly", "ohlcv")

    # 5. reference 数据
    migrate_reference(provider)

    # 完成
    log.info("=" * 80)
    log.success("迁移完成！")
    info = provider.get_info()
    for lib_name, lib_info in info.get("libraries", {}).items():
        for sym, sym_info in lib_info.items():
            if isinstance(sym_info, dict) and "shape" in sym_info:
                log.success(f"  {lib_name}/{sym}: {sym_info['shape']}")
    log.info("=" * 80)

    # 删除断点文件
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


if __name__ == "__main__":
    main()
