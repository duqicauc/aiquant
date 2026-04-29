#!/usr/bin/env python3
"""
股票基本信息缓存脚本

从 Tushare 获取 stock_basic 并写入 SQLite stock_basic 表。
在 auto_daily_pipeline.py 中定期调用（建议每周一次）。

Usage:
    python scripts/cache_stock_basic.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tushare_data_provider import TushareDataProvider
from src.utils.logger import log

DB_PATH = PROJECT_ROOT / "data" / "cache" / "quant_data.db"


def create_table(conn: sqlite3.Connection):
    """创建 stock_basic 表（如果不存在）"""
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_basic (
            ts_code TEXT PRIMARY KEY,
            symbol TEXT,
            name TEXT,
            area TEXT,
            industry TEXT,
            cnspell TEXT,
            market TEXT,
            exchange TEXT,
            curr_type TEXT,
            list_status TEXT,
            list_date TEXT,
            delist_date TEXT,
            is_hs TEXT,
            act_name TEXT,
            act_ent_type TEXT,
            fullname TEXT,
            enname TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def cache_stock_basic():
    """从 Tushare 获取并缓存 stock_basic"""
    log.info("开始缓存 stock_basic...")
    provider = TushareDataProvider()

    try:
        df = provider.pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,cnspell,market,exchange,curr_type,list_status,list_date,delist_date,is_hs,act_name,act_ent_type,fullname,enname",
        )
    except Exception as e:
        log.error(f"Tushare 获取 stock_basic 失败: {e}")
        return False

    if df is None or df.empty:
        log.warning("stock_basic 为空")
        return False

    log.info(f"获取到 {len(df)} 只股票基本信息")

    conn = sqlite3.connect(str(DB_PATH))
    create_table(conn)

    # 清空旧数据并插入新数据
    cursor = conn.cursor()
    cursor.execute("DELETE FROM stock_basic")
    conn.commit()

    df.to_sql("stock_basic", conn, if_exists="append", index=False)
    conn.close()

    log.success(f"stock_basic 已缓存: {len(df)} 条")
    return True


if __name__ == "__main__":
    cache_stock_basic()
