"""
将 SQLite daily_data 历史行情追加到 ArcticDB daily/ohlcv
以 ArcticDB 为唯一行情数据源
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.arctic_provider import ArcticDataProvider
from src.utils.logger import get_logger

log = get_logger(__name__)


def migrate_ohlcv_to_arctic():
    db_path = PROJECT_ROOT / "data" / "cache" / "quant_data.db"
    conn = sqlite3.connect(db_path)

    # 统计
    total = conn.execute("SELECT COUNT(*) FROM daily_data").fetchone()[0]
    min_date, max_date = conn.execute(
        "SELECT MIN(trade_date), MAX(trade_date) FROM daily_data"
    ).fetchone()
    log.info(f"SQLite daily_data: {total} 行, {min_date} ~ {max_date}")

    provider = ArcticDataProvider()

    # 按年分批读取，避免内存爆炸
    years = pd.read_sql_query(
        "SELECT DISTINCT substr(trade_date,1,4) as y FROM daily_data ORDER BY y",
        conn,
    )["y"].tolist()

    for year in years:
        df = pd.read_sql_query(
            f"SELECT * FROM daily_data WHERE trade_date LIKE '{year}%'",
            conn,
            parse_dates=["trade_date"],
        )
        if df.empty:
            continue
        df = df.set_index("trade_date").sort_index()
        provider.append_daily_ohlcv(df)
        log.info(f"  已追加 {year}: {len(df)} 行")

    conn.close()

    # 验证
    df_check = provider.read_daily_ohlcv(str(min_date), str(max_date))
    log.info(
        f"ArcticDB ohlcv 验证: {len(df_check)} 行, {df_check.index.nunique()} 个交易日, "
        f"{df_check.index.min()} ~ {df_check.index.max()}"
    )


if __name__ == "__main__":
    migrate_ohlcv_to_arctic()
