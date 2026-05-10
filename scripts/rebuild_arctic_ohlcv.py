"""
重建 ArcticDB daily/ohlcv：删除损坏数据，从 Tushare 拉取完整历史重新写入。
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.arctic_provider import ArcticDataProvider
import logging

log = logging.getLogger("rebuild_arctic")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

load_dotenv()
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
if not TUSHARE_TOKEN:
    raise ValueError("TUSHARE_TOKEN not found")
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


def rebuild_ohlcv(days: int = 150):
    provider = ArcticDataProvider()
    lib = provider.get_library("daily")

    # 1. 删除现有 ohlcv
    if "ohlcv" in lib.list_symbols():
        lib.delete("ohlcv")
        log.info("已删除现有 ohlcv symbol")

    end = datetime.now()
    start = end - timedelta(days=days + 30)

    # 2. 获取交易日历
    log.info("获取交易日历...")
    cal = pro.trade_cal(exchange="SSE", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"), is_open="1")
    dates = cal["cal_date"].tolist()
    log.info(f"需要拉取 {len(dates)} 个交易日")

    # 3. 逐日拉取全市场日线
    all_dfs = []
    for i, td in enumerate(dates):
        try:
            df = pro.daily(trade_date=td)
            if df.empty:
                continue
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.set_index("trade_date")
            cols_keep = ["ts_code", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
            df = df[[c for c in cols_keep if c in df.columns]]
            all_dfs.append(df)
            if (i + 1) % 10 == 0 or i == 0:
                log.info(f"  [{i+1}/{len(dates)}] {td}: {len(df)} 只")
            time.sleep(0.12)
        except Exception as e:
            log.warning(f"  {td} 拉取失败: {e}")
            time.sleep(0.5)

    if not all_dfs:
        log.error("没有拉取到任何数据")
        return

    # 4. 合并并写入
    combined = pd.concat(all_dfs).sort_index()
    log.info(f"合并完成: {len(combined)} 行, {combined.index.nunique()} 个交易日")
    log.info(f"每日股票数分布: min={combined.groupby(level=0).size().min()}, max={combined.groupby(level=0).size().max()}")

    lib.write("ohlcv", combined)
    log.info("已写入 ArcticDB")

    # 5. 验证
    df_check = provider.read_daily_ohlcv(dates[0], dates[-1])
    log.info(
        f"验证: {len(df_check)} 行, {df_check.index.nunique()} 个交易日, "
        f"{df_check['ts_code'].nunique()} 只股票"
    )


if __name__ == "__main__":
    rebuild_ohlcv(days=150)
