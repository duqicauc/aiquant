"""
从 Tushare 补全 ArcticDB daily/ohlcv 历史数据。
策略：按交易日逐日拉取全市场日线（pro.daily(trade_date=...)），
效率最高，避免逐只股票请求。
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

log = logging.getLogger("backfill_arctic")
load_dotenv()

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
if not TUSHARE_TOKEN:
    raise ValueError("TUSHARE_TOKEN not found in .env")
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


def get_trade_calendar(start: str, end: str) -> list:
    """获取交易日历"""
    df = pro.trade_cal(exchange="SSE", start_date=start, end_date=end, is_open="1")
    return df["cal_date"].tolist()


def fetch_daily_for_date(trade_date: str) -> pd.DataFrame:
    """拉取某一天的全部股票日线"""
    df = pro.daily(trade_date=trade_date)
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date")
    # 列名对齐 ArcticDB 现有格式
    cols_keep = ["ts_code", "open", "high", "low", "close", "pre_close",
                 "change", "pct_chg", "vol", "amount"]
    for c in cols_keep:
        if c not in df.columns:
            df[c] = None
    return df[cols_keep].sort_index()


def backfill_days(days: int = 150):
    """补全最近 N 个交易日的全市场 ohlcv 到 ArcticDB"""
    provider = ArcticDataProvider()

    end = datetime.now()
    start = end - timedelta(days=days + 30)  # 多取一些，过滤非交易日
    dates = get_trade_calendar(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))

    # 只补 ArcticDB 中缺失的日期
    log.info("检查 ArcticDB 现有 ohlcv 日期...")
    try:
        existing = provider.read_daily_ohlcv(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
        existing_dates = set(existing.index.strftime("%Y%m%d"))
        log.info(f"ArcticDB 现有 ohlcv 日期数: {len(existing_dates)}")
    except Exception as e:
        existing_dates = set()
        log.info(f"ArcticDB ohlcv 读取失败或为空，全量补全: {e}")

    missing_dates = [d for d in dates if d not in existing_dates]
    log.info(f"需要补全的交易日数: {len(missing_dates)}")

    all_dfs = []
    for i, td in enumerate(missing_dates):
        try:
            df = fetch_daily_for_date(td)
            if not df.empty:
                all_dfs.append(df)
                if (i + 1) % 10 == 0 or i == 0:
                    log.info(f"  [{i+1}/{len(missing_dates)}] {td}: {len(df)} 只")
            time.sleep(0.12)  # 限速 ~8 req/s
        except Exception as e:
            log.warning(f"  {td} 拉取失败: {e}")
            time.sleep(0.5)

    if all_dfs:
        new_data = pd.concat(all_dfs).sort_index()
        log.info(f"Tushare 拉取完成: {len(new_data)} 行, {new_data.index.nunique()} 个交易日")

        # ArcticDB append 不支持往回插，需要合并后覆盖写入
        log.info("合并现有数据并覆盖写入 ArcticDB...")
        try:
            existing = provider.read_daily_ohlcv("19900101", "20991231")
            if not existing.empty:
                combined = pd.concat([existing, new_data])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
            else:
                combined = new_data
        except Exception:
            combined = new_data

        lib = provider.get_library("daily")
        lib.write("ohlcv", combined)
        log.info(f"已覆盖写入 ArcticDB: {len(combined)} 行, {combined.index.nunique()} 个交易日")
    else:
        log.info("无新数据需要写入")

    # 最终验证
    df_check = provider.read_daily_ohlcv(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
    log.info(
        f"ArcticDB ohlcv 最终: {len(df_check)} 行, {df_check.index.nunique()} 个交易日, "
        f"{df_check.index.min()} ~ {df_check.index.max()}"
    )


if __name__ == "__main__":
    backfill_days(days=150)
