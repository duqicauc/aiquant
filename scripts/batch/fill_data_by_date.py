#!/usr/bin/env python3
"""
按天批量补全历史数据（高效积分利用）

策略：
1. 按交易日历逐天调用 pro.daily(trade_date=date) - 一次获取全市场
2. 按天调用 pro.daily_basic(trade_date=date) - 一次获取全市场指标
3. 数据存入 stock_data_cache（按 ts_code 拆分存储）

积分估算（60天）：
- daily: ~60次调用, 每次~5000条
- daily_basic: ~60次调用
- 总计约 1000-2000 积分（在 8120 范围内）
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import tushare as ts

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.storage.cache_manager import CacheManager
from src.utils.logger import log


def load_token():
    """加载 Tushare Token"""
    from dotenv import load_dotenv
    import os
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("TUSHARE_TOKEN")
    if not token or token == "YOUR_TUSHARE_TOKEN":
        raise ValueError("请在 .env 文件中设置有效的 TUSHARE_TOKEN")
    ts.set_token(token)
    return ts.pro_api(token)


def get_trade_dates(pro, start_date: str, end_date: str) -> list:
    """获取交易日历"""
    df = pro.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date, is_open="1")
    if df is None or df.empty:
        return []
    return sorted(df["cal_date"].tolist())


def fetch_daily_by_date(pro, trade_date: str) -> pd.DataFrame:
    """按日期获取全市场日线数据"""
    try:
        df = pro.daily(trade_date=trade_date)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        log.warning(f"daily {trade_date}: {e}")
    return pd.DataFrame()


def fetch_daily_basic_by_date(pro, trade_date: str) -> pd.DataFrame:
    """按日期获取全市场每日指标"""
    try:
        df = pro.daily_basic(trade_date=trade_date)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        log.warning(f"daily_basic {trade_date}: {e}")
    return pd.DataFrame()


def save_to_cache(cache: CacheManager, df: pd.DataFrame, data_type: str):
    """将 DataFrame 按股票拆分存入 cache"""
    if df is None or df.empty:
        return 0

    if "trade_date" not in df.columns and "ts_code" not in df.columns:
        log.warning(f"数据缺少必要列: {df.columns.tolist()}")
        return 0

    # 确保 trade_date 是字符串格式
    if "trade_date" in df.columns:
        df = df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")

    count = 0
    for ts_code, group in df.groupby("ts_code"):
        try:
            cache.save_data(group, data_type=data_type, ts_code=ts_code)
            count += 1
        except Exception as e:
            log.warning(f"保存 {ts_code} 失败: {e}")

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default=None,
                        help="开始日期(YYYYMMDD), 默认90天前")
    parser.add_argument("--end-date", type=str, default=None,
                        help="结束日期(YYYYMMDD), 默认今天")
    parser.add_argument("--data-types", type=str, default="daily,daily_basic",
                        help="要补全的数据类型，逗号分隔")
    parser.add_argument("--checkpoint", type=str, default=".checkpoint_fill_data.txt",
                        help="断点续传文件")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="每次API调用间隔(秒)")
    args = parser.parse_args()

    # 默认日期范围
    if args.end_date is None:
        args.end_date = datetime.now().strftime("%Y%m%d")
    if args.start_date is None:
        args.start_date = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")

    data_types = [t.strip() for t in args.data_types.split(",")]

    log.info(f"{'='*60}")
    log.info(f"数据补全: {args.start_date} ~ {args.end_date}")
    log.info(f"数据类型: {data_types}")
    log.info(f"{'='*60}")

    # 初始化
    pro = load_token()
    cache = CacheManager()

    # 获取交易日历
    trade_dates = get_trade_dates(pro, args.start_date, args.end_date)
    log.info(f"交易日数: {len(trade_dates)}")

    if not trade_dates:
        log.error("无交易日")
        return

    # 读取 checkpoint
    checkpoint_file = PROJECT_ROOT / args.checkpoint
    completed = set()
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            completed = set(line.strip() for line in f if line.strip())
        log.info(f"断点续传: 已完成 {len(completed)} 天")

    remaining = [d for d in trade_dates if d not in completed]
    log.info(f"剩余待处理: {len(remaining)} 天")

    # 逐天处理
    for i, date in enumerate(remaining):
        log.info(f"[{i+1}/{len(remaining)}] 处理 {date}")

        for dtype in data_types:
            if dtype == "daily":
                df = fetch_daily_by_date(pro, date)
                if not df.empty:
                    count = save_to_cache(cache, df, "daily_data")
                    log.info(f"  daily: {count} 只股票")
            elif dtype == "daily_basic":
                df = fetch_daily_basic_by_date(pro, date)
                if not df.empty:
                    count = save_to_cache(cache, df, "daily_basic")
                    log.info(f"  daily_basic: {count} 只股票")

        # 保存 checkpoint
        with open(checkpoint_file, "a") as f:
            f.write(f"{date}\n")

        # 限流
        if i < len(remaining) - 1:
            time.sleep(args.sleep)

    log.success("数据补全完成！")


if __name__ == "__main__":
    main()
