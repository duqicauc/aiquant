#!/usr/bin/env python3
"""强制重新补全 ArcticDB 日线数据，从指定日期开始"""
import sys
sys.path.insert(0, '/app')

import pandas as pd
from datetime import datetime, timedelta
from src.data.arctic_provider import ArcticDataProvider
from src.data.fetcher.tushare_fetcher import TushareFetcher
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def backfill_force(start_date: str, end_date: str):
    ap = ArcticDataProvider()
    tf = TushareFetcher()
    pro = tf.pro

    logger.info(f"=== 强制补全 ArcticDB OHLCV: {start_date} ~ {end_date} ===")

    # 获取交易日历
    cal = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
    trade_days = cal[cal.is_open == 1]['cal_date'].tolist()
    logger.info(f"共 {len(trade_days)} 个交易日")

    total_rows = 0
    for i, date in enumerate(trade_days):
        try:
            df = tf.pro.daily(trade_date=date)
            if df.empty:
                logger.warning(f"[{i+1}/{len(trade_days)}] {date}: 无数据")
                continue

            # 写入 ArcticDB
            ap.append_daily_ohlcv(df)
            total_rows += len(df)
            logger.info(f"[{i+1}/{len(trade_days)}] {date}: {len(df)} 只股票 → ArcticDB (累计 {total_rows} 行)")
        except Exception as e:
            logger.error(f"[{i+1}/{len(trade_days)}] {date}: 错误 {e}")

    logger.info(f"=== 补全完成: 共 {total_rows} 行数据写入 ===")

if __name__ == '__main__':
    backfill_force('20240101', datetime.now().strftime('%Y%m%d'))
