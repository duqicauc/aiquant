#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为新硬负样本(v291)提取v5格式基础数据

输入: data/training/samples/hard_negatives_v291.csv
输出: data/training/features/hard_negative_v5_base.csv

从 cache DB 批量查询:
- daily_data: 行情数据
- stk_factor: Tushare技术因子(MACD/KDJ/RSI)
- daily_basic: 每日指标(换手率/市值/量比)
"""

import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.logger import log

INPUT = "data/training/samples/hard_negatives_v291.csv"
OUTPUT = "data/training/features/hard_negative_v5_base.csv"
DB_PATH = "data/cache/quant_data.db"
LOOKBACK = 34  # v5是34天数据


def get_sample_data(conn, ts_code, t1_date, lookback=34):
    """从cache DB查询单只股票在T1前lookback天的数据"""
    # 获取T1前lookback+5天的日期范围（留buffer）
    date_df = pd.read_sql(
        f"SELECT trade_date FROM daily_data WHERE ts_code = '{ts_code}' "
        f"AND trade_date <= '{t1_date}' ORDER BY trade_date DESC LIMIT {lookback + 5}",
        conn
    )
    if len(date_df) < lookback:
        return pd.DataFrame()

    start_date = date_df['trade_date'].iloc[-1]

    # 1. daily_data
    df_daily = pd.read_sql(
        f"SELECT trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount "
        f"FROM daily_data WHERE ts_code = '{ts_code}' AND trade_date BETWEEN '{start_date}' AND '{t1_date}' "
        f"ORDER BY trade_date",
        conn
    )
    if df_daily.empty or len(df_daily) < lookback * 0.8:
        return pd.DataFrame()

    # 2. stk_factor (Tushare技术因子)
    df_factor = pd.read_sql(
        f"SELECT trade_date, macd_dif, macd_dea, macd, kdj_k, kdj_d, kdj_j, "
        f"rsi_6, rsi_12, rsi_24 FROM stk_factor "
        f"WHERE ts_code = '{ts_code}' AND trade_date BETWEEN '{start_date}' AND '{t1_date}'",
        conn
    )
    if not df_factor.empty:
        df_daily = df_daily.merge(df_factor, on='trade_date', how='left')

    # 3. daily_basic (换手率/市值/量比)
    df_basic = pd.read_sql(
        f"SELECT trade_date, turnover_rate, total_mv, circ_mv, volume_ratio "
        f"FROM daily_basic WHERE ts_code = '{ts_code}' AND trade_date BETWEEN '{start_date}' AND '{t1_date}'",
        conn
    )
    if not df_basic.empty:
        df_daily = df_daily.merge(df_basic, on='trade_date', how='left')

    # 只取最后lookback天
    df_daily = df_daily.tail(lookback).reset_index(drop=True)

    # 添加ts_code
    df_daily.insert(0, 'ts_code', ts_code)

    return df_daily


def main():
    log.info("=" * 80)
    log.info("新硬负样本v5基础数据提取")
    log.info("=" * 80)

    samples = pd.read_csv(INPUT)
    log.info(f"样本数量: {len(samples)}")

    conn = sqlite3.connect(DB_PATH)
    all_results = []

    for idx, row in samples.iterrows():
        ts_code = row['ts_code']
        t1_date = str(row['t1_date'])
        name = row.get('name', '')

        df = get_sample_data(conn, ts_code, t1_date, LOOKBACK)
        if df.empty:
            continue

        # 添加元数据
        df['sample_id'] = f"HN290_{idx}"
        df['name'] = name
        df['days_to_t1'] = list(range(-len(df) + 1, 1))
        df['label'] = 0

        all_results.append(df)

        if (idx + 1) % 100 == 0 or idx == len(samples) - 1:
            log.info(f"  进度: {idx + 1}/{len(samples)} ({(idx+1)/len(samples)*100:.1f}%)")

    conn.close()

    if not all_results:
        log.error("未提取到任何数据")
        return

    df_all = pd.concat(all_results, ignore_index=True)
    log.info(f"\n总记录数: {len(df_all)}")
    log.info(f"独立样本: {df_all['sample_id'].nunique()}")
    log.info(f"列数: {len(df_all.columns)}")

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    log.success(f"已保存: {OUTPUT}")


if __name__ == '__main__':
    main()
