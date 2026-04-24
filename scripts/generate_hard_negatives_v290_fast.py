#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.9.0 硬负样本扩充脚本（高效版）
基于 cache DB 批量查询，避免逐只从 Tushare 拉取

目标：生成 2,000+ 硬负样本
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

DB_PATH = "data/cache/quant_data.db"
POSITIVE_SAMPLES = "data/training/samples/positive_samples.csv"
OUTPUT = "data/training/samples/hard_negatives_v290.csv"

def get_trade_date_offset(conn, trade_date, offset):
    """获取 trade_date 前后 offset 个交易日的日期"""
    if offset >= 0:
        df = pd.read_sql(
            f"SELECT trade_date FROM ("
            f"  SELECT DISTINCT trade_date FROM daily_data "
            f"  WHERE trade_date > '{trade_date}' ORDER BY trade_date LIMIT {offset}"
            f") ORDER BY trade_date DESC LIMIT 1",
            conn
        )
    else:
        df = pd.read_sql(
            f"SELECT trade_date FROM ("
            f"  SELECT DISTINCT trade_date FROM daily_data "
            f"  WHERE trade_date < '{trade_date}' ORDER BY trade_date DESC LIMIT {abs(offset)}"
            f") ORDER BY trade_date LIMIT 1",
            conn
        )
    return df['trade_date'].iloc[0] if not df.empty else None

def main():
    print("=" * 80)
    print("v2.9.0 硬负样本扩充（高效版）")
    print("=" * 80)

    conn = sqlite3.connect(DB_PATH)

    # 读取正样本
    pos_df = pd.read_csv(POSITIVE_SAMPLES)
    pos_df['t1_date'] = pos_df['t1_date'].astype(str)
    print(f"正样本数量: {len(pos_df)}")

    # 正样本的股票-日期集合（用于排除）
    positive_set = set(zip(pos_df['ts_code'], pos_df['t1_date']))

    # 获取所有 T1 日期
    t1_dates = sorted(pos_df['t1_date'].unique())
    print(f"T1 日期数量: {len(t1_dates)}")

    # 确定需要查询的日期范围（所有 T1 日期的前34天到后20天）
    all_dates = set()
    for t1 in t1_dates:
        # 前34天
        d = pd.read_sql(
            f"SELECT trade_date FROM daily_data WHERE trade_date <= '{t1}' "
            f"GROUP BY trade_date ORDER BY trade_date DESC LIMIT 35",
            conn
        )
        all_dates.update(d['trade_date'].tolist())

        # 后20天
        d = pd.read_sql(
            f"SELECT trade_date FROM daily_data WHERE trade_date >= '{t1}' "
            f"GROUP BY trade_date ORDER BY trade_date LIMIT 21",
            conn
        )
        all_dates.update(d['trade_date'].tolist())

    date_list = sorted(all_dates)
    print(f"需要覆盖的交易日: {len(date_list)} 天 ({date_list[0]} ~ {date_list[-1]})")

    # 获取所有有效股票代码
    stocks_df = pd.read_sql(
        "SELECT DISTINCT ts_code FROM daily_data WHERE trade_date >= '20200101'",
        conn
    )
    all_stocks = stocks_df['ts_code'].tolist()
    print(f"有效股票数: {len(all_stocks)}")

    # 分批处理股票（每批 500 只，避免内存溢出）
    BATCH_SIZE = 500
    all_hard_negatives = []

    total_batches = (len(all_stocks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(all_stocks))
        batch_stocks = all_stocks[start_idx:end_idx]

        # 批量查询这批股票在所需日期范围内的数据
        stock_list = "','".join(batch_stocks)
        date_start = date_list[0]
        date_end = date_list[-1]

        df = pd.read_sql(
            f"SELECT ts_code, trade_date, close FROM daily_data "
            f"WHERE ts_code IN ('{stock_list}') "
            f"AND trade_date BETWEEN '{date_start}' AND '{date_end}' "
            f"ORDER BY ts_code, trade_date",
            conn
        )

        if df.empty:
            continue

        # 按股票分组处理
        for ts_code, group in df.groupby('ts_code'):
            group = group.sort_values('trade_date').reset_index(drop=True)

            # 计算 34 日涨幅（当前 / 34 日前 - 1）
            group['close_34d_ago'] = group['close'].shift(34)
            group['return_34d'] = (group['close'] / group['close_34d_ago'] - 1) * 100

            # 计算 20 日后涨幅
            group['close_20d_future'] = group['close'].shift(-20)
            group['return_20d_future'] = (group['close_20d_future'] / group['close'] - 1) * 100

            # 计算 5 日后涨幅（伪突破判断用）
            group['close_5d_future'] = group['close'].shift(-5)
            group['return_5d_future'] = (group['close_5d_future'] / group['close'] - 1) * 100

            # 计算 20 日高点和突破判断
            group['high_20d'] = group['close'].rolling(20).max()
            group['is_breakout'] = group['close'] >= group['high_20d'].shift(1)

            # 计算 T1 前 34 日涨幅（用于 high_position_fail）
            group['pre_34d_return'] = group['return_34d']

            for _, row in group.iterrows():
                t1_date = row['trade_date']
                if t1_date not in t1_dates:
                    continue

                # 排除已经是正样本的
                if (ts_code, t1_date) in positive_set:
                    continue

                return_34d = row['return_34d']
                return_20d = row['return_20d_future']
                return_5d = row['return_5d_future']
                is_breakout = row['is_breakout']
                pre_return = row['pre_34d_return']

                if pd.isna(return_34d):
                    continue

                # 类型 1: near_miss（34日涨幅 15%-50%）
                if 15 <= return_34d <= 50:
                    all_hard_negatives.append({
                        'ts_code': ts_code,
                        't1_date': t1_date,
                        'return_34d': return_34d,
                        'return_20d_future': return_20d,
                        'sample_type': 'near_miss',
                    })

                # 类型 2: high_position_fail（T1前已涨>25%，但T1后下跌）
                elif pre_return > 25 and not pd.isna(return_20d) and return_20d < 0:
                    all_hard_negatives.append({
                        'ts_code': ts_code,
                        't1_date': t1_date,
                        'return_34d': return_34d,
                        'return_20d_future': return_20d,
                        'sample_type': 'high_position_fail',
                    })

                # 类型 3: false_breakout（突破20日高点后5日内回落>5%）
                elif is_breakout and not pd.isna(return_5d) and return_5d < -5:
                    all_hard_negatives.append({
                        'ts_code': ts_code,
                        't1_date': t1_date,
                        'return_34d': return_34d,
                        'return_20d_future': return_5d,
                        'sample_type': 'false_breakout',
                    })

        if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
            print(f"  批次 {batch_idx + 1}/{total_batches} 完成, 累计硬负样本: {len(all_hard_negatives)}")

    conn.close()

    print(f"\n原始硬负样本: {len(all_hard_negatives)}")

    if len(all_hard_negatives) == 0:
        print("❌ 未生成硬负样本！")
        return

    # 去重
    hn_df = pd.DataFrame(all_hard_negatives)
    before_dedup = len(hn_df)
    hn_df = hn_df.drop_duplicates(subset=['ts_code', 't1_date', 'sample_type'])
    after_dedup = len(hn_df)
    print(f"去重后: {after_dedup} (去重前: {before_dedup})")

    # 统计
    type_counts = hn_df['sample_type'].value_counts()
    print("\n样本类型分布:")
    for t, c in type_counts.items():
        print(f"  {t}: {c}")

    # 如果超过 3000 个，随机采样到 2000-2500 个
    TARGET_COUNT = 2500
    if len(hn_df) > TARGET_COUNT:
        print(f"\n硬负样本过多，随机采样至 {TARGET_COUNT} 个...")
        hn_df = hn_df.sample(n=TARGET_COUNT, random_state=42)

    # 保存
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    hn_df.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    print(f"\n✅ 已保存: {OUTPUT} ({len(hn_df)} 条)")

    # 与现有对比
    existing = Path("data/training/samples/hard_negative_samples.csv")
    if existing.exists():
        existing_df = pd.read_csv(existing)
        print(f"\n对比:")
        print(f"  现有硬负样本: {len(existing_df)}")
        print(f"  新增硬负样本: {len(hn_df)}")
        print(f"  增长倍数: {len(hn_df) / max(len(existing_df), 1):.1f}x")

    # 计算占负样本比例
    neg_df = pd.read_csv("data/training/samples/negative_samples_v2.csv")
    total_negative = len(neg_df)
    hard_pct = len(hn_df) / (total_negative + len(hn_df)) * 100
    print(f"\n硬负样本占比: {hard_pct:.1f}% (目标: 15-20%)")

if __name__ == '__main__':
    main()
