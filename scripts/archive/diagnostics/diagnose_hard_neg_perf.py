#!/usr/bin/env python3
"""诊断硬负样本筛选性能瓶颈"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.models.screening.hard_negative_screener import HardNegativeSampleScreener
from src.data.data_manager import DataManager

def main():
    print("=" * 60)
    print("硬负样本筛选性能诊断")
    print("=" * 60)

    # 1. 初始化
    t0 = time.time()
    dm = DataManager(source="tushare", use_cache=True)
    screener = HardNegativeSampleScreener(dm)
    print(f"初始化耗时: {time.time() - t0:.2f}s")

    # 2. 获取股票列表
    t0 = time.time()
    all_stocks = screener._get_valid_stock_list()
    print(f"获取股票列表: {len(all_stocks)} 只, 耗时 {time.time() - t0:.2f}s")

    # 3. 加载正样本
    pos_path = Path("data/training/samples/positive_samples_v295.csv")
    df_pos = pd.read_csv(pos_path)
    positive_stocks = set(df_pos["ts_code"].unique())
    t1_dates = df_pos["t1_date"].unique()[:10]  # 只测前10个日期
    print(f"正样本: {len(df_pos)} 个, 唯一股票: {len(positive_stocks)}, T1日期(前10): {len(t1_dates)}")

    # 4. 测试单次 _screen_hard_negatives_for_date
    print("\n--- 测试 _screen_hard_negatives_for_date ---")
    test_date = str(t1_dates[0])
    t0 = time.time()
    samples = screener._screen_hard_negatives_for_date(
        t1_date=test_date,
        all_stocks=all_stocks,
        positive_stocks=positive_stocks,
        min_return=20.0,
        max_return=40.0,
        samples_per_date=3,
        random_seed=42,
    )
    elapsed = time.time() - t0
    print(f"  日期 {test_date}: 找到 {len(samples)} 个, 耗时 {elapsed:.2f}s")

    # 5. 测试单次 _screen_high_position_fail_for_date
    print("\n--- 测试 _screen_high_position_fail_for_date ---")
    t0 = time.time()
    samples = screener._screen_high_position_fail_for_date(
        t1_date=test_date,
        all_stocks=all_stocks,
        positive_stocks=positive_stocks,
        samples_per_date=1,
        random_seed=42,
    )
    elapsed = time.time() - t0
    print(f"  日期 {test_date}: 找到 {len(samples)} 个, 耗时 {elapsed:.2f}s")

    # 6. 测试 get_daily_data 缓存性能
    print("\n--- 测试 get_daily_data 缓存性能 ---")
    candidate = all_stocks[~all_stocks["ts_code"].isin(positive_stocks)].sample(10, random_state=42)
    total_cache = 0
    total_api = 0
    cache_hits = 0
    for _, row in candidate.iterrows():
        ts_code = row["ts_code"]
        t0 = time.time()
        df = dm.get_daily_data(ts_code, "20240101", "20240301", adjust="qfq")
        elapsed = time.time() - t0
        if len(df) > 0:
            cache_hits += 1
            total_cache += elapsed
        else:
            total_api += elapsed
        print(f"  {ts_code}: {len(df)} 行, 耗时 {elapsed:.3f}s")
    print(f"  缓存命中: {cache_hits}/10, 平均耗时(有数据): {total_cache/cache_hits:.3f}s" if cache_hits > 0 else "  无缓存命中")

    # 7. 估算完整运行时间
    print("\n--- 估算 ---")
    per_date = 5.0  # 保守估计
    total_dates = len(df_pos["t1_date"].unique())
    est_minutes = total_dates * per_date / 60
    print(f"每日期估算耗时: {per_date:.1f}s")
    print(f"总日期数: {total_dates}")
    print(f"估算总耗时: {est_minutes:.1f} 分钟 ({est_minutes/60:.1f} 小时)")

    print("\n诊断完成")

if __name__ == "__main__":
    main()
