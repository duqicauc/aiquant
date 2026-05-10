#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断硬负样本筛选卡住原因"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from src.utils.logger import log
from src.data.data_manager import DataManager
from src.models.screening.hard_negative_screener import HardNegativeSampleScreener

# 加载正样本
df_pos = pd.read_csv(PROJECT_ROOT / "data/training/samples/positive_samples_v295.csv")
log.info(f"正样本: {len(df_pos)} 个")

# 取前3个T1日期测试
t1_dates = df_pos["t1_date"].unique()[:3]
log.info(f"测试T1日期: {list(t1_dates)}")

dm = DataManager()
screener = HardNegativeSampleScreener(data_manager=dm)

# 测试单个T1日期的筛选
for t1_date in t1_dates:
    log.info("")
    log.info(f"=== 测试 T1={t1_date} ===")

    # 测试 near_miss
    log.info("测试 near_miss...")
    try:
        samples = screener._screen_hard_negatives_for_date(
            t1_date=str(t1_date),
            all_stocks=screener._get_valid_stock_list(),
            positive_stocks=set(df_pos["ts_code"].unique()),
            min_return=20.0,
            max_return=40.0,
            samples_per_date=2,
            random_seed=42,
        )
        log.info(f"  near_miss: {len(samples)} 个")
    except Exception as e:
        log.error(f"  near_miss 失败: {e}")

    # 测试 high_position_fail
    log.info("测试 high_position_fail...")
    try:
        samples = screener._screen_high_position_fail_for_date(
            t1_date=str(t1_date),
            all_stocks=screener._get_valid_stock_list(),
            positive_stocks=set(df_pos["ts_code"].unique()),
            samples_per_date=2,
            random_seed=42,
        )
        log.info(f"  high_position_fail: {len(samples)} 个")
    except Exception as e:
        log.error(f"  high_position_fail 失败: {e}")

    # 测试 false_breakout
    log.info("测试 false_breakout...")
    try:
        samples = screener._screen_false_breakout_for_date(
            t1_date=str(t1_date),
            all_stocks=screener._get_valid_stock_list(),
            positive_stocks=set(df_pos["ts_code"].unique()),
            samples_per_date=2,
            random_seed=42,
        )
        log.info(f"  false_breakout: {len(samples)} 个")
    except Exception as e:
        log.error(f"  false_breakout 失败: {e}")

log.info("")
log.info("诊断完成")
