#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.9.0 硬负样本扩充脚本
目标：将硬负样本从 ~177 个扩充至 2,000+ 个（15-20% 占比）

策略：
1. 扩大时间范围：2020-01-01 至 2026-04-21（原可能只扫描了部分时间）
2. 降低阈值：让更多"接近突破"的股票被纳入
3. 新增"熊市假突破"类型：在大盘下跌期间出现的突破信号
4. 增加每日采样数量
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_manager import DataManager
from src.models.screening.hard_negative_screener import HardNegativeSampleScreener
from src.utils.logger import log

POSITIVE_SAMPLES = "data/training/samples/positive_samples.csv"
OUTPUT = "data/training/samples/hard_negatives_v290.csv"

def main():
    log.info("=" * 80)
    log.info("v2.9.0 硬负样本扩充")
    log.info("=" * 80)

    # 读取正样本
    pos_df = pd.read_csv(POSITIVE_SAMPLES)
    log.info(f"正样本数量: {len(pos_df)}")

    # 初始化数据管理器和筛选器
    dm = DataManager()
    screener = HardNegativeSampleScreener(dm)

    # 扩充参数：降低阈值 + 增加采样
    hard_negatives = screener.screen_hard_negatives(
        positive_samples_df=pos_df,
        min_return=15.0,      # 从 20% 降至 15%，纳入更多"接近突破"样本
        max_return=50.0,      # 从 45% 升至 50%，接近正样本阈值
        samples_per_date=30,  # 从默认 15 增至 30，大幅增加数量
        random_seed=42,
        include_high_position_fail=True,
        include_false_breakout=True,
    )

    log.info(f"\n硬负样本筛选完成: {len(hard_negatives)} 个")

    if len(hard_negatives) == 0:
        log.error("未生成硬负样本！")
        return

    # 去重
    before_dedup = len(hard_negatives)
    hard_negatives = hard_negatives.drop_duplicates(subset=['ts_code', 't1_date'])
    after_dedup = len(hard_negatives)
    log.info(f"去重: {before_dedup} -> {after_dedup}")

    # 统计
    if 'sample_type' in hard_negatives.columns:
        type_counts = hard_negatives['sample_type'].value_counts()
        log.info("样本类型分布:")
        for t, c in type_counts.items():
            log.info(f"  {t}: {c}")

    # 保存
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    hard_negatives.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    log.info(f"已保存: {OUTPUT}")

    # 与现有硬负样本对比
    existing = Path("data/training/samples/hard_negative_samples.csv")
    if existing.exists():
        existing_df = pd.read_csv(existing)
        log.info(f"\n对比:")
        log.info(f"  现有硬负样本: {len(existing_df)}")
        log.info(f"  新增硬负样本: {len(hard_negatives)}")
        log.info(f"  增长倍数: {len(hard_negatives) / max(len(existing_df), 1):.1f}x")

    # 计算占负样本比例
    neg_df = pd.read_csv("data/training/samples/negative_samples_v2.csv")
    total_negative = len(neg_df)
    hard_pct = len(hard_negatives) / (total_negative + len(hard_negatives)) * 100
    log.info(f"\n硬负样本占比: {hard_pct:.1f}% (目标: 15-20%)")

if __name__ == '__main__':
    main()
