#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v3.1.0 筛选器快速验证脚本

仅用少量股票 + 短时间范围，快速验证 Breakout/Bounce 筛选逻辑：
- 验证是否能正常生成样本
- 验证样本分布是否合理
- 验证无异常报错

Usage:
    python scripts/validate_v310_screeners.py
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from src.utils.logger import log
from src.data.data_manager import DataManager
from src.models.screening import BreakoutSampleScreener, BounceSampleScreener

# 快速验证配置：100只股票 + 2023年1年数据
MAX_STOCKS = 100
START_DATE = "20230101"
END_DATE = "20231231"


def validate_breakout():
    log.info("=" * 60)
    log.info("BreakoutScorer 快速验证")
    log.info(f"  股票数: {MAX_STOCKS} | 时间: {START_DATE} ~ {END_DATE}")
    log.info("=" * 60)

    dm = DataManager(use_cache=True)
    screener = BreakoutSampleScreener(data_manager=dm)

    # 正样本
    df_pos = screener.screen_positive_samples(START_DATE, END_DATE, max_samples=200)
    log.success(f"正样本: {len(df_pos)} 个")
    if not df_pos.empty:
        log.info(f"  列: {list(df_pos.columns)}")
        log.info(f"  年度分布: {df_pos['t1_date'].str[:4].value_counts().to_dict()}")

    # 硬负样本
    df_hard = screener.screen_hard_negative_samples(START_DATE, END_DATE, target_count=100)
    log.success(f"硬负样本: {len(df_hard)} 个")
    if not df_hard.empty and "hard_negative_type" in df_hard.columns:
        log.info(f"  类型分布: {df_hard['hard_negative_type'].value_counts().to_dict()}")

    return df_pos, df_hard


def validate_bounce():
    log.info("\n" + "=" * 60)
    log.info("BounceScorer 快速验证")
    log.info(f"  股票数: {MAX_STOCKS} | 时间: {START_DATE} ~ {END_DATE}")
    log.info("=" * 60)

    dm = DataManager(use_cache=True)
    screener = BounceSampleScreener(data_manager=dm)

    df_pos = screener.screen_positive_samples(START_DATE, END_DATE, max_samples=200)
    log.success(f"正样本: {len(df_pos)} 个")
    if not df_pos.empty:
        log.info(f"  列: {list(df_pos.columns)}")
        log.info(f"  年度分布: {df_pos['t1_date'].str[:4].value_counts().to_dict()}")

    df_hard = screener.screen_hard_negative_samples(START_DATE, END_DATE, target_count=100)
    log.success(f"硬负样本: {len(df_hard)} 个")
    if not df_hard.empty and "hard_negative_type" in df_hard.columns:
        log.info(f"  类型分布: {df_hard['hard_negative_type'].value_counts().to_dict()}")

    return df_pos, df_hard


def main():
    log.info("v3.1.0 筛选器快速验证启动")

    # 限制股票数量以加速验证
    import src.models.screening.breakout_sample_screener as breakout_mod
    import src.models.screening.bounce_sample_screener as bounce_mod

    # Monkey-patch _get_eligible_stocks 以限制股票数量
    original_get_eligible = BreakoutSampleScreener._get_eligible_stocks

    def limited_get_eligible(self):
        df = original_get_eligible(self)
        if len(df) > MAX_STOCKS:
            df = df.head(MAX_STOCKS)
            log.info(f"限制股票数量: {MAX_STOCKS} 只")
        return df

    BreakoutSampleScreener._get_eligible_stocks = limited_get_eligible
    BounceSampleScreener._get_eligible_stocks = limited_get_eligible

    breakout_pos, breakout_hard = validate_breakout()
    bounce_pos, bounce_hard = validate_bounce()

    log.info("\n" + "=" * 60)
    log.info("验证总结")
    log.info("=" * 60)
    log.info(f"Breakout: 正样本={len(breakout_pos)}, 硬负={len(breakout_hard)}")
    log.info(f"Bounce:   正样本={len(bounce_pos)}, 硬负={len(bounce_hard)}")

    # 基本门槛检查
    ok = True
    for name, pos, hard in [("Breakout", breakout_pos, breakout_hard),
                            ("Bounce", bounce_pos, bounce_hard)]:
        if len(pos) < 10:
            log.warning(f"{name} 正样本过少 ({len(pos)} < 10)")
            ok = False
        if len(hard) < 5:
            log.warning(f"{name} 硬负样本过少 ({len(hard)} < 5)")
            ok = False

    if ok:
        log.success("✅ 筛选器验证通过，可以全量运行")
    else:
        log.warning("⚠️ 样本数量偏低，建议检查参数或数据范围")


if __name__ == "__main__":
    main()
