#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动化样本生成（阶段2）
在数据补全完成后执行，生成2025-12-27至2026-04-14的新样本并与历史样本合并。
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager
from src.models.screening.positive_sample_screener import PositiveSampleScreener
from src.utils.logger import log

def main():
    log.info("=" * 80)
    log.info("v2.8.0 自动化样本生成")
    log.info("=" * 80)

    dm = DataManager()

    # 1. 生成新增正样本
    log.info("\n[1/3] 扫描新增正样本 (20251227 ~ 20260414)...")
    screener = PositiveSampleScreener(dm)
    df_new_pos = screener.screen_all_stocks(start_date="20251227", end_date="20260414")

    if df_new_pos is not None and not df_new_pos.empty:
        log.info(f"新增正样本: {len(df_new_pos)} 条")
        df_new_pos.to_csv(PROJECT_ROOT / "data" / "training" / "samples" / "positive_samples_new.csv", index=False)
    else:
        log.warning("新增正样本为空")
        df_new_pos = pd.DataFrame()

    # 2. 合并历史正样本
    log.info("\n[2/3] 合并历史正样本...")
    df_old_pos = pd.read_csv(PROJECT_ROOT / "data" / "training" / "samples" / "positive_samples.csv")
    log.info(f"历史正样本: {len(df_old_pos)} 条")

    if not df_new_pos.empty:
        df_pos = pd.concat([df_old_pos, df_new_pos], ignore_index=True)
        df_pos = df_pos.drop_duplicates(subset=["ts_code", "t1_date"])
        log.info(f"合并后正样本: {len(df_pos)} 条 (去重前: {len(df_old_pos) + len(df_new_pos)})")
    else:
        df_pos = df_old_pos.copy()

    df_pos.to_csv(PROJECT_ROOT / "data" / "training" / "samples" / "positive_samples_v280.csv", index=False)
    log.success("✓ 正样本合并完成")

    # 3. 负样本（简化为拷贝并记录，负样本筛选需要feature_stats，较复杂）
    log.info("\n[3/3] 负样本处理（待完整实现）...")
    log.info("负样本筛选需要正样本特征统计，建议数据补全后手动执行 screen_negative_samples_v2.py")

    log.info("\n阶段2完成。待执行:")
    log.info("  - 负样本筛选: python scripts/screen_negative_samples_v2.py")
    log.info("  - 特征工程: 确认 enrich_features_v6.py 或 add_advanced_factors 的调用方式")

if __name__ == "__main__":
    main()
