#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成更多硬负样本

目标：将硬负样本占比从11.6%提升到20%+
当前：998个硬负样本
目标：约2500个硬负样本（占总负样本的20%）
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
from src.models.screening.hard_negative_screener import HardNegativeSampleScreener


def generate_additional_hard_negatives():
    """生成额外的硬负样本"""
    log.info("=" * 80)
    log.info("生成额外硬负样本 - 目标占比20%+")
    log.info("=" * 80)

    # 加载现有数据统计
    pos_file = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v5.csv"
    neg_file = PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v5.csv"
    hard_neg_file = PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v5.csv"

    df_pos = pd.read_csv(pos_file)
    df_neg = pd.read_csv(neg_file)
    df_hard_neg = pd.read_csv(hard_neg_file)

    pos_samples = df_pos["sample_id"].nunique()
    neg_rows = len(df_neg)
    hard_neg_samples = df_hard_neg["sample_id"].nunique()

    log.info("当前数据统计:")
    log.info(f"  正样本: {pos_samples} 个")
    log.info(f"  负样本: {neg_rows} 行")
    log.info(f"  硬负样本: {hard_neg_samples} 个")

    # 计算目标硬负样本数量
    # 目标：硬负样本占总负样本的20%
    total_neg_samples = neg_rows // 34 + hard_neg_samples  # 估算负样本数
    target_hard_neg = int(total_neg_samples * 0.25)  # 目标25%
    additional_needed = target_hard_neg - hard_neg_samples

    log.info(f"\n目标硬负样本数: {target_hard_neg} (占比25%)")
    log.info(f"需要额外生成: {additional_needed} 个")

    if additional_needed <= 0:
        log.info("硬负样本已足够，无需额外生成")
        return

    # 初始化
    dm = DataManager()
    screener = HardNegativeSampleScreener(dm)

    # 准备正样本数据
    positive_samples = (
        df_pos.groupby("sample_id")
        .agg({"ts_code": "first", "name": "first", "trade_date": "max", "circ_mv": "first"})
        .reset_index()
    )
    positive_samples["trade_date"] = pd.to_datetime(positive_samples["trade_date"])
    positive_samples["t1_date"] = positive_samples["trade_date"].dt.strftime("%Y%m%d")

    # 筛选更多硬负样本
    log.info("\n开始筛选额外硬负样本...")
    log.info("  - near_miss: 每日25只 (原15只)")
    log.info("  - high_position_fail: 每日25只 (原15只)")
    log.info("  - false_breakout: 每日20只 (原10只)")

    # 使用不同的随机种子生成新样本
    additional_hard_negatives = screener.screen_hard_negatives(
        positive_samples_df=positive_samples,
        min_return=15.0,  # 降低门槛，获取更多样本
        max_return=48.0,  # 提高上限
        samples_per_date={"near_miss": 25, "high_position_fail": 25, "false_breakout": 20},
        random_seed=123,  # 不同的随机种子
        include_high_position_fail=True,
        include_false_breakout=True,
    )

    if additional_hard_negatives.empty:
        log.error("额外硬负样本筛选失败")
        return

    log.info(f"筛选到 {len(additional_hard_negatives)} 个额外硬负样本")

    # 统计类型分布
    if "sample_type" in additional_hard_negatives.columns:
        type_counts = additional_hard_negatives["sample_type"].value_counts()
        log.info("\n额外硬负样本类型分布:")
        for sample_type, count in type_counts.items():
            log.info(f"  - {sample_type}: {count} 个")

    # 提取特征
    log.info("\n开始提取额外硬负样本特征...")
    additional_features = screener.extract_features(
        hard_negative_samples_df=additional_hard_negatives, lookback_days=34
    )

    if additional_features.empty:
        log.error("额外硬负样本特征提取失败")
        return

    # 合并现有硬负样本
    log.info("\n合并硬负样本...")

    # 确保列一致
    common_cols = list(set(df_hard_neg.columns) & set(additional_features.columns))

    # 更新sample_id避免冲突
    max_sample_id = df_hard_neg["sample_id"].max()
    additional_features["sample_id"] = additional_features["sample_id"] + max_sample_id + 1

    # 合并
    df_combined = pd.concat([df_hard_neg[common_cols], additional_features[common_cols]], ignore_index=True)

    # 去重（基于ts_code和trade_date）
    if "ts_code" in df_combined.columns and "trade_date" in df_combined.columns:
        before_dedup = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=["ts_code", "trade_date"])
        after_dedup = len(df_combined)
        log.info(f"去重: {before_dedup} -> {after_dedup} 行")

    # 保存
    output_file = PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v5_extended.csv"
    df_combined.to_csv(output_file, index=False)

    final_samples = df_combined["sample_id"].nunique()
    final_rows = len(df_combined)

    log.success("\n✓ 扩展硬负样本生成完成!")
    log.info(f"  原硬负样本: {hard_neg_samples} 个")
    log.info(f"  扩展后: {final_samples} 个样本, {final_rows} 行")
    log.info(f"  输出文件: {output_file}")

    # 计算新的占比
    new_total_neg = neg_rows // 34 + final_samples
    new_ratio = final_samples / new_total_neg * 100
    log.info(f"  新硬负样本占比: {new_ratio:.1f}%")

    return output_file


if __name__ == "__main__":
    generate_additional_hard_negatives()
