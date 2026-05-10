#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.9.5+ 综合样本生成脚本

基于新的样本定义决策，一次性生成正/负/硬负三类样本：
1. 正样本：300天上市、40%反追龙头上限、时间分布均匀性降采样
2. 负样本：同T1日期、市值分层匹配、2:1比例
3. 硬负样本：near_miss 20-40%、high_position_fail消除未来函数、动态配额15-20%

原则：所有数据基于真实API，不模拟任何数据。

Usage:
    python scripts/prepare_v295_samples.py

Output:
    data/training/samples/positive_samples_v295.csv
    data/training/samples/negative_samples_v295.csv
    data/training/samples/hard_negatives_v295.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from src.utils.logger import log
from src.data.data_manager import DataManager
from src.models.screening.positive_sample_screener import PositiveSampleScreener
from src.models.screening.negative_sample_screener_v2 import NegativeSampleScreenerV2
from src.models.screening.hard_negative_screener import HardNegativeSampleScreener

# ============================================================================
# 配置
# ============================================================================
SAMPLES_DIR = PROJECT_ROOT / "data" / "training" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# 时间分布均匀性：降采样目标（按季度）
# v295c: 关闭降采样，保留全部正样本
TEMPORAL_BALANCE_ENABLED = False

# 硬负动态配额系数（硬负总数 = 正样本数 × HARD_NEG_RATIO）
HARD_NEG_RATIO = 0.6  # 约15-20%的总样本比例


# ============================================================================
# 正样本生成
# ============================================================================
def generate_positive_samples():
    """生成正样本"""
    log.info("=" * 80)
    log.info("阶段1: 生成正样本")
    log.info("  - 上市天数 ≥ 180天 (对齐v2.7.0/v2.3.2)")
    log.info("  - 反追龙头约束: 已关闭 (与v2.7.0保持一致)")
    log.info("  - 时间分布均匀性: 已关闭 (保留全部正样本)")
    log.info("=" * 80)

    dm = DataManager(use_cache=False)  # 已删除 SQLite 缓存，改用 ArcticDB
    # v295d: 上市天数改回180，与v2.7.0/v2.3.2保持一致
    screener = PositiveSampleScreener(
        data_manager=dm,
        config={
            "enable_anti_chasing": False,  # 关闭反追龙头约束
            "min_listing_days": 180,  # 对齐v2.7.0/v2.3.2
        }
    )

    # 生成正样本
    df_pos = screener.screen_all_stocks(start_date="20000101")

    if df_pos.empty:
        log.error("正样本生成失败")
        return None

    log.success(f"原始正样本: {len(df_pos)} 个")

    # 时间分布均匀性
    if TEMPORAL_BALANCE_ENABLED:
        df_pos = apply_temporal_balance(df_pos)

    # 保存
    output_path = SAMPLES_DIR / "positive_samples_v295.csv"
    df_pos.to_csv(output_path, index=False)
    log.success(f"正样本已保存: {output_path}")

    return df_pos


def apply_temporal_balance(df):
    """按季度降采样，控制时间分布均匀性"""
    df = df.copy()
    df["t1_date"] = pd.to_datetime(df["t1_date"].astype(str), format="%Y%m%d", errors="coerce")
    df["year_quarter"] = df["t1_date"].dt.to_period("Q")

    # 计算目标数量
    total = len(df)
    n_quarters = df["year_quarter"].nunique()
    # v295b: 提高每季度上限，从自动计算改为固定 50，显著扩大样本量
    target_per_quarter = max(50, total // n_quarters)

    log.info(f"时间分布均匀化: {n_quarters} 个季度, 目标每季度约 {target_per_quarter} 个")

    balanced = []
    for quarter, group in df.groupby("year_quarter"):
        if len(group) > target_per_quarter:
            # 降采样
            sampled = group.sample(n=target_per_quarter, random_state=42)
            log.info(f"  {quarter}: {len(group)} -> {target_per_quarter} (降采样)")
        else:
            sampled = group
            log.info(f"  {quarter}: {len(group)} (保留全部)")
        balanced.append(sampled)

    df_balanced = pd.concat(balanced, ignore_index=True)
    df_balanced = df_balanced.drop(columns=["year_quarter"], errors="ignore")
    log.success(f"均匀化后正样本: {len(df_balanced)} 个 (原 {total} 个)")

    return df_balanced


# ============================================================================
# 负样本生成
# ============================================================================
def generate_negative_samples(df_pos):
    """生成负样本"""
    log.info("")
    log.info("=" * 80)
    log.info("阶段2: 生成负样本")
    log.info("  - 同T1日期其他股票")
    log.info("  - 市值分层采样（默认启用）")
    log.info("  - 负/正比例 2:1")
    log.info("=" * 80)

    dm = DataManager(use_cache=False)  # 已删除 SQLite 缓存，改用 ArcticDB
    screener = NegativeSampleScreenerV2(data_manager=dm)

    df_neg = screener.screen_negative_samples(
        positive_samples_df=df_pos,
        samples_per_positive=2,
        stratified_by_mv=True,
    )

    if df_neg.empty:
        log.error("负样本生成失败")
        return None

    log.success(f"负样本: {len(df_neg)} 个")

    # 保存
    output_path = SAMPLES_DIR / "negative_samples_v295.csv"
    df_neg.to_csv(output_path, index=False)
    log.success(f"负样本已保存: {output_path}")

    return df_neg


# ============================================================================
# 硬负样本生成
# ============================================================================
def generate_hard_negative_samples(df_pos):
    """生成硬负样本（false_breakout + high_position_fail + near_miss配额）"""
    log.info("")
    log.info("=" * 80)
    log.info("阶段3: 生成硬负样本")
    log.info("  - near_miss: 34日涨幅20-40%, 每日3只 (主类型，对齐v2.7.0)")
    log.info("  - high_position_fail: T1前已涨≥20% + 当日冲高回落(上影线>3%), 每日1只")
    log.info("  - false_breakout: 已移除")
    log.info("=" * 80)

    dm = DataManager(use_cache=False)  # 已删除 SQLite 缓存，改用 ArcticDB
    screener = HardNegativeSampleScreener(data_manager=dm)

    # 生成硬负样本：对齐v2.7.0，near_miss为主 + 少量high_position_fail
    df_hard = screener.screen_hard_negatives(
        positive_samples_df=df_pos,
        min_return=20.0,
        max_return=40.0,
        near_miss_per_date=3,
        high_position_fail_per_date=1,
        include_high_position_fail=True,
        include_false_breakout=False,
    )

    if df_hard.empty:
        log.error("硬负样本生成失败")
        return None

    log.success(f"硬负样本: {len(df_hard)} 个")

    # 保存
    output_path = SAMPLES_DIR / "hard_negatives_v295.csv"
    df_hard.to_csv(output_path, index=False)
    log.success(f"硬负样本已保存: {output_path}")

    return df_hard


# ============================================================================
# 验证
# ============================================================================
def validate_samples(df_pos, df_neg, df_hard):
    """验证样本分布是否符合预期"""
    log.info("")
    log.info("=" * 80)
    log.info("阶段4: 样本分布验证")
    log.info("=" * 80)

    total = len(df_pos) + len(df_neg) + len(df_hard)
    hard_ratio = len(df_hard) / total if total > 0 else 0
    neg_pos_ratio = len(df_neg) / len(df_pos) if len(df_pos) > 0 else 0

    log.info(f"  正样本: {len(df_pos)}")
    log.info(f"  负样本: {len(df_neg)}")
    log.info(f"  硬负样本: {len(df_hard)}")
    log.info(f"  总样本: {total}")
    log.info(f"  负/正比例: {neg_pos_ratio:.2f} (目标 2.0)")
    log.info(f"  硬负比例: {hard_ratio:.1%} (目标 15%-20%)")

    # 时间分布
    df_pos["t1_date"] = pd.to_datetime(df_pos["t1_date"].astype(str), format="%Y%m%d", errors="coerce")
    df_pos["year"] = df_pos["t1_date"].dt.year
    yearly_counts = df_pos["year"].value_counts().sort_index()
    log.info(f"\n正样本时间分布:")
    for year, count in yearly_counts.head(10).items():
        log.info(f"  {year}: {count} 个")
    if len(yearly_counts) > 10:
        log.info(f"  ... 共 {len(yearly_counts)} 个年份")

    # 硬负类型分布
    if "sample_type" in df_hard.columns:
        type_counts = df_hard["sample_type"].value_counts()
        log.info(f"\n硬负样本类型分布:")
        for t, c in type_counts.items():
            log.info(f"  {t}: {c} 个 ({c/len(df_hard):.1%})")

    # 校验结果
    checks = []
    if 1.8 <= neg_pos_ratio <= 2.2:
        checks.append(("✅", "负/正比例", f"{neg_pos_ratio:.2f}", "在目标范围 1.8-2.2 内"))
    else:
        checks.append(("⚠️", "负/正比例", f"{neg_pos_ratio:.2f}", "偏离目标 2.0"))

    if 0.15 <= hard_ratio <= 0.20:
        checks.append(("✅", "硬负比例", f"{hard_ratio:.1%}", "在目标范围 15%-20% 内"))
    else:
        checks.append(("⚠️", "硬负比例", f"{hard_ratio:.1%}", "偏离目标 15%-20%"))

    log.info("")
    for status, name, value, note in checks:
        log.info(f"  {status} {name}: {value} - {note}")

    log.info("=" * 80)

    return all(c[0] == "✅" for c in checks)


# ============================================================================
# 主流程
# ============================================================================
def main():
    log.info("=" * 80)
    log.info("v2.9.5+ 综合样本生成")
    log.info("原则: 所有数据基于真实API，不模拟任何数据")
    log.info("=" * 80)

    # 1. 正样本（断点续传：如果文件已存在则直接加载）
    pos_path = SAMPLES_DIR / "positive_samples_v295.csv"
    if pos_path.exists():
        log.info(f"检测到已存在的正样本文件，直接加载: {pos_path}")
        df_pos = pd.read_csv(pos_path)
        log.success(f"已加载正样本: {len(df_pos)} 个")
    else:
        df_pos = generate_positive_samples()
        if df_pos is None:
            log.error("正样本生成失败，终止")
            return

    # 2. 负样本（断点续传）
    neg_path = SAMPLES_DIR / "negative_samples_v295.csv"
    if neg_path.exists():
        log.info(f"检测到已存在的负样本文件，直接加载: {neg_path}")
        df_neg = pd.read_csv(neg_path)
        log.success(f"已加载负样本: {len(df_neg)} 个")
    else:
        df_neg = generate_negative_samples(df_pos)
        if df_neg is None:
            log.error("负样本生成失败，终止")
            return

    # 3. 硬负样本（断点续传）
    hard_path = SAMPLES_DIR / "hard_negatives_v295.csv"
    if hard_path.exists():
        log.info(f"检测到已存在的硬负样本文件，直接加载: {hard_path}")
        df_hard = pd.read_csv(hard_path)
        log.success(f"已加载硬负样本: {len(df_hard)} 个")
    else:
        df_hard = generate_hard_negative_samples(df_pos)
        if df_hard is None:
            log.error("硬负样本生成失败，终止")
            return

    # 4. 去重：删除硬负样本中与负样本重叠的 (ts_code, t1_date)
    neg_keys = set(zip(df_neg["ts_code"], df_neg["t1_date"].astype(str)))
    hard_keys = set(zip(df_hard["ts_code"], df_hard["t1_date"].astype(str)))
    overlap = hard_keys & neg_keys
    if overlap:
        log.warning(f"发现 {len(overlap)} 个硬负样本与负样本 (股票,日期) 重叠，正在删除...")
        overlap_set = {(ts, str(td)) for ts, td in overlap}
        mask = df_hard.apply(lambda r: (r["ts_code"], str(r["t1_date"])) not in overlap_set, axis=1)
        df_hard = df_hard[mask].copy()
        df_hard.to_csv(hard_path, index=False)
        log.success(f"去重后硬负样本: {len(df_hard)} 个")
    else:
        log.info("硬负样本与负样本无重叠，通过")

    # 5. 验证
    all_pass = validate_samples(df_pos, df_neg, df_hard)

    log.info("")
    if all_pass:
        log.success("✅ 全部样本验证通过！")
        log.info("下一步: 运行特征提取脚本")
    else:
        log.warning("⚠️ 部分验证未通过，请检查样本分布")


if __name__ == "__main__":
    main()
