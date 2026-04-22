#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 v6 版本样本数据

v6 版本优化：
1. 历史数据天数从34天扩展到70天（支持55日长期特征）
2. 增加硬负样本数量 - 目标占比 15-20%
3. 新增伪突破类型 - 突破后5日内回落>5%

（市值分层采样已移除，因为市值特征重要性仅2.76%-3.85%，影响有限）

输出文件：
- data/training/processed/feature_data_34d_v6.csv (正样本，70天数据)
- data/training/features/negative_feature_data_v2_34d_v6.csv (负样本，随机采样+70天)
- data/training/features/hard_negative_feature_data_34d_v6.csv (硬负样本，增加数量+伪突破+70天)
"""
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
from src.models.screening.negative_sample_screener_v2 import NegativeSampleScreenerV2
from src.models.screening.hard_negative_screener import HardNegativeSampleScreener
from src.models.screening.positive_sample_screener import PositiveSampleScreener

# v6配置
LOOKBACK_DAYS = 70  # 从34天扩展到70天


def generate_positive_samples_v6():
    """重新生成正样本（70天历史数据）"""
    log.info("=" * 80)
    log.info("步骤1: 重新生成正样本 v6（70天历史数据）")
    log.info("=" * 80)

    # 初始化
    dm = DataManager()
    screener = PositiveSampleScreener(dm)

    # 加载已有的正样本列表（从v5获取样本信息，但重新提取特征）
    v5_file = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v5.csv"
    if not v5_file.exists():
        log.error(f"v5正样本文件不存在: {v5_file}")
        return None

    df_v5 = pd.read_csv(v5_file)

    # 获取唯一样本列表
    samples = (
        df_v5.groupby("sample_id")
        .agg({"ts_code": "first", "name": "first", "trade_date": "max"})  # 取最后一天作为T1前一天
        .reset_index()
    )

    # 将trade_date转换为t1_date，并确保格式正确
    # 注意：v5数据中的trade_date可能是YYYY-MM-DD格式，需要转换为YYYYMMDD
    samples["trade_date"] = pd.to_datetime(samples["trade_date"])
    samples["t1_date"] = samples["trade_date"].dt.strftime("%Y%m%d")

    log.info(f"正样本数量: {len(samples)}")
    log.info(f"历史数据天数: {LOOKBACK_DAYS}")

    # 提取特征（使用70天）
    log.info("\n开始提取正样本特征...")
    positive_features = screener.extract_features(samples_df=samples, lookback_days=LOOKBACK_DAYS)

    if positive_features.empty:
        log.error("正样本特征提取失败")
        return None

    # 添加label列
    positive_features["label"] = 1

    # 保存
    output_file = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v6.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    positive_features.to_csv(output_file, index=False)

    n_samples = positive_features["sample_id"].nunique()
    n_rows = len(positive_features)
    log.success(f"✓ 正样本生成完成: {n_samples} 个样本, {n_rows} 条记录")
    log.info(f"  输出文件: {output_file}")

    return positive_features


def generate_negative_samples_v6(positive_df: pd.DataFrame):
    """生成 v6 版本负样本（随机采样 + 70天历史数据）"""
    log.info("\n" + "=" * 80)
    log.info("步骤2: 生成负样本 v6（随机采样 + 70天历史数据）")
    log.info("=" * 80)

    # 初始化数据管理器和筛选器
    dm = DataManager()
    screener = NegativeSampleScreenerV2(dm)

    # 准备正样本数据（用于获取T1日期和市值分布）
    positive_samples = (
        positive_df.groupby("sample_id")
        .agg({"ts_code": "first", "name": "first", "trade_date": "max", "circ_mv": "first"})  # 取最后一天
        .reset_index()
    )

    # 将最后一天的trade_date作为t1_date，确保格式正确
    positive_samples["trade_date"] = pd.to_datetime(positive_samples["trade_date"])
    positive_samples["t1_date"] = positive_samples["trade_date"].dt.strftime("%Y%m%d")

    # 计算市值分位数
    if "circ_mv" in positive_samples.columns:
        pos_mv = positive_samples["circ_mv"].dropna()
        if len(pos_mv) > 0:
            mv_quantiles = pos_mv.quantile([0.25, 0.5, 0.75]).tolist()
            log.info(f"正样本市值分位数: {[f'{q:.0f}' for q in mv_quantiles]}")
        else:
            mv_quantiles = None
            log.warning("正样本市值数据为空，将使用随机采样")
    else:
        mv_quantiles = None
        log.warning("正样本无市值数据，将使用随机采样")

    # 筛选负样本（使用随机采样，不做市值分层）
    log.info("\n开始筛选负样本...")
    negative_samples = screener.screen_negative_samples(
        positive_samples_df=positive_samples,
        samples_per_positive=1,
        random_seed=42,
        stratified_by_mv=False,  # 禁用市值分层采样
        mv_quantiles=None,
    )

    if negative_samples.empty:
        log.error("负样本筛选失败")
        return None

    log.info(f"筛选到 {len(negative_samples)} 个负样本")

    # 提取特征（使用70天）
    log.info("\n开始提取负样本特征...")
    negative_features = screener.extract_features(negative_samples_df=negative_samples, lookback_days=LOOKBACK_DAYS)

    if negative_features.empty:
        log.error("负样本特征提取失败")
        return None

    # 保存
    output_file = PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v6.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    negative_features.to_csv(output_file, index=False)

    n_samples = negative_features["sample_id"].nunique()
    n_rows = len(negative_features)
    log.success(f"✓ 负样本生成完成: {n_samples} 个样本, {n_rows} 条记录")
    log.info(f"  输出文件: {output_file}")

    # 验证市值分布
    if "circ_mv" in negative_features.columns and mv_quantiles:
        neg_mv = negative_features.groupby("sample_id")["circ_mv"].first().dropna()
        if len(neg_mv) > 0 and len(pos_mv) > 0:
            pos_mean = pos_mv.mean()
            neg_mean = neg_mv.mean()
            bias = (neg_mean - pos_mean) / pos_mean * 100
            log.info("\n市值分布验证:")
            log.info(f"  正样本均值: {pos_mean:.0f}")
            log.info(f"  负样本均值: {neg_mean:.0f}")
            log.info(f"  偏差: {bias:+.1f}%")

    return negative_features


def generate_hard_negative_samples_v6(positive_df: pd.DataFrame):
    """生成 v6 版本硬负样本（增加数量 + 伪突破 + 70天历史数据）"""
    log.info("\n" + "=" * 80)
    log.info("步骤3: 生成硬负样本 v6（增加数量 + 伪突破 + 70天历史数据）")
    log.info("=" * 80)

    # 初始化数据管理器和筛选器
    dm = DataManager()
    screener = HardNegativeSampleScreener(dm)

    # 准备正样本数据（需要添加t1_date列）
    positive_samples = (
        positive_df.groupby("sample_id")
        .agg({"ts_code": "first", "name": "first", "trade_date": "max", "circ_mv": "first"})  # 取最后一天
        .reset_index()
    )
    positive_samples["trade_date"] = pd.to_datetime(positive_samples["trade_date"])
    positive_samples["t1_date"] = positive_samples["trade_date"].dt.strftime("%Y%m%d")

    # 筛选硬负样本（启用所有类型）
    log.info("\n开始筛选硬负样本...")
    log.info("  - near_miss: 每日15只")
    log.info("  - high_position_fail: 每日15只")
    log.info("  - false_breakout: 每日10只")

    hard_negative_samples = screener.screen_hard_negatives(
        positive_samples_df=positive_samples,
        min_return=20.0,
        max_return=45.0,
        samples_per_date=None,  # 使用默认值（v3增加后的数量）
        random_seed=42,
        include_high_position_fail=True,
        include_false_breakout=True,  # v3新增
    )

    if hard_negative_samples.empty:
        log.error("硬负样本筛选失败")
        return None

    log.info(f"筛选到 {len(hard_negative_samples)} 个硬负样本")

    # 统计各类型数量
    if "sample_type" in hard_negative_samples.columns:
        type_counts = hard_negative_samples["sample_type"].value_counts()
        log.info("\n硬负样本类型分布:")
        for sample_type, count in type_counts.items():
            log.info(f"  - {sample_type}: {count} 个")

    # 提取特征（使用70天）
    log.info("\n开始提取硬负样本特征...")
    hard_negative_features = screener.extract_features(
        hard_negative_samples_df=hard_negative_samples, lookback_days=LOOKBACK_DAYS
    )

    if hard_negative_features.empty:
        log.error("硬负样本特征提取失败")
        return None

    # 保存
    output_file = PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v6.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    hard_negative_features.to_csv(output_file, index=False)

    n_samples = hard_negative_features["sample_id"].nunique()
    n_rows = len(hard_negative_features)
    log.success(f"✓ 硬负样本生成完成: {n_samples} 个样本, {n_rows} 条记录")
    log.info(f"  输出文件: {output_file}")

    return hard_negative_features


def verify_sample_distribution():
    """验证 v6 样本分布"""
    log.info("\n" + "=" * 80)
    log.info("步骤4: 验证 v6 样本分布")
    log.info("=" * 80)

    # 加载数据
    pos_file = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v6.csv"
    neg_file = PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v6.csv"
    hard_file = PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v6.csv"

    df_pos = pd.read_csv(pos_file)
    df_neg = pd.read_csv(neg_file)
    df_hard = pd.read_csv(hard_file)

    pos_count = df_pos["sample_id"].nunique()
    neg_count = df_neg["sample_id"].nunique()
    hard_count = df_hard["sample_id"].nunique()

    pos_rows = len(df_pos)
    neg_rows = len(df_neg)
    hard_rows = len(df_hard)

    total_neg = neg_count + hard_count
    ratio = total_neg / pos_count
    hard_ratio = hard_count / total_neg * 100

    log.info("\n【v6 样本统计】")
    log.info(f"  正样本: {pos_count} 个样本, {pos_rows} 条记录 ({pos_rows/pos_count:.0f}天/样本)")
    log.info(f"  负样本: {neg_count} 个样本, {neg_rows} 条记录 ({neg_rows/neg_count:.0f}天/样本)")
    log.info(f"  硬负样本: {hard_count} 个样本, {hard_rows} 条记录 ({hard_rows/hard_count:.0f}天/样本)")
    log.info(f"  总负样本: {total_neg}")
    log.info(f"  正负比例: 1:{ratio:.2f}")
    log.info(f"  硬负样本占比: {hard_ratio:.1f}%")

    # 市值分布对比
    if "circ_mv" in df_pos.columns and "circ_mv" in df_neg.columns:
        pos_mv = df_pos.groupby("sample_id")["circ_mv"].first().dropna()
        neg_mv = df_neg.groupby("sample_id")["circ_mv"].first().dropna()

        if len(pos_mv) > 0 and len(neg_mv) > 0:
            bias = (neg_mv.mean() - pos_mv.mean()) / pos_mv.mean() * 100
            log.info("\n【市值分布】")
            log.info(f"  正样本均值: {pos_mv.mean():.0f}")
            log.info(f"  负样本均值: {neg_mv.mean():.0f}")
            log.info(f"  偏差: {bias:+.1f}%")

            if abs(bias) < 30:
                log.success("  ✓ 市值偏差在可接受范围内")
            else:
                log.warning("  ⚠ 市值偏差仍较大，建议进一步优化")

    # 硬负样本比例检查
    if hard_ratio >= 15:
        log.success("  ✓ 硬负样本比例达标 (目标>=15%)")
    else:
        log.warning("  ⚠ 硬负样本比例未达标 (目标>=15%)")

    return {
        "positive": pos_count,
        "negative": neg_count,
        "hard_negative": hard_count,
        "ratio": ratio,
        "hard_ratio": hard_ratio,
    }


def main():
    start_time = datetime.now()

    log.info("=" * 80)
    log.info("生成 v6 版本样本数据")
    log.info("=" * 80)
    log.info(f"开始时间: {start_time}")
    log.info(f"历史数据天数: {LOOKBACK_DAYS}")
    log.info("")

    # 步骤1: 重新生成正样本（70天数据）
    positive_df = generate_positive_samples_v6()
    if positive_df is None:
        log.error("正样本生成失败，退出")
        return

    # 步骤2: 生成负样本（市值分层采样 + 70天数据）
    negative_df = generate_negative_samples_v6(positive_df)
    if negative_df is None:
        log.error("负样本生成失败，退出")
        return

    # 步骤3: 生成硬负样本（增加数量 + 伪突破 + 70天数据）
    hard_negative_df = generate_hard_negative_samples_v6(positive_df)
    if hard_negative_df is None:
        log.error("硬负样本生成失败，退出")
        return

    # 步骤4: 验证分布
    stats = verify_sample_distribution()

    end_time = datetime.now()
    duration = end_time - start_time

    log.info("\n" + "=" * 80)
    log.success("✓ v6 样本数据生成完成！")
    log.info("=" * 80)
    log.info(f"耗时: {duration}")
    log.info("\n下一步: 运行特征对齐和增强脚本")
    log.info("  1. python scripts/align_all_sample_features.py --version v6")
    log.info("  2. python scripts/enrich_breakout_features.py --version v6")
    log.info("  3. python scripts/enrich_market_features.py --version v6")
    log.info("  4. python scripts/enrich_interaction_features.py --version v6")


if __name__ == "__main__":
    main()
