#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
负样本策略优化 v3

优化内容：
1. 市值分层采样 - 按正样本的市值分布来抽样负样本，避免市值偏差
2. 增加硬负样本类型 - 伪突破样本（突破后5日内回落>5%）
3. 样本平衡 - 调整正负样本比例和硬负样本权重

此脚本用于分析当前样本分布并提供优化建议。
实际的负样本重新生成需要较长时间，建议作为后续迭代任务。
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.logger import log


def analyze_sample_distribution():
    """分析当前样本分布"""
    log.info("=" * 80)
    log.info("分析当前样本分布")
    log.info("=" * 80)

    # 加载数据
    positive_file = PROJECT_ROOT / "data" / "training" / "processed" / "feature_data_34d_v5.csv"
    negative_file = PROJECT_ROOT / "data" / "training" / "features" / "negative_feature_data_v2_34d_v5.csv"
    hard_negative_file = PROJECT_ROOT / "data" / "training" / "features" / "hard_negative_feature_data_34d_v5.csv"

    log.info("\n加载数据...")

    # 正样本
    df_pos = pd.read_csv(positive_file)
    pos_samples = df_pos.groupby("sample_id").first()
    log.info(f"正样本: {len(pos_samples)} 个样本")

    # 负样本
    df_neg = pd.read_csv(negative_file)
    neg_samples = df_neg.groupby("sample_id").first()
    log.info(f"负样本: {len(neg_samples)} 个样本")

    # 硬负样本
    df_hard = pd.read_csv(hard_negative_file)
    hard_samples = df_hard.groupby("sample_id").first()
    log.info(f"硬负样本: {len(hard_samples)} 个样本")

    # 样本比例
    total_neg = len(neg_samples) + len(hard_samples)
    ratio = total_neg / len(pos_samples)
    log.info(f"\n样本比例: 正:负 = 1:{ratio:.2f}")
    log.info(f"  - 普通负样本: {len(neg_samples)} ({len(neg_samples)/total_neg*100:.1f}%)")
    log.info(f"  - 硬负样本: {len(hard_samples)} ({len(hard_samples)/total_neg*100:.1f}%)")

    # 市值分布分析
    log.info("\n" + "=" * 80)
    log.info("市值分布分析")
    log.info("=" * 80)

    if "circ_mv" in pos_samples.columns:
        # 正样本市值分布
        pos_mv = pos_samples["circ_mv"].dropna()
        log.info("\n正样本市值分布:")
        log.info(f"  - 均值: {pos_mv.mean():.2f}")
        log.info(f"  - 中位数: {pos_mv.median():.2f}")
        log.info(f"  - 25%分位: {pos_mv.quantile(0.25):.2f}")
        log.info(f"  - 75%分位: {pos_mv.quantile(0.75):.2f}")

        # 负样本市值分布
        neg_mv = neg_samples["circ_mv"].dropna() if "circ_mv" in neg_samples.columns else pd.Series()
        if len(neg_mv) > 0:
            log.info("\n负样本市值分布:")
            log.info(f"  - 均值: {neg_mv.mean():.2f}")
            log.info(f"  - 中位数: {neg_mv.median():.2f}")
            log.info(f"  - 25%分位: {neg_mv.quantile(0.25):.2f}")
            log.info(f"  - 75%分位: {neg_mv.quantile(0.75):.2f}")

            # 市值偏差
            mv_bias = (neg_mv.mean() - pos_mv.mean()) / pos_mv.mean() * 100
            log.info(f"\n市值偏差: {mv_bias:+.1f}%")
            if abs(mv_bias) > 20:
                log.warning("⚠️ 市值偏差较大，建议进行市值分层采样")
            else:
                log.info("✓ 市值分布相对均衡")

    # 时间分布分析
    log.info("\n" + "=" * 80)
    log.info("时间分布分析")
    log.info("=" * 80)

    if "trade_date" in df_pos.columns:
        df_pos["trade_date"] = pd.to_datetime(df_pos["trade_date"])
        df_neg["trade_date"] = pd.to_datetime(df_neg["trade_date"])

        pos_years = df_pos.groupby("sample_id")["trade_date"].max().dt.year.value_counts().sort_index()
        neg_years = df_neg.groupby("sample_id")["trade_date"].max().dt.year.value_counts().sort_index()

        log.info("\n按年份分布:")
        log.info(f"{'年份':<8} {'正样本':>10} {'负样本':>10} {'比例':>10}")
        log.info("-" * 40)

        all_years = sorted(set(pos_years.index) | set(neg_years.index))
        for year in all_years:
            pos_count = pos_years.get(year, 0)
            neg_count = neg_years.get(year, 0)
            year_ratio = neg_count / pos_count if pos_count > 0 else 0
            log.info(f"{year:<8} {pos_count:>10} {neg_count:>10} {year_ratio:>10.2f}")

    return {
        "positive_count": len(pos_samples),
        "negative_count": len(neg_samples),
        "hard_negative_count": len(hard_samples),
        "ratio": ratio,
    }


def suggest_optimizations(stats: dict):
    """提供优化建议"""
    log.info("\n" + "=" * 80)
    log.info("优化建议")
    log.info("=" * 80)

    suggestions = []

    # 1. 样本比例建议
    if stats["ratio"] > 3:
        suggestions.append(
            {
                "priority": "中",
                "issue": f"负样本比例过高 (1:{stats['ratio']:.1f})",
                "suggestion": "考虑减少普通负样本数量，或增加正样本权重",
                "expected_impact": "减少模型对负样本的过拟合",
            }
        )
    elif stats["ratio"] < 2:
        suggestions.append(
            {
                "priority": "中",
                "issue": f"负样本比例偏低 (1:{stats['ratio']:.1f})",
                "suggestion": "考虑增加负样本数量",
                "expected_impact": "提高模型的泛化能力",
            }
        )

    # 2. 硬负样本比例建议
    hard_ratio = stats["hard_negative_count"] / (stats["negative_count"] + stats["hard_negative_count"])
    if hard_ratio < 0.1:
        suggestions.append(
            {
                "priority": "高",
                "issue": f"硬负样本比例过低 ({hard_ratio*100:.1f}%)",
                "suggestion": "增加硬负样本数量，特别是伪突破和高位假启动类型",
                "expected_impact": "提高模型对边界样本的区分能力",
            }
        )

    # 3. 伪突破样本建议
    suggestions.append(
        {
            "priority": "高",
            "issue": "缺少伪突破类型的硬负样本",
            "suggestion": '添加"突破后5日内回落>5%"的伪突破样本',
            "expected_impact": "减少模型对假突破的误判",
        }
    )

    # 打印建议
    for i, s in enumerate(suggestions, 1):
        log.info(f"\n{i}. [{s['priority']}优先级] {s['issue']}")
        log.info(f"   建议: {s['suggestion']}")
        log.info(f"   预期效果: {s['expected_impact']}")

    return suggestions


def create_optimization_plan():
    """创建优化计划"""
    log.info("\n" + "=" * 80)
    log.info("负样本优化实施计划")
    log.info("=" * 80)

    plan = """
负样本策略优化 v3 实施计划
========================

【当前状态】
- 正样本: ~3,190 个
- 负样本: ~7,636 个
- 硬负样本: ~130 个
- 比例: 1:2.4

【优化目标】
1. 市值分层采样 - 确保负样本市值分布与正样本一致
2. 增加硬负样本 - 目标占比 15-20%
3. 新增伪突破样本 - 突破后回落的样本

【实施步骤】
1. 修改 NegativeSampleScreenerV2
   - 添加市值分层采样逻辑
   - 按正样本市值分位数进行分层

2. 修改 HardNegativeSampleScreener
   - 添加伪突破类型
   - 条件: 突破20日高点后5日内回落>5%

3. 重新生成负样本
   - 预计耗时: 2-4小时
   - 需要网络访问 Tushare API

【预期效果】
- 减少市值偏差导致的模型偏见
- 提高对假突破的识别能力
- AUC 预期提升 0.5-1%

【注意事项】
- 重新生成负样本后需要重新计算所有特征
- 建议在 v2.6.0 版本中实施
"""

    log.info(plan)

    # 保存计划
    plan_file = PROJECT_ROOT / "docs" / "negative_sample_optimization_v3.md"
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_file, "w", encoding="utf-8") as f:
        f.write(plan)
    log.info(f"\n计划已保存到: {plan_file}")


def main():
    log.info("=" * 80)
    log.info("负样本策略优化分析 v3")
    log.info("=" * 80)

    # 分析当前分布
    stats = analyze_sample_distribution()

    # 提供优化建议
    suggest_optimizations(stats)

    # 创建优化计划
    create_optimization_plan()

    log.success("\n✓ 负样本策略分析完成！")
    log.info("注意: 实际的负样本重新生成需要较长时间，建议作为 v2.6.0 的任务")


if __name__ == "__main__":
    main()
