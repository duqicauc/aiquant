#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析v2.5.1模型优化空间
"""
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log


def analyze_feature_importance():
    """分析特征重要性分布"""
    log.info("=" * 80)
    log.info("v2.5.1 特征重要性分析")
    log.info("=" * 80)

    df = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "models"
        / "breakout_launch_scorer"
        / "versions"
        / "v2.5.1"
        / "feature_importance_v251.csv"
    )

    # 1. Top特征分析
    log.info("\n【问题1】特征集中度分析")
    log.info("-" * 80)
    top1_pct = df.iloc[0]["percentage"]
    top7_pct = df.head(7)["percentage"].sum()
    top20_pct = df.head(20)["percentage"].sum()

    log.info(f"Top 1特征占比: {top1_pct:.2f}% (days_to_t1)")
    log.info(f"Top 7特征占比: {top7_pct:.2f}%")
    log.info(f"Top 20特征占比: {top20_pct:.2f}%")

    if top1_pct > 50:
        log.warning("⚠️  Top 1特征占比超过50%，存在过度依赖风险！")

    # 2. days_to_t1特征分析
    log.info("\n【问题2】days_to_t1特征分析")
    log.info("-" * 80)
    days_to_t1_row = df[df["feature"] == "days_to_t1"].iloc[0]
    log.info(f"重要性: {days_to_t1_row['importance']:.2f}")
    log.info(f"占比: {days_to_t1_row['percentage']:.2f}%")
    log.warning("⚠️  该特征在预测时可能不可用（需要知道T1日期）")
    log.warning("⚠️  如果预测时无法计算，这是数据泄露风险！")

    # 3. 按类别分析
    log.info("\n【问题3】特征类别分布")
    log.info("-" * 80)
    cat_stats = (
        df.groupby("category").agg({"percentage": "sum", "feature": "count"}).sort_values("percentage", ascending=False)
    )

    for cat, row in cat_stats.iterrows():
        log.info(f"{cat:<20} {row['percentage']:>6.2f}% ({row['feature']:>2}个特征)")

    # 4. 核心业务特征分析
    log.info("\n【问题4】核心业务特征重要性")
    log.info("-" * 80)
    business_features = {
        "突破特征": df[df["category"] == "突破特征"]["percentage"].sum(),
        "量价特征": df[df["category"] == "量价特征"]["percentage"].sum(),
        "动量特征": df[df["category"] == "动量特征"]["percentage"].sum(),
        "风险特征": df[df["category"] == "风险特征"]["percentage"].sum(),
    }

    for feat_type, pct in business_features.items():
        log.info(f"{feat_type:<20} {pct:>6.2f}%")
        if pct < 3:
            log.warning(f"  ⚠️  {feat_type}重要性偏低，可能未充分利用")

    # 5. 未使用特征
    log.info("\n【问题5】未使用的特征")
    log.info("-" * 80)
    unused = df[df["importance"] == 0]
    if len(unused) > 0:
        log.info(f"未使用特征数: {len(unused)} / {len(df)} ({len(unused)/len(df)*100:.1f}%)")
        log.info("未使用特征类别分布:")
        unused_cats = unused.groupby("category")["feature"].count()
        for cat, count in unused_cats.items():
            log.info(f"  {cat:<20} {count:>2}个")
    else:
        log.info("所有特征都被使用")

    return df


def suggest_optimizations(df):
    """提出优化建议"""
    log.info("\n" + "=" * 80)
    log.info("模型优化建议")
    log.info("=" * 80)

    suggestions = []

    # 1. days_to_t1特征问题
    days_to_t1_pct = df[df["feature"] == "days_to_t1"].iloc[0]["percentage"]
    if days_to_t1_pct > 50:
        suggestions.append(
            {
                "priority": "P0",
                "issue": "days_to_t1特征占比过高（52.97%）",
                "risk": "数据泄露风险：预测时可能无法计算此特征",
                "solution": [
                    "1. 检查预测脚本中是否使用days_to_t1",
                    "2. 如果预测时不可用，应从训练中排除",
                    "3. 如果可用，考虑降低其权重或分桶处理",
                ],
            }
        )

    # 2. 特征集中度
    top7_pct = df.head(7)["percentage"].sum()
    if top7_pct > 65:
        suggestions.append(
            {
                "priority": "P1",
                "issue": f"Top 7特征占比过高（{top7_pct:.1f}%）",
                "risk": "模型过于简单，可能欠拟合",
                "solution": [
                    "1. 增加正则化，强制使用更多特征",
                    "2. 调整colsample_bytree参数（当前0.8）",
                    "3. 使用特征选择，强制包含突破/量价特征",
                ],
            }
        )

    # 3. 核心特征重要性低
    breakout_pct = df[df["category"] == "突破特征"]["percentage"].sum()
    if breakout_pct < 2:
        suggestions.append(
            {
                "priority": "P1",
                "issue": f"突破特征重要性偏低（{breakout_pct:.2f}%）",
                "risk": "核心业务特征未充分利用",
                "solution": [
                    "1. 特征工程：增强突破特征的区分度",
                    "2. 样本优化：增加硬负样本（伪突破）",
                    "3. 模型参数：调整min_child_weight降低门槛",
                ],
            }
        )

    # 4. 未使用特征
    unused_count = len(df[df["importance"] == 0])
    if unused_count > 10:
        suggestions.append(
            {
                "priority": "P2",
                "issue": f"未使用特征过多（{unused_count}个）",
                "risk": "特征工程投入浪费",
                "solution": [
                    "1. 特征选择：移除冗余特征",
                    "2. 特征组合：将相关特征组合",
                    "3. 特征重要性筛选：只保留Top N特征",
                ],
            }
        )

    # 打印建议
    for i, sug in enumerate(suggestions, 1):
        log.info(f"\n【建议{i}】优先级: {sug['priority']}")
        log.info(f"问题: {sug['issue']}")
        log.info(f"风险: {sug['risk']}")
        log.info("解决方案:")
        for sol in sug["solution"]:
            log.info(f"  {sol}")

    return suggestions


def main():
    df = analyze_feature_importance()
    suggestions = suggest_optimizations(df)

    log.info("\n" + "=" * 80)
    log.info("总结")
    log.info("=" * 80)
    log.info(f"总特征数: {len(df)}")
    log.info(f"使用特征数: {len(df[df['importance'] > 0])}")
    log.info(f"优化建议数: {len(suggestions)}")

    # 保存分析结果
    output_file = (
        PROJECT_ROOT / "data" / "models" / "breakout_launch_scorer" / "versions" / "v2.5.1" / "optimization_analysis.md"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# v2.5.1 模型优化分析\n\n")
        f.write("## 特征重要性分布\n\n")
        f.write(df.head(30).to_string(index=False))
        f.write("\n\n## 优化建议\n\n")
        for i, sug in enumerate(suggestions, 1):
            f.write(f"### 建议{i}: {sug['issue']}\n\n")
            f.write(f"**优先级**: {sug['priority']}\n\n")
            f.write(f"**风险**: {sug['risk']}\n\n")
            f.write("**解决方案**:\n")
            for sol in sug["solution"]:
                f.write(f"- {sol}\n")
            f.write("\n")

    log.success(f"\n✓ 分析完成，结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
