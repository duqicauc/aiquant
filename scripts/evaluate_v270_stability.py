#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估v2.7.0模型在1月5日到1月15日期间的预测稳定性

分析指标：
1. Top50股票的平均概率变化
2. Top50股票的重叠率（稳定性）
3. 预测概率的分布变化
4. 模型评分的一致性
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log


def load_predictions(date_str):
    """加载指定日期的预测结果"""
    top50_file = PROJECT_ROOT / "data" / "prediction" / "results" / f"v270_ensemble_top50_{date_str}.csv"
    all_file = PROJECT_ROOT / "data" / "prediction" / "results" / f"v270_ensemble_all_{date_str}.csv"

    top50 = None
    all_results = None

    if top50_file.exists():
        top50 = pd.read_csv(top50_file)
        log.info(f"✓ 加载Top50: {len(top50)}只股票")
    else:
        log.warning(f"✗ Top50文件不存在: {top50_file.name}")

    if all_file.exists():
        all_results = pd.read_csv(all_file)
        log.info(f"✓ 加载全市场: {len(all_results)}只股票")
    else:
        log.warning(f"✗ 全市场文件不存在: {all_file.name}")

    return top50, all_results


def calculate_overlap(df1, df2, top_n=50):
    """计算两个预测结果的重叠率"""
    if df1 is None or df2 is None:
        return 0.0, []

    stocks1 = set(df1.head(top_n)["ts_code"].tolist())
    stocks2 = set(df2.head(top_n)["ts_code"].tolist())

    overlap = stocks1 & stocks2
    overlap_rate = len(overlap) / top_n if top_n > 0 else 0.0

    return overlap_rate, list(overlap)


def analyze_stability(dates):
    """分析预测稳定性"""
    log.info("=" * 80)
    log.info("v2.7.0模型预测稳定性评估")
    log.info("=" * 80)
    log.info(f"评估日期范围: {dates[0]} 至 {dates[-1]}")
    log.info(f"共 {len(dates)} 个交易日\n")

    # 加载所有日期的预测结果
    predictions = {}
    for date_str in dates:
        top50, all_results = load_predictions(date_str)
        if top50 is not None:
            predictions[date_str] = {"top50": top50, "all": all_results}

    if len(predictions) < 2:
        log.error(f"可用预测结果不足（需要至少2个），实际只有 {len(predictions)} 个")
        return

    log.info(f"\n成功加载 {len(predictions)} 个日期的预测结果\n")

    # 1. 分析Top50的平均概率变化
    log.info("=" * 80)
    log.info("1. Top50平均概率变化")
    log.info("=" * 80)

    avg_probs = []
    for date_str in sorted(predictions.keys()):
        top50 = predictions[date_str]["top50"]
        avg_prob = top50["probability"].mean()
        max_prob = top50["probability"].max()
        min_prob = top50["probability"].min()
        avg_probs.append({"date": date_str, "avg_prob": avg_prob, "max_prob": max_prob, "min_prob": min_prob})
        log.info(f"{date_str}: 平均={avg_prob:.4f}, 最高={max_prob:.4f}, 最低={min_prob:.4f}")

    df_probs = pd.DataFrame(avg_probs)
    prob_std = df_probs["avg_prob"].std()
    prob_range = df_probs["avg_prob"].max() - df_probs["avg_prob"].min()

    log.info("\n平均概率统计:")
    log.info(f"  均值: {df_probs['avg_prob'].mean():.4f}")
    log.info(f"  标准差: {prob_std:.4f}")
    log.info(f"  变异系数: {prob_std/df_probs['avg_prob'].mean():.4f}")
    log.info(f"  极差: {prob_range:.4f}")

    # 2. 分析Top50股票重叠率（稳定性）
    log.info("\n" + "=" * 80)
    log.info("2. Top50股票重叠率（稳定性）")
    log.info("=" * 80)

    sorted_dates = sorted(predictions.keys())
    overlaps = []

    for i in range(len(sorted_dates) - 1):
        date1 = sorted_dates[i]
        date2 = sorted_dates[i + 1]

        top50_1 = predictions[date1]["top50"]
        top50_2 = predictions[date2]["top50"]

        overlap_rate, overlap_stocks = calculate_overlap(top50_1, top50_2, top_n=50)
        overlaps.append(
            {"date1": date1, "date2": date2, "overlap_rate": overlap_rate, "overlap_count": len(overlap_stocks)}
        )

        log.info(f"{date1} → {date2}: 重叠率={overlap_rate:.2%} ({len(overlap_stocks)}/50)")

    df_overlaps = pd.DataFrame(overlaps)
    avg_overlap = df_overlaps["overlap_rate"].mean()
    min_overlap = df_overlaps["overlap_rate"].min()
    max_overlap = df_overlaps["overlap_rate"].max()

    log.info("\n重叠率统计:")
    log.info(f"  平均重叠率: {avg_overlap:.2%}")
    log.info(f"  最低重叠率: {min_overlap:.2%}")
    log.info(f"  最高重叠率: {max_overlap:.2%}")

    # 3. 分析Top10股票稳定性
    log.info("\n" + "=" * 80)
    log.info("3. Top10股票稳定性")
    log.info("=" * 80)

    top10_overlaps = []
    for i in range(len(sorted_dates) - 1):
        date1 = sorted_dates[i]
        date2 = sorted_dates[i + 1]

        top50_1 = predictions[date1]["top50"]
        top50_2 = predictions[date2]["top50"]

        overlap_rate, overlap_stocks = calculate_overlap(top50_1, top50_2, top_n=10)
        top10_overlaps.append(overlap_rate)

        log.info(f"{date1} → {date2}: Top10重叠率={overlap_rate:.2%} ({len(overlap_stocks)}/10)")

    avg_top10_overlap = np.mean(top10_overlaps) if top10_overlaps else 0.0
    log.info(f"\nTop10平均重叠率: {avg_top10_overlap:.2%}")

    # 4. 分析全市场概率分布变化
    log.info("\n" + "=" * 80)
    log.info("4. 全市场概率分布变化")
    log.info("=" * 80)

    prob_stats = []
    for date_str in sorted(predictions.keys()):
        all_results = predictions[date_str]["all"]
        if all_results is not None:
            stats = {
                "date": date_str,
                "mean": all_results["probability"].mean(),
                "std": all_results["probability"].std(),
                "median": all_results["probability"].median(),
                "q75": all_results["probability"].quantile(0.75),
                "q25": all_results["probability"].quantile(0.25),
                "max": all_results["probability"].max(),
                "min": all_results["probability"].min(),
            }
            prob_stats.append(stats)
            log.info(f"{date_str}: 均值={stats['mean']:.4f}, 中位数={stats['median']:.4f}, 标准差={stats['std']:.4f}")

    if prob_stats:
        df_stats = pd.DataFrame(prob_stats)
        log.info("\n全市场概率分布稳定性:")
        log.info(f"  均值标准差: {df_stats['mean'].std():.4f}")
        log.info(f"  中位数标准差: {df_stats['median'].std():.4f}")

    # 5. 找出持续在Top50的股票
    log.info("\n" + "=" * 80)
    log.info("5. 持续在Top50的股票（高稳定性）")
    log.info("=" * 80)

    # 统计每只股票出现在Top50的次数
    stock_counts = defaultdict(int)
    stock_details = {}

    for date_str, pred in predictions.items():
        top50 = pred["top50"]
        for _, row in top50.iterrows():
            ts_code = row["ts_code"]
            stock_counts[ts_code] += 1
            if ts_code not in stock_details:
                stock_details[ts_code] = {"name": row["name"], "dates": [], "probs": []}
            stock_details[ts_code]["dates"].append(date_str)
            stock_details[ts_code]["probs"].append(row["probability"])

    # 找出出现次数最多的股票
    sorted_stocks = sorted(stock_counts.items(), key=lambda x: x[1], reverse=True)

    log.info("\n出现次数最多的Top10股票:")
    for i, (ts_code, count) in enumerate(sorted_stocks[:10], 1):
        details = stock_details[ts_code]
        avg_prob = np.mean(details["probs"])
        log.info(f"  {i}. {details['name']} ({ts_code})")
        log.info(f"     出现次数: {count}/{len(predictions)} ({count/len(predictions):.1%})")
        log.info(f"     平均概率: {avg_prob:.4f}")

    # 6. 总结评估结果
    log.info("\n" + "=" * 80)
    log.info("6. 稳定性评估总结")
    log.info("=" * 80)

    log.info("\n📊 关键指标:")
    log.info(f"  • Top50平均概率稳定性: 变异系数={prob_std/df_probs['avg_prob'].mean():.4f}")
    log.info(f"  • Top50股票重叠率: 平均={avg_overlap:.2%}")
    log.info(f"  • Top10股票重叠率: 平均={avg_top10_overlap:.2%}")

    # 稳定性评级
    if avg_overlap >= 0.70 and prob_std / df_probs["avg_prob"].mean() < 0.10:
        stability_rating = "⭐⭐⭐⭐⭐ 非常稳定"
    elif avg_overlap >= 0.60 and prob_std / df_probs["avg_prob"].mean() < 0.15:
        stability_rating = "⭐⭐⭐⭐ 较稳定"
    elif avg_overlap >= 0.50 and prob_std / df_probs["avg_prob"].mean() < 0.20:
        stability_rating = "⭐⭐⭐ 中等稳定"
    elif avg_overlap >= 0.40:
        stability_rating = "⭐⭐ 较不稳定"
    else:
        stability_rating = "⭐ 不稳定"

    log.info(f"\n🎯 稳定性评级: {stability_rating}")

    # 保存评估结果
    output_file = (
        PROJECT_ROOT / "data" / "prediction" / "evaluation" / f"v270_stability_evaluation_{dates[0]}_to_{dates[-1]}.md"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# v2.7.0模型预测稳定性评估报告\n\n")
        f.write(f"**评估日期**: {dates[0]} 至 {dates[-1]}\n")
        f.write(f"**评估日期数**: {len(predictions)} 个交易日\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 关键指标\n\n")
        f.write(f"- Top50平均概率: {df_probs['avg_prob'].mean():.4f} (标准差: {prob_std:.4f})\n")
        f.write(f"- Top50重叠率: {avg_overlap:.2%}\n")
        f.write(f"- Top10重叠率: {avg_top10_overlap:.2%}\n")
        f.write(f"- 稳定性评级: {stability_rating}\n")

    log.info(f"\n✓ 评估结果已保存: {output_file}")


def main():
    import sys

    # 默认评估1月5日到1月15日
    if len(sys.argv) > 1:
        dates = sys.argv[1:]
    else:
        # 生成1月5日到1月15日的日期列表（排除周末）
        dates = []
        start_date = datetime(2026, 1, 5)
        end_date = datetime(2026, 1, 15)
        current = start_date
        while current <= end_date:
            # 简单判断：周一到周五（0-4）
            if current.weekday() < 5:
                dates.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)

    analyze_stability(dates)


if __name__ == "__main__":
    main()
