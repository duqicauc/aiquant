#!/usr/bin/env python3
"""
预测效果分析脚本
对比预测日期和实际日期的股票表现
"""
import pandas as pd
from pathlib import Path
import sys
import argparse

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager
from src.utils.logger import log


def analyze_prediction(pred_date: str, actual_date: str, top_n: int = 50):
    """
    分析预测效果

    Args:
        pred_date: 预测日期 (YYYYMMDD)
        actual_date: 实际日期 (YYYYMMDD)
        top_n: 分析Top N只股票
    """
    log.info("=" * 70)
    log.info(f"📊 预测效果分析：{pred_date}预测 vs {actual_date}实际表现")
    log.info("=" * 70)

    # 1. 加载预测结果
    pred_file = PROJECT_ROOT / "data" / "prediction" / "results" / f"stock_scores_advanced_{pred_date}.csv"
    if not pred_file.exists():
        log.error(f"预测文件不存在: {pred_file}")
        return

    df_pred = pd.read_csv(pred_file)
    log.info(f"✓ 加载预测数据: {len(df_pred)} 只股票")

    # 2. 获取实际价格（只分析Top N，加快速度）
    log.info(f"\n正在获取实际价格数据（分析Top {top_n}只）...")
    dm = DataManager()

    # 只分析Top N
    df_pred_top = df_pred.nlargest(top_n, "牛股概率").copy()

    results = []
    for idx, (_, row) in enumerate(df_pred_top.iterrows()):
        ts_code = row["股票代码"]
        pred_price = row["最新价格"]
        prob = row["牛股概率"]

        try:
            # 获取实际日期的价格
            df_actual = dm.get_daily_data(ts_code, actual_date, actual_date)
            if df_actual is not None and len(df_actual) > 0:
                actual_price = df_actual["close"].iloc[-1]
                return_pct = (actual_price - pred_price) / pred_price * 100

                results.append(
                    {
                        "股票代码": ts_code,
                        "股票名称": row["股票名称"],
                        "预测概率": prob,
                        "预测价格": pred_price,
                        "实际价格": actual_price,
                        "收益率%": return_pct,
                        "是否上涨": 1 if return_pct > 0 else 0,
                        "是否大涨": 1 if return_pct > 10 else 0,
                    }
                )
        except Exception as e:
            log.warning(f"获取 {ts_code} 数据失败: {e}")
            continue

        if (idx + 1) % 10 == 0:
            log.info(f"  进度: {idx+1}/{len(df_pred_top)}")

    if not results:
        log.error("未获取到任何实际数据")
        return

    df_results = pd.DataFrame(results)
    log.info(f"✓ 成功获取: {len(df_results)} 只股票的实际数据")

    # 3. 分析预测效果
    log.info("\n" + "=" * 70)
    log.info("【预测效果分析】")
    log.info("=" * 70)

    # 整体统计
    log.info(f"\n📈 Top {top_n} 整体表现：")
    log.info(f"  平均收益率: {df_results['收益率%'].mean():.2f}%")
    log.info(f"  中位数收益率: {df_results['收益率%'].median():.2f}%")
    log.info(f"  上涨率: {df_results['是否上涨'].mean()*100:.1f}%")
    log.info(f"  大涨率(>10%): {df_results['是否大涨'].mean()*100:.1f}%")
    log.info(f"  最大涨幅: {df_results['收益率%'].max():.2f}%")
    log.info(f"  最大跌幅: {df_results['收益率%'].min():.2f}%")
    log.info(f"  正收益股票数: {(df_results['收益率%'] > 0).sum()}")
    log.info(f"  负收益股票数: {(df_results['收益率%'] < 0).sum()}")

    # Top 10分析
    if len(df_results) >= 10:
        log.info("\n🥇 Top 10 表现：")
        top10 = df_results.nlargest(10, "预测概率")
        log.info(f"  平均收益率: {top10['收益率%'].mean():.2f}%")
        log.info(f"  上涨率: {top10['是否上涨'].mean()*100:.1f}%")
        log.info(f"  大涨率(>10%): {top10['是否大涨'].mean()*100:.1f}%")

    # 按概率区间分析
    df_results["概率区间"] = pd.cut(
        df_results["预测概率"], bins=[0, 0.85, 0.90, 0.95, 1.0], labels=["85-90%", "90-95%", "95-98%", "≥98%"]
    )

    log.info("\n📊 按预测概率分组表现：")
    group_stats = (
        df_results.groupby("概率区间")
        .agg({"收益率%": ["count", "mean", "median"], "是否上涨": "mean", "是否大涨": "mean"})
        .round(2)
    )

    for prob_range, stats in group_stats.iterrows():
        count = int(stats[("收益率%", "count")])
        mean_return = stats[("收益率%", "mean")]
        up_rate = stats[("是否上涨", "mean")] * 100
        log.info(f"  {prob_range}: {count}只, 平均收益{mean_return:.2f}%, 上涨率{up_rate:.1f}%")

    # 最佳/最差表现
    log.info("\n🏆 最佳表现 Top 5：")
    best = df_results.nlargest(5, "收益率%")
    for _, row in best.iterrows():
        log.info(f"  {row['股票代码']} {row['股票名称']}: {row['收益率%']:.2f}% (预测概率: {row['预测概率']:.2%})")

    log.info("\n📉 最差表现 Top 5：")
    worst = df_results.nsmallest(5, "收益率%")
    for _, row in worst.iterrows():
        log.info(f"  {row['股票代码']} {row['股票名称']}: {row['收益率%']:.2f}% (预测概率: {row['预测概率']:.2%})")

    # 保存结果
    output_file = (
        PROJECT_ROOT / "data" / "prediction" / "results" / f"prediction_analysis_{pred_date}_to_{actual_date}.csv"
    )
    df_results.to_csv(output_file, index=False, encoding="utf-8-sig")
    log.success(f"\n✅ 详细分析结果已保存: {output_file}")

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析预测效果")
    parser.add_argument("--pred-date", type=str, required=True, help="预测日期 (YYYYMMDD)")
    parser.add_argument("--actual-date", type=str, required=True, help="实际日期 (YYYYMMDD)")
    parser.add_argument("--top-n", type=int, default=50, help="分析Top N只股票 (默认50)")

    args = parser.parse_args()

    analyze_prediction(args.pred_date, args.actual_date, args.top_n)
