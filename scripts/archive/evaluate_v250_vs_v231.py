#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估v2.5.0 vs v2.3.1/v2.3.2模型预测效果

对比2025.12.31的预测结果，用2026.01.14的实际收盘数据评估
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.utils.logger import log
from src.data.data_manager import DataManager


def get_stock_return(dm, ts_code, start_date, end_date):
    """获取股票在指定日期范围内的收益"""
    try:
        # 获取开始日期附近的数据
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        start_range_start = (start_dt - timedelta(days=5)).strftime("%Y%m%d")
        start_range_end = (start_dt + timedelta(days=5)).strftime("%Y%m%d")

        df_start = dm.get_daily_data(ts_code, start_range_start, start_range_end)
        if df_start is None or len(df_start) == 0:
            return None

        # 找最接近开始日期的数据
        df_start["date_diff"] = abs(pd.to_datetime(df_start["trade_date"]) - start_dt)
        closest_start = df_start.loc[df_start["date_diff"].idxmin()]
        start_close = closest_start["close"]
        start_date_actual = str(closest_start["trade_date"])[:10]

        # 获取结束日期附近的数据
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        end_range_start = (end_dt - timedelta(days=5)).strftime("%Y%m%d")
        end_range_end = (end_dt + timedelta(days=5)).strftime("%Y%m%d")

        df_end = dm.get_daily_data(ts_code, end_range_start, end_range_end)
        if df_end is None or len(df_end) == 0:
            return None

        # 找最接近结束日期的数据
        df_end["date_diff"] = abs(pd.to_datetime(df_end["trade_date"]) - end_dt)
        closest_end = df_end.loc[df_end["date_diff"].idxmin()]
        end_close = closest_end["close"]
        end_date_actual = str(closest_end["trade_date"])[:10]

        # 计算收益
        return_pct = (end_close - start_close) / start_close * 100

        return {
            "return_pct": return_pct,
            "start_price": start_close,
            "end_price": end_close,
            "start_date_actual": start_date_actual,
            "end_date_actual": end_date_actual,
        }
    except Exception as e:
        log.warning(f"  获取{ts_code}收益失败: {e}")
        return None


def evaluate_predictions(dm, predictions_file, predict_date, eval_date, version):
    """评估预测结果"""
    log.info(f"\n{'='*80}")
    log.info(f"评估 {version}")
    log.info(f"{'='*80}")

    if not Path(predictions_file).exists():
        log.warning(f"文件不存在: {predictions_file}")
        return None

    df = pd.read_csv(predictions_file)
    log.info(f"预测股票数: {len(df)}")

    # 计算每只股票从预测日到评估日的收益
    results = []
    for _, row in df.iterrows():
        ts_code = row["ts_code"]
        name = row["name"]

        ret_info = get_stock_return(dm, ts_code, predict_date, eval_date)
        if ret_info is not None:
            results.append(
                {
                    "ts_code": ts_code,
                    "name": name,
                    "predict_price": row.get("close", 0),
                    "predict_pct_chg": row.get("pct_chg", 0),
                    "calibrated_probability": row.get("calibrated_probability", 0),
                    "final_score": row.get("final_score", 0),
                    "return_34d": row.get("return_34d", 0),
                    "actual_return": ret_info["return_pct"],
                    "start_price": ret_info["start_price"],
                    "end_price": ret_info["end_price"],
                    "start_date_actual": ret_info["start_date_actual"],
                    "end_date_actual": ret_info["end_date_actual"],
                }
            )

    if not results:
        log.error("无法获取收益数据")
        return None

    df_eval = pd.DataFrame(results)

    # 统计
    avg_return = df_eval["actual_return"].mean()
    win_rate = (df_eval["actual_return"] > 0).mean() * 100
    avg_pre_t1 = df_eval["return_34d"].mean()
    avg_pct_chg = df_eval["predict_pct_chg"].mean()
    chase_high_count = (df_eval["predict_pct_chg"] > 9).sum()
    max_return = df_eval["actual_return"].max()
    min_return = df_eval["actual_return"].min()
    median_return = df_eval["actual_return"].median()

    log.info(f"\n【{version} Top10 效果】")
    log.info(f"  预测日平均涨幅: {avg_pct_chg:.2f}%")
    log.info(f"  追高数量(>9%): {chase_high_count}/10")
    log.info(f"  T1前平均涨幅: {avg_pre_t1:.1f}%")
    log.info(f"  实际平均收益: {avg_return:.2f}%")
    log.info(f"  实际中位数收益: {median_return:.2f}%")
    log.info(f"  最大收益: {max_return:.2f}%")
    log.info(f"  最小收益: {min_return:.2f}%")
    log.info(f"  胜率: {win_rate:.0f}%")

    log.info("\n  详细：")
    log.info(
        f"  {'排名':<4} {'代码':<12} {'名称':<10} {'预测日涨幅':<12} {'预测概率':<10} {'实际收益':<12} {'状态':<8}"
    )
    log.info(f"  {'-'*80}")

    df_eval_sorted = df_eval.sort_values("actual_return", ascending=False)
    for i, (_, row) in enumerate(df_eval_sorted.iterrows(), 1):
        status = "✅盈利" if row["actual_return"] > 0 else "❌亏损"
        log.info(
            f"  {i:<4} {row['ts_code']:<12} {row['name']:<10} "
            f"{row['predict_pct_chg']:>+10.2f}%  {row['calibrated_probability']:>8.4f}  "
            f"{row['actual_return']:>+10.2f}%  {status:<8}"
        )

    return {
        "version": version,
        "avg_pre_t1": avg_pre_t1,
        "avg_pct_chg": avg_pct_chg,
        "chase_high_count": chase_high_count,
        "avg_return": avg_return,
        "median_return": median_return,
        "max_return": max_return,
        "min_return": min_return,
        "win_rate": win_rate,
        "details": df_eval,
    }


def compare_models(results_v250, results_v231):
    """对比两个模型的结果"""
    log.info("\n" + "=" * 80)
    log.info("模型对比总结")
    log.info("=" * 80)

    if results_v250 is None or results_v231 is None:
        log.warning("缺少对比数据，无法进行完整对比")
        return

    log.info(f"\n{'指标':<20} {'v2.5.0':<15} {'v2.3.1':<15} {'差异':<15} {'优势':<10}")
    log.info("-" * 80)

    metrics = [
        ("平均收益", "avg_return", "%", True),
        ("中位数收益", "median_return", "%", True),
        ("最大收益", "max_return", "%", True),
        ("最小收益", "min_return", "%", True),
        ("胜率", "win_rate", "%", True),
        ("预测日平均涨幅", "avg_pct_chg", "%", False),
        ("追高数量", "chase_high_count", "只", False),
    ]

    for metric_name, metric_key, unit, higher_better in metrics:
        v250_val = results_v250.get(metric_key, 0)
        v231_val = results_v231.get(metric_key, 0)

        if v250_val is None or v231_val is None:
            continue

        diff = v250_val - v231_val
        if higher_better:
            advantage = "v2.5.0" if diff > 0 else "v2.3.1" if diff < 0 else "平"
        else:
            advantage = "v2.5.0" if diff < 0 else "v2.3.1" if diff > 0 else "平"

        log.info(
            f"{metric_name:<20} {v250_val:>12.2f}{unit:<3} {v231_val:>12.2f}{unit:<3} "
            f"{diff:>+12.2f}{unit:<3} {advantage:<10}"
        )

    # 综合评分
    log.info("\n" + "-" * 80)
    log.info("综合评估:")

    v250_score = 0
    v231_score = 0

    # 收益对比
    if results_v250["avg_return"] > results_v231["avg_return"]:
        v250_score += 2
    elif results_v250["avg_return"] < results_v231["avg_return"]:
        v231_score += 2

    # 胜率对比
    if results_v250["win_rate"] > results_v231["win_rate"]:
        v250_score += 1
    elif results_v250["win_rate"] < results_v231["win_rate"]:
        v231_score += 1

    # 追高数量对比（越少越好）
    if results_v250["chase_high_count"] < results_v231["chase_high_count"]:
        v250_score += 1
    elif results_v250["chase_high_count"] > results_v231["chase_high_count"]:
        v231_score += 1

    log.info(f"  v2.5.0得分: {v250_score}/4")
    log.info(f"  v2.3.1得分: {v231_score}/4")

    if v250_score > v231_score:
        log.success("  ✅ v2.5.0 表现更优")
    elif v250_score < v231_score:
        log.warning("  ⚠️  v2.3.1 表现更优")
    else:
        log.info("  ➖ 两个模型表现相当")


def main():
    predict_date = "20251231"
    eval_date = "20260114"  # 今天收盘后

    log.info("=" * 80)
    log.info("v2.5.0 vs v2.3.1 模型效果对比")
    log.info("=" * 80)
    log.info(f"预测日期: {predict_date}")
    log.info(f"评估日期: {eval_date}")
    log.info("=" * 80)

    # 初始化数据管理器
    try:
        dm = DataManager()
    except Exception as e:
        log.error(f"初始化DataManager失败: {e}")
        log.info("提示: 如果预测结果文件已存在，可以直接读取进行评估")
        return

    # 评估v2.5.0
    v250_file = PROJECT_ROOT / "data" / "prediction" / "results" / f"v2.5.0_top10_{predict_date}.csv"
    results_v250 = evaluate_predictions(dm, str(v250_file), predict_date, eval_date, "v2.5.0")

    # 评估v2.3.1
    v231_file = PROJECT_ROOT / "data" / "prediction" / "results" / f"v2.3.1_top10_{predict_date}.csv"
    results_v231 = evaluate_predictions(dm, str(v231_file), predict_date, eval_date, "v2.3.1")

    # 如果v2.3.1不存在，尝试v2.3.2
    if results_v231 is None:
        v232_file = PROJECT_ROOT / "data" / "prediction" / "results" / f"v2.3.2_top10_{predict_date}.csv"
        results_v231 = evaluate_predictions(dm, str(v232_file), predict_date, eval_date, "v2.3.2")

    # 对比
    compare_models(results_v250, results_v231)

    # 保存对比报告
    if results_v250 is not None and results_v231 is not None:
        report_dir = PROJECT_ROOT / "data" / "prediction" / "evaluation"
        report_dir.mkdir(parents=True, exist_ok=True)

        report_file = report_dir / f"v250_vs_v231_comparison_{predict_date}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# v2.5.0 vs v2.3.1 模型对比报告\n\n")
            f.write(f"**预测日期**: {predict_date}\n")
            f.write(f"**评估日期**: {eval_date}\n\n")

            f.write("## v2.5.0 结果\n\n")
            f.write(f"- 平均收益: {results_v250['avg_return']:.2f}%\n")
            f.write(f"- 胜率: {results_v250['win_rate']:.0f}%\n")
            f.write(f"- 追高数量: {results_v250['chase_high_count']}/10\n\n")

            f.write("## v2.3.1 结果\n\n")
            f.write(f"- 平均收益: {results_v231['avg_return']:.2f}%\n")
            f.write(f"- 胜率: {results_v231['win_rate']:.0f}%\n")
            f.write(f"- 追高数量: {results_v231['chase_high_count']}/10\n\n")

        log.success(f"\n✓ 对比报告已保存: {report_file}")


if __name__ == "__main__":
    main()
