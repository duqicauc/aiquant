#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.3.1和v2.3.2完整评估脚本

评估多个预测日期的表现，计算整体统计数据
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log
from src.data.data_manager import DataManager


def get_eval_price(dm, ts_code, predict_date, days_after=10):
    """获取评估日的价格"""
    try:
        start = datetime.strptime(predict_date, "%Y%m%d")
        end = start + timedelta(days=days_after + 5)

        df = dm.get_daily_data(ts_code, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
        if df is None or len(df) == 0:
            return None, None

        df = df.sort_values("trade_date")
        # 获取预测日之后第N个交易日的价格
        predict_date_ts = pd.to_datetime(predict_date)
        df["trade_date_ts"] = pd.to_datetime(df["trade_date"])
        df_after = df[df["trade_date_ts"] > predict_date_ts]

        if len(df_after) >= days_after:
            eval_row = df_after.iloc[days_after - 1]
            return float(eval_row["close"]), eval_row["trade_date"]
        elif len(df_after) > 0:
            eval_row = df_after.iloc[-1]
            return float(eval_row["close"]), eval_row["trade_date"]
        return None, None
    except Exception:
        return None, None


def evaluate_single_prediction(dm, predict_file, version, days_after=10):
    """评估单个预测文件"""
    if not predict_file.exists():
        return None

    df = pd.read_csv(predict_file)
    predict_date = predict_file.stem.split("_")[-1]

    results = []
    for _, row in df.iterrows():
        ts_code = row["ts_code"]
        predict_price = row["close"]

        eval_price, eval_date = get_eval_price(dm, ts_code, predict_date, days_after)
        if eval_price is None:
            continue

        return_pct = (eval_price / predict_price - 1) * 100
        results.append(
            {
                "ts_code": ts_code,
                "name": row["name"],
                "predict_date": predict_date,
                "eval_date": eval_date,
                "predict_price": predict_price,
                "eval_price": eval_price,
                "return_pct": return_pct,
                "pct_chg": row.get("pct_chg", 0),  # 预测日涨幅
                "calibrated_probability": row.get("calibrated_probability", 0),
                "final_score": row.get("final_score", 0),
            }
        )

    if not results:
        return None

    df_eval = pd.DataFrame(results)

    # 计算统计
    stats = {
        "version": version,
        "predict_date": predict_date,
        "stock_count": len(df_eval),
        "avg_return": df_eval["return_pct"].mean(),
        "median_return": df_eval["return_pct"].median(),
        "max_return": df_eval["return_pct"].max(),
        "min_return": df_eval["return_pct"].min(),
        "std_return": df_eval["return_pct"].std(),
        "win_rate": (df_eval["return_pct"] > 0).sum() / len(df_eval) * 100,
        "high_return_count": (df_eval["return_pct"] > 20).sum(),
        "big_loss_count": (df_eval["return_pct"] < -10).sum(),
        "avg_pct_chg": df_eval["pct_chg"].mean(),  # 预测日平均涨幅
        "chase_high_count": (df_eval["pct_chg"] > 9).sum(),  # 追高数量
    }

    return stats, df_eval


def main():
    log.info("=" * 80)
    log.info("v2.3.1 & v2.3.2 完整评估")
    log.info("=" * 80)

    dm = DataManager()
    results_dir = PROJECT_ROOT / "data" / "prediction" / "results"

    # 定义评估配置
    eval_configs = [
        # v2.3.1
        ("v2.3.1", "v2.3.1_top10_20251212.csv", 10),
        ("v2.3.1", "v2.3.1_top10_20251231.csv", 10),
        ("v2.3.1", "v2.3.1_top10_20260105.csv", 5),
        ("v2.3.1", "v2.3.1_top10_20260106.csv", 5),
        ("v2.3.1", "v2.3.1_top10_20260107.csv", 5),
        # v2.3.2
        ("v2.3.2", "v2.3.2_top10_20260109.csv", 5),
        ("v2.3.2", "v2.3.2_top10_20260112.csv", 3),
        ("v2.3.2", "v2.3.2_top10_20260113.csv", 2),
    ]

    all_stats = []
    all_details = {}

    for version, filename, days_after in eval_configs:
        predict_file = results_dir / filename
        log.info(f"\n评估 {version} - {filename} (T+{days_after}日)...")

        result = evaluate_single_prediction(dm, predict_file, version, days_after)
        if result:
            stats, df_eval = result
            all_stats.append(stats)
            all_details[f"{version}_{filename}"] = df_eval
            log.success(f"  ✓ 平均收益: {stats['avg_return']:+.2f}%, 胜率: {stats['win_rate']:.0f}%")
        else:
            log.warning("  ⚠️ 无法评估")

    # 汇总统计
    df_stats = pd.DataFrame(all_stats)

    # 按版本分组
    log.info("\n" + "=" * 80)
    log.info("📊 评估结果汇总")
    log.info("=" * 80)

    for version in ["v2.3.1", "v2.3.2"]:
        df_v = df_stats[df_stats["version"] == version]
        if len(df_v) == 0:
            continue

        log.info(f"\n{'='*40}")
        log.info(f"🔹 {version} 汇总")
        log.info(f"{'='*40}")
        log.info(
            f"\n{'预测日期':<12} {'收益均值':>10} {'中位数':>10} {'最高':>10} {'最低':>10} {'胜率':>8} {'追高':>6}"
        )
        log.info("-" * 75)

        for _, row in df_v.iterrows():
            log.info(
                f"{row['predict_date']:<12} "
                f"{row['avg_return']:>+9.2f}% "
                f"{row['median_return']:>+9.2f}% "
                f"{row['max_return']:>+9.2f}% "
                f"{row['min_return']:>+9.2f}% "
                f"{row['win_rate']:>7.0f}% "
                f"{row['chase_high_count']:>5.0f}"
            )

        # 整体统计
        log.info("-" * 75)
        log.info(
            f"{'整体平均':<12} "
            f"{df_v['avg_return'].mean():>+9.2f}% "
            f"{df_v['median_return'].mean():>+9.2f}% "
            f"{df_v['max_return'].max():>+9.2f}% "
            f"{df_v['min_return'].min():>+9.2f}% "
            f"{df_v['win_rate'].mean():>7.0f}% "
            f"{df_v['chase_high_count'].mean():>5.1f}"
        )

    # 版本对比
    log.info("\n" + "=" * 80)
    log.info("📈 v2.3.1 vs v2.3.2 对比")
    log.info("=" * 80)

    v231_stats = df_stats[df_stats["version"] == "v2.3.1"]
    v232_stats = df_stats[df_stats["version"] == "v2.3.2"]

    if len(v231_stats) > 0 and len(v232_stats) > 0:
        log.info(f"\n{'指标':<20} {'v2.3.1':<15} {'v2.3.2':<15} {'差异':<15}")
        log.info("-" * 65)

        metrics = [
            ("平均收益率", "avg_return"),
            ("中位数收益", "median_return"),
            ("胜率", "win_rate"),
            ("平均追高数", "chase_high_count"),
            ("预测日平均涨幅", "avg_pct_chg"),
        ]

        for name, col in metrics:
            v231_val = v231_stats[col].mean()
            v232_val = v232_stats[col].mean()
            diff = v232_val - v231_val
            unit = "%" if col != "chase_high_count" else ""
            log.info(f"{name:<20} {v231_val:>+12.2f}{unit:<2} {v232_val:>+12.2f}{unit:<2} {diff:>+12.2f}{unit}")

    # 输出详细结果
    log.info("\n" + "=" * 80)
    log.info("📋 Top10 详细收益")
    log.info("=" * 80)

    for key, df_eval in all_details.items():
        version = key.split("_")[0]
        predict_date = key.split("_")[-1].replace(".csv", "")

        log.info(f"\n【{version} - {predict_date}】")
        log.info(f"{'代码':<12} {'名称':<10} {'预测价':>8} {'评估价':>8} {'收益率':>10} {'当日涨幅':>10}")
        log.info("-" * 70)

        for _, row in df_eval.iterrows():
            log.info(
                f"{row['ts_code']:<12} {row['name']:<10} "
                f"{row['predict_price']:>8.2f} {row['eval_price']:>8.2f} "
                f"{row['return_pct']:>+9.2f}% {row['pct_chg']:>+9.2f}%"
            )

    # 保存结果
    output_dir = PROJECT_ROOT / "data" / "prediction" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_stats.to_csv(output_dir / "v23x_evaluation_summary.csv", index=False, encoding="utf-8-sig")
    log.success(f"\n💾 评估汇总已保存: {output_dir / 'v23x_evaluation_summary.csv'}")

    return df_stats


if __name__ == "__main__":
    main()
