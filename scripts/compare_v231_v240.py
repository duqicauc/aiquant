#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比较v2.3.1和v2.4.0模型效果

使用2025-12-12的预测结果，用2026-01-05的数据评估实际收益
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


def get_stock_return(dm, ts_code, start_date, end_date):
    """获取股票在指定日期范围内的收益"""
    try:
        df = dm.get_daily_data(ts_code, start_date, end_date)
        if df is None or len(df) < 2:
            return None

        df = df.sort_values("trade_date")
        start_close = df.iloc[0]["close"]
        end_close = df.iloc[-1]["close"]

        return (end_close - start_close) / start_close * 100
    except:
        return None


def evaluate_predictions(dm, predictions_file, predict_date, eval_date, version):
    """评估预测结果"""
    log.info(f"\n{'='*60}")
    log.info(f"评估 {version}")
    log.info(f"{'='*60}")

    if not Path(predictions_file).exists():
        log.error(f"文件不存在: {predictions_file}")
        return None

    df = pd.read_csv(predictions_file)
    log.info(f"预测股票数: {len(df)}")

    # 计算每只股票从预测日到评估日的收益
    results = []
    for _, row in df.iterrows():
        ts_code = row["ts_code"]
        name = row["name"]

        ret = get_stock_return(dm, ts_code, predict_date, eval_date)
        if ret is not None:
            results.append(
                {
                    "ts_code": ts_code,
                    "name": name,
                    "return_34d": row.get("return_34d", 0),  # T1前涨幅
                    "actual_return": ret,  # 实际收益
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

    log.info(f"\n【{version} Top10 效果】")
    log.info(f"  T1前平均涨幅: {avg_pre_t1:.1f}%")
    log.info(f"  实际平均收益: {avg_return:.1f}%")
    log.info(f"  胜率: {win_rate:.0f}%")

    log.info("\n  详细：")
    log.info(f"  {'代码':<12} {'名称':<10} {'T1前涨幅':<10} {'实际收益':<10}")
    log.info(f"  {'-'*45}")

    for _, row in df_eval.iterrows():
        pre_ret = row["return_34d"]
        actual_ret = row["actual_return"]
        log.info(f"  {row['ts_code']:<12} {row['name']:<10} {pre_ret:>+8.1f}%  {actual_ret:>+8.1f}%")

    return {
        "version": version,
        "avg_pre_t1": avg_pre_t1,
        "avg_return": avg_return,
        "win_rate": win_rate,
        "details": df_eval,
    }


def main():
    predict_date = "20251212"
    eval_date = "20260103"  # 用最近的交易日

    log.info("=" * 80)
    log.info("v2.3.1 vs v2.4.0 模型效果对比")
    log.info("=" * 80)
    log.info(f"预测日期: {predict_date}")
    log.info(f"评估日期: {eval_date}")

    dm = DataManager()

    # v2.3.1 Top10
    v231_file = PROJECT_ROOT / "data" / "prediction" / "results" / f"v2.3.1_top10_{predict_date}.csv"
    v231_result = evaluate_predictions(dm, v231_file, predict_date, eval_date, "v2.3.1")

    # v2.4.0 Top10
    v240_file = PROJECT_ROOT / "data" / "prediction" / "results" / f"v2.4.0_top10_{predict_date}.csv"
    v240_result = evaluate_predictions(dm, v240_file, predict_date, eval_date, "v2.4.0")

    # 对比
    log.info("\n" + "=" * 80)
    log.info("对比总结")
    log.info("=" * 80)

    if v231_result and v240_result:
        log.info(f"\n{'指标':<20} {'v2.3.1':<15} {'v2.4.0':<15} {'改进':<15}")
        log.info("-" * 60)

        # T1前涨幅对比
        pre_t1_diff = v240_result["avg_pre_t1"] - v231_result["avg_pre_t1"]
        log.info(
            f"{'T1前平均涨幅':<20} {v231_result['avg_pre_t1']:>+10.1f}%     {v240_result['avg_pre_t1']:>+10.1f}%     {pre_t1_diff:>+10.1f}%"
        )

        # 实际收益对比
        return_diff = v240_result["avg_return"] - v231_result["avg_return"]
        log.info(
            f"{'实际平均收益':<20} {v231_result['avg_return']:>+10.1f}%     {v240_result['avg_return']:>+10.1f}%     {return_diff:>+10.1f}%"
        )

        # 胜率对比
        win_diff = v240_result["win_rate"] - v231_result["win_rate"]
        log.info(
            f"{'胜率':<20} {v231_result['win_rate']:>10.0f}%     {v240_result['win_rate']:>10.0f}%     {win_diff:>+10.0f}%"
        )

        log.info("\n" + "=" * 80)
        log.info("结论")
        log.info("=" * 80)

        if v240_result["avg_pre_t1"] < v231_result["avg_pre_t1"]:
            log.success("✅ v2.4.0成功过滤'追龙头'股票")
            log.info(f"   T1前涨幅从{v231_result['avg_pre_t1']:.1f}%降低到{v240_result['avg_pre_t1']:.1f}%")

        if v240_result["avg_return"] > v231_result["avg_return"]:
            log.success("✅ v2.4.0实际收益更高")
        elif v240_result["avg_return"] < v231_result["avg_return"]:
            log.warning(
                f"⚠️  v2.4.0实际收益较低（{v240_result['avg_return']:.1f}% vs {v231_result['avg_return']:.1f}%）"
            )

    log.info("")


if __name__ == "__main__":
    main()
