#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估v2.3.1模型的Top10表现

使用预测日期和评估日期，计算实际收益率
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log
from src.data.data_manager import DataManager


def evaluate_top10(predict_file, eval_date):
    """评估Top10的实际表现"""
    log.info("=" * 80)
    log.info("v2.3.1 Top10 评估")
    log.info("=" * 80)

    # 加载预测结果
    df_pred = pd.read_csv(predict_file)
    log.info(f"📊 加载预测结果: {len(df_pred)} 只股票")

    # 初始化数据管理器
    dm = DataManager()

    # 获取评估数据
    log.info(f"\n📅 评估日期: {eval_date}")
    eval_start = (datetime.strptime(eval_date, "%Y%m%d") - timedelta(days=10)).strftime("%Y%m%d")
    eval_end = (datetime.strptime(eval_date, "%Y%m%d") + timedelta(days=5)).strftime("%Y%m%d")

    # 批量获取评估数据
    stock_codes = df_pred["ts_code"].tolist()
    log.info(f"批量获取 {len(stock_codes)} 只股票的评估数据...")
    daily_data_dict = dm.batch_get_daily_data(stock_codes, eval_start, eval_end)
    log.success(
        f"✓ 批量获取完成: {len([k for k, v in daily_data_dict.items() if not v.empty])}/{len(stock_codes)} 只股票"
    )

    # 处理评估结果
    eval_results = []
    for _, row in df_pred.iterrows():
        ts_code = row["ts_code"]

        try:
            df_eval = daily_data_dict.get(ts_code)
            if df_eval is None or len(df_eval) == 0:
                continue

            df_eval["date_diff"] = abs(pd.to_datetime(df_eval["trade_date"]) - pd.to_datetime(eval_date))
            closest = df_eval.loc[df_eval["date_diff"].idxmin()]

            eval_price = closest["close"]
            predict_price = row["close"]
            return_pct = (eval_price / predict_price - 1) * 100

            result = row.to_dict()
            result["eval_price"] = eval_price
            result["return_pct"] = return_pct
            eval_results.append(result)
        except Exception:
            continue

    df_eval = pd.DataFrame(eval_results)

    if len(df_eval) == 0:
        log.error("没有评估数据")
        return None

    log.success(f"✓ 评估完成: {len(df_eval)} 只股票")

    # 统计信息
    log.info("\n" + "=" * 80)
    log.info("📊 v2.3.1 Top10 评估结果")
    log.info("=" * 80)

    avg_return = df_eval["return_pct"].mean()
    median_return = df_eval["return_pct"].median()
    win_rate = (df_eval["return_pct"] > 0).sum() / len(df_eval) * 100
    max_return = df_eval["return_pct"].max()
    min_return = df_eval["return_pct"].min()
    std_return = df_eval["return_pct"].std()

    log.info("\n整体统计:")
    log.info(f"  有效股票数: {len(df_eval)}")
    log.info(f"  平均收益率: {avg_return:>+7.2f}%")
    log.info(f"  中位数收益: {median_return:>+7.2f}%")
    log.info(f"  胜率: {win_rate:>7.1f}%")
    log.info(f"  最高收益: {max_return:>+7.2f}%")
    log.info(f"  最低收益: {min_return:>+7.2f}%")
    log.info(f"  标准差: {std_return:>7.2f}%")

    # 显示详细结果
    log.info("\n📋 Top10 详细结果:")
    log.info("-" * 100)
    log.info(
        f"{'排名':<4} {'代码':<12} {'名称':<10} {'综合评分':<10} {'校准概率':<10} {'预期收益':<10} {'预测价':<8} {'评估价':<8} {'收益率':<10}"
    )
    log.info("-" * 100)

    for i, (_, row) in enumerate(df_eval.iterrows(), 1):
        log.info(
            f"{i:<4} {row['ts_code']:<12} {row['name']:<10} "
            f"{row['final_score']:<10.4f} {row['calibrated_probability']:<10.4f} "
            f"{row['expected_return_score']:<10.4f} {row['close']:<8.2f} "
            f"{row['eval_price']:<8.2f} {row['return_pct']:>+9.2f}%"
        )

    # 保存评估结果
    output_dir = PROJECT_ROOT / "data" / "prediction" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    predict_date = Path(predict_file).stem.split("_")[-1]
    output_file = output_dir / f"v2.3.1_top10_eval_{predict_date}_to_{eval_date}.csv"
    df_eval.to_csv(output_file, index=False, encoding="utf-8-sig")
    log.success(f"\n💾 评估结果已保存: {output_file}")

    return {
        "avg_return": avg_return,
        "median_return": median_return,
        "win_rate": win_rate,
        "max_return": max_return,
        "min_return": min_return,
        "std_return": std_return,
        "count": len(df_eval),
    }


def compare_with_v230(predict_date, eval_date):
    """对比v2.3.0和v2.3.1的结果"""
    log.info("\n" + "=" * 80)
    log.info("v2.3.0 vs v2.3.1 对比")
    log.info("=" * 80)

    # 加载v2.3.0的评估结果
    v230_file = PROJECT_ROOT / "data" / "prediction" / "evaluation" / "v2.3.0_top100_20251212.csv"
    if not v230_file.exists():
        log.warning("v2.3.0评估结果不存在，跳过对比")
        return

    df_v230 = pd.read_csv(v230_file)
    df_v230_top10 = df_v230.head(10)

    # 加载v2.3.1的评估结果
    v231_file = (
        PROJECT_ROOT / "data" / "prediction" / "evaluation" / f"v2.3.1_top10_eval_{predict_date}_to_{eval_date}.csv"
    )
    if not v231_file.exists():
        log.warning("v2.3.1评估结果不存在，跳过对比")
        return

    df_v231 = pd.read_csv(v231_file)

    # 计算v2.3.0 Top10的统计
    v230_valid = df_v230_top10[df_v230_top10["return_pct"].notna()]
    if len(v230_valid) > 0:
        v230_avg = v230_valid["return_pct"].mean()
        v230_win = (v230_valid["return_pct"] > 0).sum() / len(v230_valid) * 100
    else:
        v230_avg = 0
        v230_win = 0

    # v2.3.1的统计
    v231_avg = df_v231["return_pct"].mean()
    v231_win = (df_v231["return_pct"] > 0).sum() / len(df_v231) * 100

    log.info("\n对比结果:")
    log.info(f"{'指标':<20} {'v2.3.0':<15} {'v2.3.1':<15} {'变化':<15}")
    log.info("-" * 65)
    log.info(f"{'平均收益率':<20} {v230_avg:>+7.2f}%{'':<6} {v231_avg:>+7.2f}%{'':<6} {v231_avg - v230_avg:>+7.2f}%")
    log.info(f"{'胜率':<20} {v230_win:>7.1f}%{'':<6} {v231_win:>7.1f}%{'':<6} {v231_win - v230_win:>+7.1f}%")

    improvement = v231_avg - v230_avg
    if improvement > 0:
        log.success(f"\n✓ v2.3.1相比v2.3.0，平均收益率提升了 {improvement:.2f}%")
    else:
        log.warning(f"\n⚠️  v2.3.1相比v2.3.0，平均收益率下降了 {abs(improvement):.2f}%")


def main():
    parser = argparse.ArgumentParser(description="评估v2.3.1模型的Top10表现")
    parser.add_argument("--predict-date", type=str, default="20251212", help="预测日期")
    parser.add_argument("--eval-date", type=str, default="20251231", help="评估日期")
    parser.add_argument("--predict-file", type=str, default=None, help="预测结果文件（可选）")
    args = parser.parse_args()

    # 确定预测文件
    if args.predict_file:
        predict_file = Path(args.predict_file)
    else:
        predict_file = PROJECT_ROOT / "data" / "prediction" / "results" / f"v2.3.1_top10_{args.predict_date}.csv"

    if not predict_file.exists():
        log.error(f"预测文件不存在: {predict_file}")
        return

    # 评估
    stats = evaluate_top10(predict_file, args.eval_date)

    # 对比
    if stats:
        compare_with_v230(args.predict_date, args.eval_date)


if __name__ == "__main__":
    main()
