#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对已有预测结果进行基本面筛选（支持自定义市值上限）

用法：
    python scripts/screen_existing_predictions_custom.py --file data/prediction/results/v270_ensemble_top50_20260116.csv --date 20260116 --market-cap-max 200
"""
import sys
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import log
from src.data.data_manager import DataManager
from src.models.screening.fundamental_screener import FundamentalScreener


def screen_predictions(prediction_file: str, trade_date: str, market_cap_max: int = 100):
    """
    对已有预测结果进行基本面筛选（支持自定义市值上限）

    Args:
        prediction_file: 预测结果文件路径
        trade_date: 交易日期（YYYYMMDD）
        market_cap_max: 市值上限（单位：亿）
    """
    log.info("=" * 80)
    log.info("对已有预测结果进行基本面筛选（自定义市值上限）")
    log.info("=" * 80)
    log.info(f"预测文件: {prediction_file}")
    log.info(f"交易日期: {trade_date}")
    log.info(f"市值上限: {market_cap_max}亿")
    log.info("")

    # 读取预测结果
    if not Path(prediction_file).exists():
        log.error(f"文件不存在: {prediction_file}")
        return

    df = pd.read_csv(prediction_file)
    log.info(f"加载预测结果: {len(df)} 只股票")

    # 检查必要的列
    if "ts_code" not in df.columns:
        log.error("预测结果文件缺少ts_code列")
        return

    # 如果文件已经排序，取top50；否则取全部
    if len(df) > 50:
        log.info(f"文件包含{len(df)}只股票，取前50只进行筛选")
        df = df.head(50)
    else:
        log.info(f"文件包含{len(df)}只股票，全部进行筛选")

    # 初始化数据管理器和筛选器（使用自定义市值上限）
    dm = DataManager()
    fundamental_screener = FundamentalScreener(
        dm,
        config={
            "enabled": True,
            "market_cap_min": 100000,  # 10亿（万元）
            "market_cap_max": market_cap_max * 10000,  # 自定义上限（万元）
            "revenue_min": 5e8,  # 营业收入>5亿（元）- 标准方案
            "net_profit_min": 5000000,  # 净利润>500万（元）- 标准方案
            "roe_min": 5,  # ROE>5% - 标准方案
            "roa_min": 2,  # ROA>2% - 标准方案
        },
    )

    # 进行基本面筛选
    log.info("\n开始基本面筛选...")
    df_screened = fundamental_screener.screen_stocks(df, trade_date)

    # 统计结果
    passed = df_screened[df_screened["fundamental_pass"] == True]
    failed = df_screened[df_screened["fundamental_pass"] == False]

    log.info("\n" + "=" * 80)
    log.info("筛选结果统计")
    log.info("=" * 80)
    log.info(f"总股票数: {len(df_screened)}")
    log.info(f"通过筛选: {len(passed)} ({len(passed)/len(df_screened)*100:.1f}%)")
    log.info(f"未通过筛选: {len(failed)} ({len(failed)/len(df_screened)*100:.1f}%)")

    # 显示通过筛选的股票
    if len(passed) > 0:
        log.info("\n" + "=" * 80)
        log.info(f"通过基本面筛选的股票 ({len(passed)}只)")
        log.info("=" * 80)

        # 按原始排序显示
        if "probability" in passed.columns:
            passed = passed.sort_values("probability", ascending=False)
        elif "final_score" in passed.columns:
            passed = passed.sort_values("final_score", ascending=False)

        log.info(f"\n{'排名':<4} {'代码':<12} {'名称':<10} {'概率/评分':>12}")
        log.info("-" * 50)

        for i, (_, row) in enumerate(passed.iterrows(), 1):
            score = row.get("probability", row.get("final_score", row.get("calibrated_probability", 0)))
            name = row.get("name", "")
            log.info(f"{i:<4} {row['ts_code']:<12} {name:<10} {score:>12.4f}")

    # 保存筛选结果
    output_dir = PROJECT_ROOT / "data" / "prediction" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存通过筛选的股票
    if len(passed) > 0:
        output_file = (
            output_dir / f"{Path(prediction_file).stem}_fundamental_screened_{market_cap_max}亿_{trade_date}.csv"
        )
        passed.to_csv(output_file, index=False, encoding="utf-8-sig")
        log.success(f"\n✓ 通过筛选的股票已保存: {output_file}")

    # 保存完整筛选结果（包含通过/未通过信息）
    full_output_file = output_dir / f"{Path(prediction_file).stem}_fundamental_full_{market_cap_max}亿_{trade_date}.csv"
    df_screened.to_csv(full_output_file, index=False, encoding="utf-8-sig")
    log.info(f"完整筛选结果已保存: {full_output_file}")

    return df_screened


def main():
    parser = argparse.ArgumentParser(description="对已有预测结果进行基本面筛选（支持自定义市值上限）")
    parser.add_argument("--file", type=str, required=True, help="预测结果文件路径（CSV格式）")
    parser.add_argument("--date", type=str, required=True, help="交易日期（YYYYMMDD）")
    parser.add_argument(
        "--market-cap-max", type=int, default=200, help="市值上限（单位：亿），默认200亿。可选值：100, 200, 300, 500等"
    )

    args = parser.parse_args()

    screen_predictions(args.file, args.date, args.market_cap_max)


if __name__ == "__main__":
    main()
