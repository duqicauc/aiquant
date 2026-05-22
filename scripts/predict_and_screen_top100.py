#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测并筛选Top100股票（包含基本面筛选的综合结果）

流程：
1. 运行v2.7.0模型预测全市场股票
2. 对Top100进行基本面筛选
3. 生成综合结果文件（包含模型评分和基本面筛选结果）

用法：
    python scripts/predict_and_screen_top100.py --date 20260119 --market-cap-max 200
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入预测函数
from scripts.predict_v270_ensemble_top50 import (
    get_valid_stocks,
    load_ensemble_model,
    process_single_stock,
)
from src.data.data_manager import DataManager
from src.models.screening.fundamental_screener import FundamentalScreener
from src.utils.logger import log


def predict_and_screen_top100(predict_date: str, market_cap_max: int = 200):
    """
    预测并筛选Top100股票

    Args:
        predict_date: 预测日期（YYYYMMDD）
        market_cap_max: 市值上限（单位：亿）
    """
    log.info("=" * 80)
    log.info("v2.7.0模型预测 + Top100基本面筛选")
    log.info("=" * 80)
    log.info(f"预测日期: {predict_date}")
    log.info(f"市值上限: {market_cap_max}亿")
    log.info("")

    # 步骤1：运行预测
    log.info("=" * 80)
    log.info("步骤1：运行v2.7.0模型预测")
    log.info("=" * 80)

    # 加载模型
    models, feature_names, weights = load_ensemble_model()

    # 初始化数据管理器
    dm = DataManager()

    # 获取有效股票
    stock_list = get_valid_stocks(dm, predict_date)

    # 预测
    log.info(f"\n开始预测 {len(stock_list)} 只股票...")

    results = []
    total = len(stock_list)

    for idx, (_, row) in enumerate(stock_list.iterrows()):
        if (idx + 1) % 100 == 0:
            log.info(f"进度: {idx+1}/{total} | 已评分: {len(results)}")

        result = process_single_stock(dm, row["ts_code"], row["name"], predict_date, feature_names, models, weights)

        if result:
            results.append(result)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("probability", ascending=False)

    log.success(f"\n✓ 预测完成: {len(df_results)} 只股票")

    # 保存全市场评分结果
    output_dir = PROJECT_ROOT / "data" / "prediction" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results_file = output_dir / f"v270_ensemble_all_{predict_date}.csv"
    df_results.to_csv(all_results_file, index=False)
    log.info(f"全市场评分已保存: {all_results_file}")

    # 步骤2：对Top100进行基本面筛选
    log.info("\n" + "=" * 80)
    log.info("步骤2：对Top100进行基本面筛选")
    log.info("=" * 80)

    top100 = df_results.head(100).copy()
    top100["model_rank"] = range(1, len(top100) + 1)

    log.info("取Top100股票进行基本面筛选")

    # 初始化基本面筛选器
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
    top100_screened = fundamental_screener.screen_stocks(top100, predict_date)

    # 统计结果
    passed = top100_screened[top100_screened["fundamental_pass"]]
    failed = top100_screened[~top100_screened["fundamental_pass"]]

    log.info("\n" + "=" * 80)
    log.info("筛选结果统计")
    log.info("=" * 80)
    log.info(f"总股票数: {len(top100_screened)}")
    log.info(f"通过筛选: {len(passed)} ({len(passed)/len(top100_screened)*100:.1f}%)")
    log.info(f"未通过筛选: {len(failed)} ({len(failed)/len(top100_screened)*100:.1f}%)")

    # 添加基本面排名（通过筛选的股票按模型评分排序）
    top100_screened["fundamental_rank"] = None
    if len(passed) > 0:
        passed_sorted = passed.sort_values("probability", ascending=False)
        for idx, (i, row) in enumerate(passed_sorted.iterrows(), 1):
            top100_screened.at[i, "fundamental_rank"] = idx

    # 步骤3：生成综合结果文件
    log.info("\n" + "=" * 80)
    log.info("步骤3：生成综合结果文件")
    log.info("=" * 80)

    # 重新排列列顺序，让重要信息在前面
    cols = ["model_rank", "fundamental_rank", "fundamental_pass", "ts_code", "name", "probability"]
    cols.extend([c for c in top100_screened.columns if c not in cols])

    # 确保所有列都存在
    available_cols = [c for c in cols if c in top100_screened.columns]
    available_cols.extend([c for c in top100_screened.columns if c not in available_cols])

    output_file = output_dir / f"v270_top100_fundamental_combined_{market_cap_max}亿_{predict_date}.csv"
    top100_screened[available_cols].to_csv(output_file, index=False, encoding="utf-8-sig")
    log.success(f"\n✓ 综合结果已保存: {output_file}")

    log.info("\n文件包含以下信息：")
    log.info("  - model_rank: 模型评分排名（1-100）")
    log.info("  - fundamental_rank: 基本面筛选排名（仅通过筛选的股票有排名）")
    log.info("  - fundamental_pass: 是否通过基本面筛选（True/False）")
    log.info("  - fundamental_reason: 未通过筛选的原因")
    log.info("  - probability: 模型预测概率")
    log.info("\n使用建议：")
    log.info("  - 按model_rank排序：查看模型评分Top100")
    log.info("  - 按fundamental_rank排序：查看通过基本面筛选的股票（按模型评分排序）")
    log.info("  - 筛选fundamental_pass=True：只查看通过基本面筛选的股票")

    # 显示通过筛选的股票（按模型评分排序）
    if len(passed) > 0:
        log.info("\n" + "=" * 80)
        log.info(f"通过基本面筛选的股票 ({len(passed)}只，按模型评分排序)")
        log.info("=" * 80)

        passed_sorted = passed.sort_values("probability", ascending=False)

        log.info(f"\n{'模型排名':<8} {'基本面排名':<10} {'代码':<12} {'名称':<10} {'模型概率':>12}")
        log.info("-" * 70)

        for _, row in passed_sorted.iterrows():
            model_rank = row["model_rank"]
            fund_rank = int(row["fundamental_rank"]) if pd.notna(row["fundamental_rank"]) else "-"
            prob = row.get("probability", 0)
            name = row.get("name", "")
            log.info(f"{model_rank:<8} {fund_rank:<10} {row['ts_code']:<12} {name:<10} {prob:>12.4f}")

    return top100_screened


def main():
    parser = argparse.ArgumentParser(description="预测并筛选Top100股票（包含基本面筛选的综合结果）")
    parser.add_argument("--date", type=str, required=True, help="预测日期（YYYYMMDD）")
    parser.add_argument("--market-cap-max", type=int, default=200, help="市值上限（单位：亿），默认200亿")

    args = parser.parse_args()

    predict_and_screen_top100(args.date, args.market_cap_max)


if __name__ == "__main__":
    main()
