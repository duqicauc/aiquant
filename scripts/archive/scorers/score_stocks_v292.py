#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.9.2 CatBoost 单模型股票评分脚本

Usage:
    # 预测今天
    python scripts/score_stocks_v292.py

    # 预测指定日期（历史回测）
    python scripts/score_stocks_v292.py --date 20260422

    # 预测日期范围（批量回测）
    python scripts/score_stocks_v292.py --start-date 20250101 --end-date 20250331

    # 指定输出目录
    python scripts/score_stocks_v292.py --date 20260422 --output-dir data/prediction/v292_stk_factor
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.catboost_predictor import CatBoostPredictor
from src.utils.logger import log


def main():
    parser = argparse.ArgumentParser(description="v2.9.2 CatBoost 单模型股票评分")
    parser.add_argument("--date", help="预测日期 YYYYMMDD，默认今天")
    parser.add_argument("--start-date", help="批量预测开始日期 YYYYMMDD")
    parser.add_argument("--end-date", help="批量预测结束日期 YYYYMMDD")
    parser.add_argument(
        "--output-dir",
        default="data/prediction/v292_stk_factor",
        help="输出目录",
    )
    parser.add_argument("--model-version", default="v2.9.2-catboost", help="模型版本")
    parser.add_argument("--lookback-days", type=int, default=70, help="回看天数")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = CatBoostPredictor(model_version=args.model_version)

    if args.start_date and args.end_date:
        # 批量预测
        results = predictor.predict_range(
            args.start_date, args.end_date, lookback_days=args.lookback_days
        )
        for date_str, df in results.items():
            predictor.save_results(df, date_str, output_dir)

    else:
        # 单日预测
        prediction_date = args.date or datetime.now().strftime("%Y%m%d")
        df = predictor.predict_date(prediction_date, lookback_days=args.lookback_days)

        if not df.empty:
            predictor.save_results(df, prediction_date, output_dir)

            # 打印 Top10
            log.info("\nTop 10 股票:")
            top10 = df.head(10)[["rank", "ts_code", "name", "prob", "prob_raw", "close", "pct_chg"]]
            for _, row in top10.iterrows():
                log.info(
                    f"  #{int(row['rank']):2d} {row['ts_code']} {row['name'][:6]:6s} "
                    f"prob={row['prob']:.4f} (raw={row['prob_raw']:.4f}) close={row['close']:.2f}"
                )


if __name__ == "__main__":
    main()
