#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.8.0 批量预测脚本（标准化版本）

使用标准化数据获取 + 特征工程 + 预测流程：
- src.data.tushare_data_provider  统一数据获取
- src.features.feature_engineer   统一特征工程
- src.prediction.predictor        统一预测

用法：
    python scripts/predict_v280_with_stk_factor.py --start-date 20260327 --end-date 20260421
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.predictor import EnsemblePredictor
from src.utils.logger import log

OUTPUT_DIR = PROJECT_ROOT / "data" / "prediction" / "v280_stk_factor"


def main():
    parser = argparse.ArgumentParser(description="v2.8.0 批量预测（标准化版本）")
    parser.add_argument("--start-date", required=True, help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="结束日期 YYYYMMDD")
    parser.add_argument("--lookback", type=int, default=70, help="回看天数")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="输出目录")
    args = parser.parse_args()

    predictor = EnsemblePredictor()
    results = predictor.predict_range(args.start_date, args.end_date, args.lookback)

    output_dir = Path(args.output_dir)
    for date, df in results.items():
        predictor.save_results(df, date, output_dir)

    log.success("=" * 80)
    log.success(f"预测完成: {len(results)} 天")
    log.success("=" * 80)


if __name__ == "__main__":
    main()
