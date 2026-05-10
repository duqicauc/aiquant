#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.9.5 批量预测脚本

使用 v2.9.5-ensemble 模型，修复温度缩放 + 动态类别权重 + 适度分歧。
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.predictor import EnsemblePredictor
from src.utils.logger import log

OUTPUT_DIR = PROJECT_ROOT / "data" / "prediction" / "v295_stk_factor"


def main():
    parser = argparse.ArgumentParser(description="v2.9.5 批量预测")
    parser.add_argument("--start-date", required=True, help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="结束日期 YYYYMMDD")
    parser.add_argument("--lookback", type=int, default=34, help="回看天数")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="输出目录")
    args = parser.parse_args()

    predictor = EnsemblePredictor(model_version="v2.9.5-ensemble")
    results = predictor.predict_range(args.start_date, args.end_date, args.lookback)

    output_dir = Path(args.output_dir)
    for date, df in results.items():
        predictor.save_results(df, date, output_dir)

    log.success("=" * 80)
    log.success(f"v2.9.5 预测完成: {len(results)} 天")
    log.success("=" * 80)


if __name__ == "__main__":
    main()
