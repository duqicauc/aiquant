#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.7.0 同期回测对比

用 v2.7.0 模型在相同回测期（2026-03-28 ~ 2026-04-22）运行预测和回测，
与 v2.8.0 结果对比，判断是模型问题还是市场环境问题。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.backtester import StrategyBacktester
from src.prediction.predictor import EnsemblePredictor
from src.utils.logger import log

# 临时修改模型路径为 v2.7.0
EnsemblePredictor.MODEL_DIR = (
    PROJECT_ROOT / "data" / "models" / "backup_v270_20260423_000920" / "v2.7.0-ensemble" / "model"
)


def main():
    log.info("=" * 80)
    log.info("v2.7.0 同期回测对比")
    log.info("=" * 80)

    start_date = "20260327"
    end_date = "20260421"
    backtest_start = "20260328"
    backtest_end = "20260422"

    output_dir = PROJECT_ROOT / "data" / "prediction" / "v270_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 用 v2.7.0 模型预测
    log.info("\n[1/2] v2.7.0 模型预测...")
    predictor = EnsemblePredictor()
    results = predictor.predict_range(start_date, end_date, lookback_days=70)

    for date, df in results.items():
        predictor.save_results(df, date, output_dir)

    log.success(f"预测完成: {len(results)} 天")

    # 2. 策略回测
    log.info("\n[2/2] 策略回测...")
    bt = StrategyBacktester(
        prediction_dir=str(output_dir),
        initial_capital=10_000_000,
        top_n_buy=10,
        stop_loss_pct=4.0,
        ma_window=5,
        ma_consecutive_days=2,
        buy_slippage_bps=15.0,
        sell_slippage_bps=20.0,
    )

    result = bt.run(start_date=backtest_start, end_date=backtest_end)

    if result:
        eval_dir = PROJECT_ROOT / "data" / "prediction" / "evaluation"
        bt.save_results(result, str(eval_dir / "v270_comparison"))

    log.success("=" * 80)
    log.success("v2.7.0 同期回测完成！")
    log.success("=" * 80)


if __name__ == "__main__":
    main()
