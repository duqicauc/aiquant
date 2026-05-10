#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速回测 v2.9.2-catboost-v291-params 模型

1. 预测 2024Q4
2. Realistic 回测
3. Integrated 回测（加 sector filter）
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.catboost_predictor import CatBoostPredictor
from src.backtest.backtester_realistic import RealisticBacktester
from src.utils.logger import log

PRED_DIR = PROJECT_ROOT / "data" / "prediction" / "v292_v291params_2024q4"
EVAL_DIR = PROJECT_ROOT / "data" / "prediction" / "evaluation"


def predict():
    """预测 2024Q4"""
    predictor = CatBoostPredictor(model_version="v2.9.2-catboost-v291-params")
    results = predictor.predict_range("20241008", "20241231", lookback_days=70)
    for date, df in results.items():
        predictor.save_results(df, date, PRED_DIR)
    log.success(f"预测完成: {len(results)} 天")
    return len(results)


def backtest_realistic():
    """Realistic 回测"""
    bt = RealisticBacktester(
        prediction_dir=str(PRED_DIR),
        initial_capital=10_000_000,
        per_stock_amount=300_000,
        top_n_buy=10,
        stop_loss_pct=10.0,
        trailing_stop_pct=2.0,
        trailing_stop_activation=5.0,
        enable_sector_filter=False,
        ma_window=5,
        ma_consecutive_days=2,
        buy_slippage_bps=15.0,
        sell_slippage_bps=20.0,
        commission_rate=0.00025,
        min_commission=5.0,
        stamp_duty_rate=0.001,
        min_amount=10_000,
    )
    result = bt.run(start_date="20241008", end_date="20241231")
    if result:
        bt.save_results(result, str(EVAL_DIR / "v292_v291params_realistic"))
        summary = result.get("summary", {})
        log.success(f"Realistic 回测: 总收益率={summary.get('total_return', 0):.2%}")
        return summary.get("total_return", 0)
    return None


def backtest_integrated():
    """Integrated 回测（加 sector filter）"""
    bt = RealisticBacktester(
        prediction_dir=str(PRED_DIR),
        initial_capital=10_000_000,
        per_stock_amount=300_000,
        top_n_buy=10,
        stop_loss_pct=10.0,
        trailing_stop_pct=2.0,
        trailing_stop_activation=5.0,
        enable_sector_filter=True,
        ma_window=5,
        ma_consecutive_days=2,
        buy_slippage_bps=15.0,
        sell_slippage_bps=20.0,
        commission_rate=0.00025,
        min_commission=5.0,
        stamp_duty_rate=0.001,
        min_amount=10_000,
    )
    result = bt.run(start_date="20241008", end_date="20241231")
    if result:
        bt.save_results(result, str(EVAL_DIR / "v292_v291params_integrated"))
        summary = result.get("summary", {})
        log.success(f"Integrated 回测: 总收益率={summary.get('total_return', 0):.2%}")
        return summary.get("total_return", 0)
    return None


def main():
    log.info("=" * 80)
    log.info("v2.9.2-catboost-v291-params 快速回测")
    log.info("=" * 80)

    # 1. 预测
    n_days = predict()
    if n_days == 0:
        log.error("预测失败")
        return

    # 2. Realistic 回测
    realistic_return = backtest_realistic()

    # 3. Integrated 回测
    integrated_return = backtest_integrated()

    # 4. 汇总
    log.info("\n" + "=" * 80)
    log.info("回测结果汇总")
    log.info("=" * 80)
    log.info(f"模型: v2.9.2-catboost-v291-params (AUC=0.9609)")
    log.info(f"Realistic 2024Q4:  {realistic_return:.2%}" if realistic_return else "Realistic: 失败")
    log.info(f"Integrated 2024Q4: {integrated_return:.2%}" if integrated_return else "Integrated: 失败")

    # 对比
    log.info("\n对比:")
    log.info(f"  v291-ensemble integrated:     +18.10%")
    log.info(f"  v292-cons realistic:          +12.11%")
    log.info(f"  v292-cons integrated(P1后):   +5.24%")


if __name__ == "__main__":
    main()
