#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试保守版模型在 2024Q4 的 realistic 回测表现"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.catboost_predictor import CatBoostPredictor
from src.backtest.backtester_realistic import RealisticBacktester

# 1. 生成预测
print("=== 生成保守版 2024Q4 预测 ===")
predictor = CatBoostPredictor('v2.9.2-catboost-conservative')

pred_dir = PROJECT_ROOT / "data" / "prediction" / "v292_conservative_stk_factor"
pred_dir.mkdir(parents=True, exist_ok=True)

results = predictor.predict_range("20241001", "20241231", lookback_days=70)

for date, df in results.items():
    predictor.save_results(df, date, pred_dir)

print(f"预测完成: {len(results)} 天")

# 2. 跑 realistic 回测
print("\n=== 保守版 realistic 回测 ===")
bt = RealisticBacktester(
    prediction_dir=str(pred_dir),
    initial_capital=10_000_000,
    per_stock_amount=300_000,
    top_n_buy=10,
    stop_loss_pct=4.0,
    trailing_stop_pct=5.0,
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

result = bt.run(start_date="20241001", end_date="20241231")
if result:
    print(f"\n结果: 初始资金={result['initial_capital']:,.0f}, 最终资金={result['final_capital']:,.0f}, 收益率={result['total_return']:.2f}%")
    print(f"交易次数: {result['total_trades']}, 胜率: {result.get('win_rate', 0):.1f}%")
