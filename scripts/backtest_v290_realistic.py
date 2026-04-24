#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.9.1 实盘策略回测脚本

与 v2.8.1 回测流程一致，使用 v2.9.1 预测结果。
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.backtester_realistic import RealisticBacktester

OUTPUT_DIR = PROJECT_ROOT / "data" / "prediction" / "evaluation"


def main():
    parser = argparse.ArgumentParser(description="v2.9.1 实盘策略回测")
    parser.add_argument("--start-date", required=True, help="回测开始日期 YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="回测结束日期 YYYYMMDD")
    parser.add_argument("--prediction-dir", default="data/prediction/v290_stk_factor", help="预测结果目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "v290_realistic"), help="输出目录")
    parser.add_argument("--per-stock", type=float, default=300_000, help="每只股票买入金额")
    parser.add_argument("--top-n", type=int, default=10, help="每日买入数量")
    parser.add_argument("--stop-loss", type=float, default=4.0, help="止损百分比")
    parser.add_argument("--take-profit", type=float, default=5.0, help="止盈百分比")
    parser.add_argument("--capital", type=float, default=10_000_000, help="初始资金")
    args = parser.parse_args()

    prediction_dir = PROJECT_ROOT / args.prediction_dir
    output_dir = PROJECT_ROOT / args.output_dir

    bt = RealisticBacktester(
        prediction_dir=str(prediction_dir),
        initial_capital=args.capital,
        per_stock_amount=args.per_stock,
        top_n_buy=args.top_n,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        ma_window=5,
        ma_consecutive_days=2,
        buy_slippage_bps=15.0,
        sell_slippage_bps=20.0,
        commission_rate=0.00025,
        min_commission=5.0,
        stamp_duty_rate=0.001,
        min_amount=10_000,
    )

    result = bt.run(start_date=args.start_date, end_date=args.end_date)

    if result:
        bt.save_results(result, str(output_dir))


if __name__ == "__main__":
    main()
