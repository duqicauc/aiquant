#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.8.0 策略回测脚本（标准化版本）

使用预计算的预测结果进行回测：
- 买入: Top10，前一日选股，当日开盘价买入
- 卖出: 4%止损(close触发) 或 MA5_cd2退出
- 无 trailing stop
- 无 sector limit
- 初始资金: 1000万
- 滑点: 买入15bp，卖出20bp

用法:
    python scripts/backtest_v280_strategy.py --start-date 20260328 --end-date 20260422
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.backtester import StrategyBacktester


def main():
    parser = argparse.ArgumentParser(description="v2.8.0 策略回测")
    parser.add_argument("--start-date", default="20260328", help="回测开始日期 YYYYMMDD")
    parser.add_argument("--end-date", default="20260422", help="回测结束日期 YYYYMMDD")
    parser.add_argument("--prediction-dir", default="data/prediction/v280_stk_factor", help="预测结果目录")
    parser.add_argument("--output-dir", default="data/prediction/evaluation", help="输出目录")
    parser.add_argument("--top-n", type=int, default=10, help="每日买入数量")
    parser.add_argument("--stop-loss", type=float, default=4.0, help="止损百分比")
    parser.add_argument("--capital", type=float, default=10_000_000, help="初始资金")
    args = parser.parse_args()

    prediction_dir = PROJECT_ROOT / args.prediction_dir
    output_dir = PROJECT_ROOT / args.output_dir

    bt = StrategyBacktester(
        prediction_dir=str(prediction_dir),
        initial_capital=args.capital,
        top_n_buy=args.top_n,
        stop_loss_pct=args.stop_loss,
        ma_window=5,
        ma_consecutive_days=2,
        buy_slippage_bps=15.0,
        sell_slippage_bps=20.0,
    )

    result = bt.run(start_date=args.start_date, end_date=args.end_date)

    if result:
        bt.save_results(result, str(output_dir))


if __name__ == "__main__":
    main()
