#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.8.1 实盘策略回测脚本（贴近实盘版本）

策略参数:
- 每支股票买入金额: 300,000 元
- 先买后卖，T+1 资金可用
- 4%止损 + MA5_cd2 退出(跌出Top50, T+1收盘卖)
- 含交易费用 + 涨跌停/停牌/量能约束

Usage:
    python scripts/backtest_v281_realistic.py --start-date 20260328 --end-date 20260422
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.backtester_realistic import RealisticBacktester

OUTPUT_DIR = PROJECT_ROOT / "data" / "prediction" / "evaluation"


def main():
    parser = argparse.ArgumentParser(description="v2.8.1 实盘策略回测")
    parser.add_argument("--start-date", required=True, help="回测开始日期 YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="回测结束日期 YYYYMMDD")
    parser.add_argument("--prediction-dir", default="data/prediction/v281_stk_factor", help="预测结果目录")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "v281_realistic"), help="输出目录")
    parser.add_argument("--per-stock", type=float, default=300_000, help="每只股票买入金额")
    parser.add_argument("--top-n", type=int, default=10, help="每日买入数量")
    parser.add_argument("--stop-loss", type=float, default=4.0, help="止损百分比")
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
        ma_window=5,
        ma_consecutive_days=2,
        buy_slippage_bps=15.0,
        sell_slippage_bps=20.0,
        commission_rate=0.00025,
        min_commission=5.0,
        stamp_duty_rate=0.001,
        min_amount=10_000,  # 1000万（Tushare amount单位为千元）
    )

    result = bt.run(start_date=args.start_date, end_date=args.end_date)

    if result:
        bt.save_results(result, str(output_dir))


if __name__ == "__main__":
    main()
