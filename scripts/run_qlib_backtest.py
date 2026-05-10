#!/usr/bin/env python3
"""
qlib 风格回测运行入口

Usage:
    python scripts/run_qlib_backtest.py \
        --prediction_dir data/prediction/v3.0.0 \
        --start 20260101 --end 20260430 \
        --top_k 10 --drop_n 5 \
        --output_dir data/backtest/qlib_v3
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.qlib_backtest import QlibStyleBacktest
from src.utils.logger import log


def main():
    parser = argparse.ArgumentParser(description="qlib 风格回测")
    parser.add_argument("--prediction_dir", required=True, help="预测结果目录")
    parser.add_argument("--start", required=True, help="开始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="结束日期 YYYYMMDD")
    parser.add_argument("--top_k", type=int, default=10, help="每日持仓数量")
    parser.add_argument("--drop_n", type=int, default=5, help="dropout 阈值")
    parser.add_argument("--hold_days", type=int, default=5, help="最少持有天数")
    parser.add_argument("--output_dir", default="data/backtest/qlib", help="输出目录")
    parser.add_argument("--capital", type=float, default=10_000_000, help="初始资金")
    args = parser.parse_args()

    log.info(f"qlib 回测入口: {args.start} ~ {args.end}")

    bt = QlibStyleBacktest(
        prediction_dir=args.prediction_dir,
        initial_capital=args.capital,
    )
    result = bt.run(
        start_date=args.start,
        end_date=args.end,
        top_k=args.top_k,
        drop_n=args.drop_n,
        hold_days=args.hold_days,
    )

    if not result:
        log.error("回测失败")
        return

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bt.save_report(result, output_dir / "report.json")
    result["daily_returns"].to_csv(output_dir / "daily_returns.csv", index=False)
    result["portfolio"].to_csv(output_dir / "portfolio.csv", index=False)

    log.success(f"回测结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
