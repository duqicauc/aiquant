#!/usr/bin/env python3
"""
vectorbt 回测运行入口

Usage:
    python scripts/run_vbt_backtest.py \
        --prediction_dir data/prediction/v3.0.0 \
        --start 20260101 --end 20260430 \
        --top_k 10 --scan
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.vbt_backtest import VBTBacktest
from src.utils.logger import log


def main():
    parser = argparse.ArgumentParser(description="vectorbt 回测")
    parser.add_argument("--prediction_dir", required=True, help="预测结果目录")
    parser.add_argument("--start", required=True, help="开始日期 YYYYMMDD")
    parser.add_argument("--end", required=True, help="结束日期 YYYYMMDD")
    parser.add_argument("--top_k", type=int, default=10, help="每日持仓数量")
    parser.add_argument("--drop_n", type=int, default=5, help="dropout 阈值")
    parser.add_argument("--hold_days", type=int, default=5, help="最少持有天数")
    parser.add_argument("--stop_loss", type=float, default=None, help="止损比例，如 0.04")
    parser.add_argument("--scan", action="store_true", help="执行参数扫描")
    parser.add_argument("--output_dir", default="data/backtest/vbt", help="输出目录")
    parser.add_argument("--capital", type=float, default=10_000_000, help="初始资金")
    args = parser.parse_args()

    log.info(f"vectorbt 回测入口: {args.start} ~ {args.end}")

    bt = VBTBacktest(
        prediction_dir=args.prediction_dir,
        initial_capital=args.capital,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.scan:
        df_scan = bt.param_scan(
            start_date=args.start,
            end_date=args.end,
            top_k_range=[5, 10, 15, 20],
            stop_loss_range=[None, 0.03, 0.05],
        )
        df_scan.to_csv(output_dir / "param_scan.csv", index=False)
        log.success(f"参数扫描结果已保存: {output_dir / 'param_scan.csv'}")
        print(df_scan.to_string(index=False))
    else:
        result = bt.run(
            start_date=args.start,
            end_date=args.end,
            top_k=args.top_k,
            drop_n=args.drop_n,
            hold_days=args.hold_days,
            stop_loss=args.stop_loss,
        )
        if result:
            import json
            with open(output_dir / "report.json", "w") as f:
                json.dump({
                    "metrics": {k: float(v) if isinstance(v, (float,)) else v
                               for k, v in result.items() if k not in ["portfolio"]},
                    "params": result["params"],
                }, f, indent=2)
            log.success(f"回测结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
