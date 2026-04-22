#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模拟盘 / 实盘成交日志：追加 JSON Lines，便于统计信号价 vs 成交价偏差与漏成交归因。

用法示例：
  python scripts/paper_trade_journal.py append --ts_code 600000.SH --side buy \\
    --signal_price 10.2 --fill_price 10.25 --qty 300 --reason limit_up_partial
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_LOG = PROJECT_ROOT / "data" / "prediction" / "trading_plans" / "paper_fills.jsonl"


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("append", help="追加一条成交记录")
    a.add_argument("--ts_code", required=True)
    a.add_argument("--side", choices=["buy", "sell"], required=True)
    a.add_argument("--signal_price", type=float, required=True)
    a.add_argument("--fill_price", type=float, required=True)
    a.add_argument("--qty", type=int, required=True)
    a.add_argument("--reason", type=str, default="")
    a.add_argument("--log-file", type=str, default=None)

    args = p.parse_args()
    log_path = Path(args.log_file) if args.log_file else DEFAULT_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)

    slip_bps = (args.fill_price - args.signal_price) / args.signal_price * 10000.0
    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "ts_code": args.ts_code,
        "side": args.side,
        "signal_price": args.signal_price,
        "fill_price": args.fill_price,
        "qty": args.qty,
        "slippage_bps": round(slip_bps, 2),
        "reason": args.reason,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
