#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MiniQMT 模拟盘执行记录：将「计划价/委托价/成交价/偏差/未成交原因」追加写入 JSONL，
便于 2～4 周跑完后做漏成交归因（不依赖 xtquant 是否已安装）。

实盘/模拟盘接入 MiniQmtAdapter 后，由执行层调用 `log_paper_execution`。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_LOG = PROJECT_ROOT / "data" / "prediction" / "paper_trading" / "executions.jsonl"


def log_paper_execution(
    row: Dict[str, Any],
    path: Optional[Path] = None,
) -> None:
    """追加一行 JSONL。"""
    p = path or DEFAULT_LOG
    p.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description="写入一条模拟盘执行样例（测试日志管道）")
    p.add_argument("--ts-code", default="000001.SZ")
    p.add_argument("--signal-price", type=float, default=10.0)
    p.add_argument("--fill-price", type=float, default=10.02)
    p.add_argument("--reason", default="ok")
    args = p.parse_args()
    dev = abs(args.fill_price - args.signal_price) / args.signal_price * 10000
    log_paper_execution(
        {
            "ts_code": args.ts_code,
            "side": "buy",
            "signal_price": args.signal_price,
            "order_price": args.signal_price,
            "fill_price": args.fill_price,
            "deviation_bps": dev,
            "status": "filled",
            "block_reason": None,
            "note": args.reason,
        }
    )
    print(f"appended to {DEFAULT_LOG}")


if __name__ == "__main__":
    main()
