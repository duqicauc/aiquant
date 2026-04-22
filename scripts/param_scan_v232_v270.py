#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
L1 参数扫描：对 top_buy / stop_loss_mode 等做网格，输出汇总 CSV。
"""
import argparse
import sys
from itertools import product
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import backtest_v232_v270_complementary as bt
from src.utils.logger import log

OUT = PROJECT_ROOT / "data" / "prediction" / "results"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    args = p.parse_args()

    modes = ["none", "close", "intraday_low"]
    tops = [8, 10, 12]
    rows = []
    for mode, top in product(modes, tops):
        log.info(f"scan mode={mode} top_buy={top}")
        r = bt.backtest_complementary_strategy(
            start_date=args.start_date,
            end_date=args.end_date,
            top_n_buy=top,
            stop_loss_mode=mode,
            apply_frictions=True,
        )
        if not r:
            continue
        rows.append(
            {
                "stop_loss_mode": mode,
                "top_n_buy": top,
                "final_return_pct": r["final_return_pct"],
                "max_drawdown": r["max_drawdown"],
                "sharpe_ratio": r["sharpe_ratio"],
                "total_fees": r.get("total_fees", 0),
            }
        )
    df = pd.DataFrame(rows)
    path = OUT / f"param_scan_{args.start_date}_{args.end_date}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log.success(f"已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
