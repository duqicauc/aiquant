#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 vectorbt 对「策略净值序列」做第二套指标校验（与主回测 numpy 结论对照）。
不重复信号逻辑，仅验证同一 equity curve 下夏普/回撤等是否一致量级。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import backtest_v232_v270_complementary as bt
from src.utils.logger import log


def _numpy_metrics(nav: pd.Series):
    nav = nav.astype(float)
    dd = (nav / nav.cummax() - 1.0) * 100
    max_dd = float(dd.min())
    rets = nav.pct_change().dropna()
    sharpe = float(np.sqrt(252) * rets.mean() / rets.std()) if len(rets) > 1 and rets.std() > 0 else 0.0
    return max_dd, sharpe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    try:
        import vectorbt as vbt
    except ImportError:
        log.error("请安装: pip install vectorbt")
        return 1

    r = bt.backtest_complementary_strategy(
        start_date=args.start_date,
        end_date=args.end_date,
        apply_frictions=True,
        stop_loss_mode="close",
    )
    if not r:
        return 1
    nav = r["daily_records"]["total_assets"].astype(float)
    max_dd_np, sharpe_np = _numpy_metrics(nav)

    # 将净值视为单资产价格，构造 buy-and-hold 组合用 vectorbt 统计
    price = nav / nav.iloc[0]
    entries = pd.Series(False, index=price.index)
    entries.iloc[0] = True
    exits = pd.Series(False, index=price.index)
    close = price.to_frame(name="s")
    entries_df = entries.to_frame(name="s")
    exits_df = exits.to_frame(name="s")

    pf = vbt.Portfolio.from_signals(
        close,
        entries_df,
        exits_df,
        init_cash=1.0,
        fees=0.0,
        freq="d",
    )
    st = pf.stats()
    # vectorbt 版本差异：取可数字段
    sharpe_vbt = float(st.get("Sharpe Ratio", np.nan)) if hasattr(st, "get") else float("nan")
    max_dd_vbt = float(st.get("Max Drawdown [%]", np.nan)) if hasattr(st, "get") else float("nan")

    out = Path(args.output) if args.output else (
        PROJECT_ROOT / "data" / "prediction" / "results" / f"vectorbt_validate_{args.start_date}_{args.end_date}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# vectorbt 净值校验",
        "",
        f"区间: {args.start_date} ~ {args.end_date}",
        "",
        "## numpy（主回测 daily_records）",
        f"- MaxDD%: {max_dd_np:.4f}",
        f"- Sharpe(252): {sharpe_np:.4f}",
        "",
        "## vectorbt Portfolio.from_signals（归一化净值）",
        f"- MaxDD%: {max_dd_vbt}",
        f"- Sharpe: {sharpe_vbt}",
        "",
        "说明：二者应对同一净值曲线给出接近的回撤/夏普；若差异大请检查 vectorbt 版本与 freq。",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    log.success(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
