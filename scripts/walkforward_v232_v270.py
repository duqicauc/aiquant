#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
滚动窗口 walk-forward：对 v232_v270 互补回测按日历窗切片，汇总每窗收益、回撤、夏普等。
用于模拟盘前稳健性评估（不偷看未来：每窗独立为历史区间回测）。
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import backtest_v232_v270_complementary as bt
from src.utils.logger import log

RESULTS = PROJECT_ROOT / "data" / "prediction" / "results"
THRESHOLDS = PROJECT_ROOT / "config" / "walkforward_thresholds.json"


def _parse(d: str) -> datetime:
    return datetime.strptime(d, "%Y%m%d")


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def generate_windows(start: str, end: str, train_days: int, step_days: int):
    """生成 [train_start, train_end] 窗口，train_end 每次前进 step_days。"""
    s, e = _parse(start), _parse(end)
    windows = []
    cur = s
    while cur <= e:
        tr_end = cur + timedelta(days=train_days)
        if tr_end > e:
            break
        windows.append((_fmt(cur), _fmt(tr_end)))
        cur += timedelta(days=step_days)
    return windows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--train-days", type=int, default=60, help="每窗长度（日历天近似）")
    p.add_argument("--step-days", type=int, default=20, help="窗口滑动步长（天）")
    p.add_argument("--stop-loss-mode", default="close")
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    out = Path(args.output_dir) if args.output_dir else RESULTS
    out.mkdir(parents=True, exist_ok=True)

    wins = generate_windows(args.start_date, args.end_date, args.train_days, args.step_days)
    if not wins:
        log.error("无有效窗口，请扩大区间或减小 train-days")
        return 1

    rows = []
    for ws, we in wins:
        log.info(f"窗口 {ws} ~ {we}")
        r = bt.backtest_complementary_strategy(
            start_date=ws,
            end_date=we,
            stop_loss_mode=args.stop_loss_mode,
            apply_frictions=True,
        )
        if not r:
            rows.append({"start": ws, "end": we, "ok": False})
            continue
        rows.append(
            {
                "start": ws,
                "end": we,
                "ok": True,
                "final_return_pct": r["final_return_pct"],
                "max_drawdown": r["max_drawdown"],
                "sharpe_ratio": r["sharpe_ratio"],
                "win_rate": r["win_rate"],
                "total_fees": r.get("total_fees", 0),
                "n_days": r.get("n_trading_days", 0),
            }
        )

    df = pd.DataFrame(rows)
    thr = {}
    if THRESHOLDS.exists():
        try:
            thr = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"读取阈值配置失败: {e}")

    max_neg_ratio = float(thr.get("max_negative_window_ratio", 0.3))
    min_sharpe = float(thr.get("min_sharpe", 0.0))
    max_dd_limit = float(thr.get("max_drawdown_pct", -15.0))

    def _pass_row(row):
        if not row.get("ok", False):
            return False
        if row.get("sharpe_ratio", 0) < min_sharpe:
            return False
        if row.get("max_drawdown", 0) < max_dd_limit:
            return False
        return True

    if len(df) > 0:
        df["threshold_pass"] = df.apply(_pass_row, axis=1)

    path = out / f"walkforward_summary_{args.start_date}_{args.end_date}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    ok_df = df[df["ok"] == True] if "ok" in df.columns else df  # noqa: E712
    fail_ratio = 1.0 - len(ok_df) / len(df) if len(df) else 0.0
    if len(ok_df) > 0 and "final_return_pct" in ok_df.columns:
        neg = (ok_df["final_return_pct"] < 0).mean()
    else:
        neg = 0.0
    pass_ratio = (
        float(df["threshold_pass"].mean())
        if not df.empty and "threshold_pass" in df.columns and df["threshold_pass"].notna().any()
        else 0.0
    )
    go = bool(not df.empty) and pass_ratio >= (1.0 - max_neg_ratio) and neg <= max_neg_ratio
    summary = out / f"walkforward_go_nogo_{args.start_date}_{args.end_date}.md"
    summary.write_text(
        "\n".join(
            [
                "# Walk-forward Go/No-Go 摘要",
                "",
                f"- 窗口数: {len(df)}",
                f"- 失败窗占比: {fail_ratio:.1%}",
                f"- 收益为负窗占比: {neg:.1%}",
                f"- 阈值内通过窗占比: {pass_ratio:.1%}",
                f"- 建议 Go: **{go}**（需结合 `config/walkforward_thresholds.json` 人工复核）",
                "",
                f"明细: `{path.name}`",
            ]
        ),
        encoding="utf-8",
    )
    log.success(f"已写入 {path}；失败窗占比 {fail_ratio:.1%}；收益为负窗占比 {neg:.1%}；Go={go}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
