#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分层归因回测：同区间按「基线→费用→滑点→成交约束→风控门控」逐层叠加，
输出 backtest_attribution_<start>_<end>.csv / .md（累计归因 + 边际归因）。

保留快捷能力：--legacy-binary 仅跑理想 vs 全摩擦二元对比（旧行为）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import backtest_v232_v270_complementary as bt  # noqa: E402
from src.trading.ashare_rules import (
    REASON_LIMIT_DOWN_NO_SELL,
    REASON_LIMIT_UP_NO_BUY,
    REASON_SUSPENDED,
    REASON_VOLUME_INSUFFICIENT,
)
from src.utils.logger import log

RESULTS = PROJECT_ROOT / "data" / "prediction" / "results"

EPS_ATTRIBUTION = 1e-6


def _common_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return dict(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_cash=args.initial_cash,
        stock_amount=args.stock_amount,
        top_n_buy=args.top_buy,
        top_n_hold=args.top_hold,
        use_ma5_sell=not args.no_ma5_sell,
        stop_loss_pct=args.stop_loss_pct,
        stop_loss_mode=args.stop_loss_mode,
        exclude_sectors=args.exclude_sectors,
        buy_slippage_bps=args.buy_slippage_bps,
        sell_slippage_bps=args.sell_slippage_bps,
        commission_rate=args.commission_rate,
        min_commission=args.min_commission,
        transfer_fee_rate=args.transfer_fee_rate,
        stamp_tax_rate=args.stamp_tax_rate,
        max_participation_rate=args.max_participation_rate,
        risk_max_daily_loss_pct=args.risk_max_daily_loss_pct,
        risk_max_drawdown_pct=args.risk_max_drawdown_pct,
    )


def _result_to_attribution_row(
    run_id: str,
    layer: str,
    r: Dict[str, Any],
    baseline_ret: float,
    prev_ret: float | None,
) -> Dict[str, Any]:
    br = r.get("blocked_reasons") or {}
    fin = float(r["final_return_pct"])
    delta_vs_base = fin - baseline_ret
    marginal = None if prev_ret is None else fin - prev_ret
    return {
        "run_id": run_id,
        "layer": layer,
        "final_return_pct": fin,
        "max_drawdown": float(r["max_drawdown"]),
        "sharpe_ratio": float(r["sharpe_ratio"]),
        "total_fees": float(r.get("total_fees", 0)),
        "blocked_limit_up_no_buy": int(br.get(REASON_LIMIT_UP_NO_BUY, 0)),
        "blocked_limit_down_no_sell": int(br.get(REASON_LIMIT_DOWN_NO_SELL, 0)),
        "blocked_suspended": int(br.get(REASON_SUSPENDED, 0)),
        "blocked_volume_insufficient": int(br.get(REASON_VOLUME_INSUFFICIENT, 0)),
        "blocked_halt_new_buy": int(br.get("halt_new_buy", 0)),
        "total_buys": int(r["total_buys"]),
        "total_sells": int(r["total_sells"]),
        "delta_return_vs_baseline_pct": delta_vs_base,
        "marginal_contribution_pct": marginal,
    }


def run_attribution_matrix(common: Dict[str, Any], generate_reports: bool, out: Path) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    五层实验矩阵（显式 enable_*，不依赖 apply_frictions 推断）：
    baseline → cost_only → cost_slippage → execution_constraints → full_stack
    """
    runs: List[Tuple[str, str, Dict[str, Any]]] = [
        (
            "baseline",
            "无成本/滑点/成交约束/风控",
            dict(
                apply_frictions=False,
                enable_fees=False,
                enable_slippage=False,
                enable_execution_constraints=False,
                enable_risk_gate=False,
            ),
        ),
        (
            "cost_only",
            "仅费用",
            dict(
                apply_frictions=False,
                enable_fees=True,
                enable_slippage=False,
                enable_execution_constraints=False,
                enable_risk_gate=False,
            ),
        ),
        (
            "cost_slippage",
            "费用+滑点",
            dict(
                apply_frictions=False,
                enable_fees=True,
                enable_slippage=True,
                enable_execution_constraints=False,
                enable_risk_gate=False,
            ),
        ),
        (
            "execution_constraints",
            "费用+滑点+涨跌停/停牌/量能",
            dict(
                apply_frictions=False,
                enable_fees=True,
                enable_slippage=True,
                enable_execution_constraints=True,
                enable_risk_gate=False,
            ),
        ),
        (
            "full_stack",
            "费用+滑点+成交约束+风控门控",
            dict(
                apply_frictions=False,
                enable_fees=True,
                enable_slippage=True,
                enable_execution_constraints=True,
                enable_risk_gate=True,
            ),
        ),
    ]

    rows: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    baseline_ret = None
    prev_ret = None

    for run_id, layer_desc, extra in runs:
        log.info(f"归因实验: {run_id} ({layer_desc})...")
        kw = {**common, **extra}
        r = bt.backtest_complementary_strategy(**kw)
        if not r:
            log.error(f"回测失败: {run_id}")
            raise RuntimeError(f"backtest failed: {run_id}")
        results.append(r)
        if generate_reports:
            bt.generate_report(r, out)
        fr = float(r["final_return_pct"])
        if baseline_ret is None:
            baseline_ret = fr
        row = _result_to_attribution_row(run_id, layer_desc, r, baseline_ret, prev_ret)
        rows.append(row)
        prev_ret = fr

    df = pd.DataFrame(rows)
    return df, rows


def run_legacy_binary(common: Dict[str, Any], out: Path) -> Tuple[Dict, Dict]:
    log.info("运行理想成交 apply_frictions=False ...")
    ideal = bt.backtest_complementary_strategy(**common, apply_frictions=False)
    if ideal:
        bt.generate_report(ideal, out)
    log.info("运行含摩擦 apply_frictions=True ...")
    real = bt.backtest_complementary_strategy(**common, apply_frictions=True)
    if real:
        bt.generate_report(real, out)
    if not ideal or not real:
        raise RuntimeError("legacy binary 回测失败")
    return ideal, real


def main() -> int:
    p = argparse.ArgumentParser(description="分层归因 vs 理想/摩擦二元对比")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument(
        "--legacy-binary",
        action="store_true",
        help="仅运行理想 vs 全摩擦二元对比（旧版 backtest_friction_compare 输出）",
    )
    p.add_argument(
        "--generate-reports",
        action="store_true",
        help="每层回测后生成完整 Markdown 报告（较慢）",
    )
    p.add_argument("--initial-cash", type=float, default=10_000_000.0)
    p.add_argument("--stock-amount", type=float, default=300_000.0)
    p.add_argument("--top-buy", type=int, default=10)
    p.add_argument("--top-hold", type=int, default=50)
    p.add_argument("--no-ma5-sell", action="store_true")
    p.add_argument("--stop-loss-pct", type=float, default=4.0)
    p.add_argument("--stop-loss-mode", type=str, default="close", choices=["none", "close", "intraday_low"])
    p.add_argument("--exclude-sectors", action="store_true")
    p.add_argument("--buy-slippage-bps", type=float, default=15.0)
    p.add_argument("--sell-slippage-bps", type=float, default=20.0)
    p.add_argument("--commission-rate", type=float, default=0.0003)
    p.add_argument("--min-commission", type=float, default=5.0)
    p.add_argument("--transfer-fee-rate", type=float, default=0.00001)
    p.add_argument("--stamp-tax-rate", type=float, default=0.001)
    p.add_argument("--max-participation-rate", type=float, default=0.05)
    p.add_argument("--risk-max-daily-loss-pct", type=float, default=3.0)
    p.add_argument("--risk-max-drawdown-pct", type=float, default=12.0)
    args = p.parse_args()

    out = Path(args.output_dir) if args.output_dir else RESULTS
    out.mkdir(parents=True, exist_ok=True)
    common = _common_kwargs(args)

    if args.legacy_binary:
        ideal, real = run_legacy_binary(common, out)
        rows_legacy = [
            {
                "mode": "ideal",
                "final_return_pct": ideal["final_return_pct"],
                "max_drawdown": ideal["max_drawdown"],
                "sharpe_ratio": ideal["sharpe_ratio"],
                "total_fees": ideal.get("total_fees", 0),
                "total_buys": ideal["total_buys"],
                "total_sells": ideal["total_sells"],
            },
            {
                "mode": "realistic",
                "final_return_pct": real["final_return_pct"],
                "max_drawdown": real["max_drawdown"],
                "sharpe_ratio": real["sharpe_ratio"],
                "total_fees": real.get("total_fees", 0),
                "total_buys": real["total_buys"],
                "total_sells": real["total_sells"],
            },
        ]
        df = pd.DataFrame(rows_legacy)
        csv_path = out / f"backtest_friction_compare_{args.start_date}_{args.end_date}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        decay = None
        if ideal["final_return_pct"] != 0:
            decay = (ideal["final_return_pct"] - real["final_return_pct"]) / abs(ideal["final_return_pct"]) * 100
        md_path = out / f"backtest_friction_compare_{args.start_date}_{args.end_date}.md"
        lines = [
            "# 理想成交 vs 含摩擦回测对比",
            "",
            f"区间: {args.start_date} ~ {args.end_date}",
            "",
            "| 模式 | 收益率% | 最大回撤% | 夏普 | 费用合计(元) | 买笔 | 卖笔 |",
            "|------|---------|-----------|------|----------------|------|------|",
        ]
        for r in rows_legacy:
            lines.append(
                f"| {r['mode']} | {r['final_return_pct']:+.2f} | {r['max_drawdown']:.2f} | "
                f"{r['sharpe_ratio']:.2f} | {r['total_fees']:,.0f} | {r['total_buys']} | {r['total_sells']} |"
            )
        lines.extend(
            [
                "",
                f"收益衰减（相对理想）: {decay:.1f}%" if decay is not None else "",
                "",
                f"详细 CSV: `{csv_path.name}`",
            ]
        )
        md_path.write_text("\n".join(lines), encoding="utf-8")
        log.success(f"已写入: {md_path} / {csv_path}")
        return 0

    df_attr, row_dicts = run_attribution_matrix(common, args.generate_reports, out)
    csv_attr = out / f"backtest_attribution_{args.start_date}_{args.end_date}.csv"
    df_attr.to_csv(csv_attr, index=False, encoding="utf-8-sig")

    baseline_ret = float(row_dicts[0]["final_return_pct"])
    full_ret = float(row_dicts[-1]["final_return_pct"])
    sum_marginal = 0.0
    for i, rd in enumerate(row_dicts):
        m = rd.get("marginal_contribution_pct")
        if m is not None and not (isinstance(m, float) and pd.isna(m)):
            sum_marginal += float(m)
    gap = abs((full_ret - baseline_ret) - sum_marginal)
    consistent = gap <= EPS_ATTRIBUTION

    md_attr = out / f"backtest_attribution_{args.start_date}_{args.end_date}.md"
    lines_md = [
        "# 收益下滑分层归因",
        "",
        f"区间: {args.start_date} ~ {args.end_date}",
        "",
        "## 累计归因（相对 baseline 收益率变化，百分点）",
        "",
        "| run_id | 说明 | 收益率% | 相对 baseline Δ% |",
        "|--------|------|---------|------------------|",
    ]
    for rd in row_dicts:
        lines_md.append(
            f"| {rd['run_id']} | {rd['layer']} | {rd['final_return_pct']:+.4f} | "
            f"{rd['delta_return_vs_baseline_pct']:+.4f} |"
        )
    lines_md.extend(
        [
            "",
            "## 边际归因（相对上一层叠加后的收益率变化，百分点）",
            "",
            "| run_id | 说明 | 边际 Δ% |",
            "|--------|------|---------|",
        ]
    )
    for rd in row_dicts:
        marg = rd["marginal_contribution_pct"]
        marg_s = "—" if marg is None or (isinstance(marg, float) and pd.isna(marg)) else f"{float(marg):+.4f}"
        lines_md.append(f"| {rd['run_id']} | {rd['layer']} | {marg_s} |")
    lines_md.extend(
        [
            "",
            "## 一致性校验",
            "",
            f"- baseline 收益率%: {baseline_ret:+.6f}",
            f"- full_stack 收益率%: {full_ret:+.6f}",
            f"- 全区间总变化 (full - baseline): {full_ret - baseline_ret:+.6f}",
            f"- 边际之和 Σ(边际): {sum_marginal:+.6f}",
            f"- 差额 |gap|: {gap:.2e} （阈值 {EPS_ATTRIBUTION:g}）→ **{'通过' if consistent else '未通过'}**",
            "",
            "## 明细指标与阻塞统计",
            "",
            "| run_id | 收益率% | 最大回撤% | 夏普 | 费用(元) | limit_up | limit_down | suspended | vol_block | halt_buy | 买 | 卖 |",
            "|--------|---------|-----------|------|----------|----------|------------|-----------|-----------|----------|----|----|",
        ]
    )
    for rd in row_dicts:
        lines_md.append(
            f"| {rd['run_id']} | {rd['final_return_pct']:+.4f} | {rd['max_drawdown']:.4f} | {rd['sharpe_ratio']:.4f} | "
            f"{rd['total_fees']:,.0f} | {rd['blocked_limit_up_no_buy']} | {rd['blocked_limit_down_no_sell']} | "
            f"{rd['blocked_suspended']} | {rd['blocked_volume_insufficient']} | {rd['blocked_halt_new_buy']} | "
            f"{rd['total_buys']} | {rd['total_sells']} |"
        )
    lines_md.extend(["", f"CSV: `{csv_attr.name}`", ""])
    md_attr.write_text("\n".join(lines_md), encoding="utf-8")
    log.success(f"已写入归因: {md_attr} / {csv_attr}")
    if not consistent:
        log.warning(f"边际之和与总差不一致 (gap={gap})，请检查浮点或回测逻辑")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
