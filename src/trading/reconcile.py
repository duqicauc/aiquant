# -*- coding: utf-8 -*-
"""收盘对账：券商持仓 vs 本地账本 vs 策略预期（数量级差异）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ReconcileResult:
    ok: bool
    diffs: List[str]
    broker_qty: Dict[str, int]
    local_qty: Dict[str, int]
    strategy_qty: Dict[str, int]


def reconcile_positions(
    broker_qty: Dict[str, int],
    local_qty: Dict[str, int],
    strategy_qty: Dict[str, int],
) -> ReconcileResult:
    """
    比较三方持仓股数。键均为 ts_code。
    """
    diffs = []
    all_codes = set(broker_qty) | set(local_qty) | set(strategy_qty)
    for c in sorted(all_codes):
        b = broker_qty.get(c, 0)
        l = local_qty.get(c, 0)
        s = strategy_qty.get(c, 0)
        if b != l or l != s:
            diffs.append(f"{c}: broker={b} local={l} strategy={s}")
    return ReconcileResult(
        ok=len(diffs) == 0,
        diffs=diffs,
        broker_qty=dict(broker_qty),
        local_qty=dict(local_qty),
        strategy_qty=dict(strategy_qty),
    )


def format_reconcile_report(res: ReconcileResult) -> str:
    lines = ["=== 收盘对账 ===", f"状态: {'一致' if res.ok else '不一致'}"]
    if res.diffs:
        lines.append("差异:")
        lines.extend("  " + d for d in res.diffs)
    return "\n".join(lines)
