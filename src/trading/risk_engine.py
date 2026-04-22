# -*- coding: utf-8 -*-
"""盘中风控：单日亏损、组合回撤熔断 → 仅减仓不加仓。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class RiskConfig:
    max_daily_loss_pct: float = 3.0  # 单日净值最大跌幅 %
    max_drawdown_pct: float = 12.0  # 组合相对高点最大回撤 %
    min_cash_ratio: float = 0.05  # 最低现金比例


class RiskEngine:
    def __init__(self, cfg: Optional[RiskConfig] = None):
        self.cfg = cfg or RiskConfig()
        self._halt_new_buy = False
        self._peak_nav = 0.0
        self._last_nav = 0.0

    def on_day_start(self, nav: float) -> None:
        self._last_nav = nav
        self._peak_nav = max(self._peak_nav, nav)

    def on_day_end(self, nav: float) -> Tuple[bool, str]:
        """
        更新风控状态。返回 (是否禁止开仓, 说明)。
        """
        if self._last_nav > 0:
            day_ret = (nav - self._last_nav) / self._last_nav * 100
            if day_ret <= -self.cfg.max_daily_loss_pct:
                self._halt_new_buy = True
                return True, f"触发单日亏损熔断(>{self.cfg.max_daily_loss_pct}%)"
        self._peak_nav = max(self._peak_nav, nav)
        if self._peak_nav > 0:
            dd = (nav - self._peak_nav) / self._peak_nav * 100
            if dd <= -self.cfg.max_drawdown_pct:
                self._halt_new_buy = True
                return True, f"触发回撤熔断(>{self.cfg.max_drawdown_pct}%)"
        self._last_nav = nav
        return self._halt_new_buy, "ok" if not self._halt_new_buy else "维持仅减仓"

    def allow_new_buy(self) -> bool:
        return not self._halt_new_buy

    def reset_halt(self) -> None:
        self._halt_new_buy = False
