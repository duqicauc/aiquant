# -*- coding: utf-8 -*-
"""
执行门控：风控熔断后仅允许卖出（不加仓、不开新仓）。
与 RiskEngine 配合，供模拟盘/实盘 OrderRouter 使用。
"""
from __future__ import annotations

from typing import Tuple

from src.trading.models import OrderIntent, OrderSide
from src.trading.risk_engine import RiskEngine


class ExecutionGate:
    def __init__(self, risk: RiskEngine):
        self.risk = risk

    def allow_intent(self, intent: OrderIntent) -> Tuple[bool, str]:
        if intent.side == OrderSide.SELL:
            return True, "ok"
        if not self.risk.allow_new_buy():
            return False, "halt_new_buy"
        return True, "ok"
