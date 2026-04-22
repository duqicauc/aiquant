# -*- coding: utf-8 -*-
"""
订单状态机（与执行层约定一致，便于 miniQMT 回报映射）。

状态: new -> submitted -> partial | filled | cancelled | rejected
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Set


class OrderLifecycleState(str, Enum):
    NEW = "new"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# 合法转移（简化）
_TRANSITIONS: Dict[OrderLifecycleState, Set[OrderLifecycleState]] = {
    OrderLifecycleState.NEW: {OrderLifecycleState.SUBMITTED, OrderLifecycleState.REJECTED},
    OrderLifecycleState.SUBMITTED: {
        OrderLifecycleState.PARTIAL,
        OrderLifecycleState.FILLED,
        OrderLifecycleState.CANCELLED,
        OrderLifecycleState.REJECTED,
    },
    OrderLifecycleState.PARTIAL: {
        OrderLifecycleState.PARTIAL,
        OrderLifecycleState.FILLED,
        OrderLifecycleState.CANCELLED,
    },
    OrderLifecycleState.FILLED: set(),
    OrderLifecycleState.CANCELLED: set(),
    OrderLifecycleState.REJECTED: set(),
}


def can_transition(from_s: OrderLifecycleState, to_s: OrderLifecycleState) -> bool:
    return to_s in _TRANSITIONS.get(from_s, set())
