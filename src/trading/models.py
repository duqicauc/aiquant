# -*- coding: utf-8 -*-
"""交易执行域模型（与券商无关，供回测/模拟盘/实盘共用）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    NEW = "new"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class OrderIntent:
    """策略层意图（未经过风控过滤）。"""
    client_order_id: str
    ts_code: str
    side: OrderSide
    quantity: int
    limit_price: Optional[float] = None
    reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionReport:
    """成交回报（可多条 partial 合并为一条逻辑单）。"""
    client_order_id: str
    ts_code: str
    side: OrderSide
    filled_qty: int
    avg_price: float
    fee: float
    status: OrderStatus
    broker_order_id: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionSnapshot:
    ts_code: str
    quantity: int
    avg_cost: float


@dataclass
class AccountSnapshot:
    cash: float
    total_asset: float
    positions: List[PositionSnapshot] = field(default_factory=list)
