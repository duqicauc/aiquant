# -*- coding: utf-8 -*-
"""券商适配抽象层：实现 BrokerAdapter 即可接入 miniqmt / 其他通道。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.trading.models import AccountSnapshot, ExecutionReport, OrderIntent, PositionSnapshot


class BrokerAdapter(ABC):
    """统一下单、撤单、查询接口（幂等由上层 OrderRouter 保证）。"""

    @abstractmethod
    def submit_order(self, intent: OrderIntent) -> ExecutionReport:
        """提交订单，返回首笔回报（或最终状态）。"""

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> bool:
        """撤单。"""

    @abstractmethod
    def get_positions(self) -> List[PositionSnapshot]:
        """当前持仓。"""

    @abstractmethod
    def get_account(self) -> AccountSnapshot:
        """资金与总资产。"""

    @abstractmethod
    def get_open_orders(self) -> List[ExecutionReport]:
        """未完结委托。"""

    def health_check(self) -> bool:
        """可选：连接/会话健康检查。"""
        return True
