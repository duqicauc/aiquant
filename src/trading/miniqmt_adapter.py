# -*- coding: utf-8 -*-
"""
MiniQMT / xtquant 适配器（桩实现 + 可扩展）。

实盘环境需在 Windows 安装 QMT 客户端，并在 Python 中安装 `xtquant`。
本仓库默认不强制依赖 xtquant；调用时若未安装则抛出明确错误。

价格与费用口径应与 `src.trading.ashare_rules` 一致（最小价位、涨跌停带等）。
"""
from __future__ import annotations

from typing import List

from src.trading.broker_base import BrokerAdapter
from src.trading.models import AccountSnapshot, ExecutionReport, OrderIntent, PositionSnapshot


class MiniQmtAdapter(BrokerAdapter):
    """
    连接 miniQMT 的 BrokerAdapter。

    使用步骤（部署时）：
    1. 启动 QMT / miniQMT 并登录资金账号
    2. `pip install xtquant`（由券商提供）
    3. 在环境变量或配置中设置 session / account_id
    4. 实现下方 TODO 中的 xttrader 调用
    """

    def __init__(self, account_id: str = "", session=None):
        self.account_id = account_id
        self._session = session
        self._xt = None
        try:
            import xtquant  # noqa: F401

            self._xt = xtquant
        except ImportError:
            self._xt = None

    def _require_xt(self):
        if self._xt is None:
            raise RuntimeError("未安装 xtquant：请在 QMT 环境执行 pip install xtquant，并启动客户端后再调用。")

    def submit_order(self, intent: OrderIntent) -> ExecutionReport:
        self._require_xt()
        # TODO: 调用 xttrader 下单 API，填充成交回报
        raise NotImplementedError("MiniQmtAdapter.submit_order 需在实盘环境接 xtquant 接口")

    def cancel_order(self, broker_order_id: str) -> bool:
        self._require_xt()
        raise NotImplementedError("MiniQmtAdapter.cancel_order 待实现")

    def get_positions(self) -> List[PositionSnapshot]:
        self._require_xt()
        raise NotImplementedError("MiniQmtAdapter.get_positions 待实现")

    def get_account(self) -> AccountSnapshot:
        self._require_xt()
        raise NotImplementedError("MiniQmtAdapter.get_account 待实现")

    def get_open_orders(self) -> List[ExecutionReport]:
        self._require_xt()
        raise NotImplementedError("MiniQmtAdapter.get_open_orders 待实现")
