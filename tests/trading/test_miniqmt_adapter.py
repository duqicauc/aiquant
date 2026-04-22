import importlib
import sys
import types

import pytest

import src.trading.miniqmt_adapter as miniqmt_module
from src.trading.models import OrderIntent, OrderSide


def _make_intent() -> OrderIntent:
    return OrderIntent(
        client_order_id="oid-1",
        ts_code="000001.SZ",
        side=OrderSide.BUY,
        quantity=100,
        limit_price=10.0,
    )


def test_methods_raise_runtime_error_without_xtquant():
    adapter = miniqmt_module.MiniQmtAdapter(account_id="acc")
    adapter._xt = None

    with pytest.raises(RuntimeError, match="未安装 xtquant"):
        adapter.submit_order(_make_intent())
    with pytest.raises(RuntimeError, match="未安装 xtquant"):
        adapter.cancel_order("broker-1")
    with pytest.raises(RuntimeError, match="未安装 xtquant"):
        adapter.get_positions()
    with pytest.raises(RuntimeError, match="未安装 xtquant"):
        adapter.get_account()
    with pytest.raises(RuntimeError, match="未安装 xtquant"):
        adapter.get_open_orders()


def test_methods_raise_not_implemented_with_xtquant_stub(monkeypatch):
    monkeypatch.setitem(sys.modules, "xtquant", types.ModuleType("xtquant"))
    importlib.reload(miniqmt_module)
    adapter = miniqmt_module.MiniQmtAdapter(account_id="acc")

    with pytest.raises(NotImplementedError, match="submit_order"):
        adapter.submit_order(_make_intent())
    with pytest.raises(NotImplementedError, match="cancel_order"):
        adapter.cancel_order("broker-1")
    with pytest.raises(NotImplementedError, match="get_positions"):
        adapter.get_positions()
    with pytest.raises(NotImplementedError, match="get_account"):
        adapter.get_account()
    with pytest.raises(NotImplementedError, match="get_open_orders"):
        adapter.get_open_orders()

    monkeypatch.delitem(sys.modules, "xtquant", raising=False)
    importlib.reload(miniqmt_module)
