from src.trading.execution_gate import ExecutionGate
from src.trading.models import OrderIntent, OrderSide
from src.trading.risk_engine import RiskConfig, RiskEngine


def _make_intent(side: OrderSide) -> OrderIntent:
    return OrderIntent(
        client_order_id="oid-1",
        ts_code="000001.SZ",
        side=side,
        quantity=100,
        limit_price=10.0,
    )


def test_allow_buy_and_sell_when_risk_not_triggered():
    engine = RiskEngine(RiskConfig(max_daily_loss_pct=3.0, max_drawdown_pct=12.0))
    gate = ExecutionGate(engine)

    allowed_buy, reason_buy = gate.allow_intent(_make_intent(OrderSide.BUY))
    allowed_sell, reason_sell = gate.allow_intent(_make_intent(OrderSide.SELL))

    assert allowed_buy is True and reason_buy == "ok"
    assert allowed_sell is True and reason_sell == "ok"


def test_block_buy_but_allow_sell_after_risk_halt():
    engine = RiskEngine(RiskConfig(max_daily_loss_pct=3.0, max_drawdown_pct=12.0))
    engine.on_day_start(100.0)
    engine.on_day_end(96.0)
    gate = ExecutionGate(engine)

    allowed_buy, reason_buy = gate.allow_intent(_make_intent(OrderSide.BUY))
    allowed_sell, reason_sell = gate.allow_intent(_make_intent(OrderSide.SELL))

    assert allowed_buy is False
    assert reason_buy == "halt_new_buy"
    assert allowed_sell is True
    assert reason_sell == "ok"
