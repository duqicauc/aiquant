from src.trading.risk_engine import RiskConfig, RiskEngine


def test_on_day_end_no_halt_when_within_threshold():
    engine = RiskEngine(RiskConfig(max_daily_loss_pct=3.0, max_drawdown_pct=12.0))
    engine.on_day_start(100.0)
    halt, reason = engine.on_day_end(98.0)

    assert halt is False
    assert reason == "ok"
    assert engine.allow_new_buy() is True


def test_on_day_end_halt_by_daily_loss():
    engine = RiskEngine(RiskConfig(max_daily_loss_pct=3.0, max_drawdown_pct=50.0))
    engine.on_day_start(100.0)
    halt, reason = engine.on_day_end(96.8)

    assert halt is True
    assert "单日亏损熔断" in reason
    assert engine.allow_new_buy() is False


def test_on_day_end_halt_on_daily_loss_exact_threshold():
    engine = RiskEngine(RiskConfig(max_daily_loss_pct=3.0, max_drawdown_pct=50.0))
    engine.on_day_start(100.0)
    halt, reason = engine.on_day_end(97.0)

    assert halt is True
    assert "单日亏损熔断" in reason
    assert engine.allow_new_buy() is False


def test_on_day_end_halt_by_drawdown():
    engine = RiskEngine(RiskConfig(max_daily_loss_pct=50.0, max_drawdown_pct=10.0))
    engine.on_day_start(100.0)
    engine.on_day_end(120.0)

    engine.on_day_start(119.0)
    halt, reason = engine.on_day_end(100.0)

    assert halt is True
    assert "回撤熔断" in reason
    assert engine.allow_new_buy() is False


def test_on_day_end_halt_on_drawdown_exact_threshold():
    engine = RiskEngine(RiskConfig(max_daily_loss_pct=50.0, max_drawdown_pct=10.0))
    engine.on_day_start(100.0)
    engine.on_day_end(120.0)

    engine.on_day_start(119.0)
    halt, reason = engine.on_day_end(108.0)

    assert halt is True
    assert "回撤熔断" in reason
    assert engine.allow_new_buy() is False


def test_reset_halt_restores_buy_permission():
    engine = RiskEngine(RiskConfig(max_daily_loss_pct=3.0, max_drawdown_pct=12.0))
    engine.on_day_start(100.0)
    engine.on_day_end(96.0)
    assert engine.allow_new_buy() is False

    engine.reset_halt()
    assert engine.allow_new_buy() is True


def test_on_day_end_keep_halt_status_after_triggered():
    engine = RiskEngine(RiskConfig(max_daily_loss_pct=3.0, max_drawdown_pct=12.0))
    engine.on_day_start(100.0)
    engine.on_day_end(96.0)
    assert engine.allow_new_buy() is False

    # 次日未再次触发阈值，也应维持仅减仓状态，直到显式 reset
    engine.on_day_start(96.0)
    halt, reason = engine.on_day_end(96.2)
    assert halt is True
    assert reason == "维持仅减仓"
