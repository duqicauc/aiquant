from src.trading.reconcile import format_reconcile_report, reconcile_positions


def test_reconcile_positions_all_match():
    res = reconcile_positions(
        broker_qty={"000001.SZ": 100},
        local_qty={"000001.SZ": 100},
        strategy_qty={"000001.SZ": 100},
    )

    assert res.ok is True
    assert res.diffs == []
    report = format_reconcile_report(res)
    assert "状态: 一致" in report


def test_reconcile_positions_with_diffs():
    res = reconcile_positions(
        broker_qty={"000001.SZ": 100, "600000.SH": 0},
        local_qty={"000001.SZ": 90},
        strategy_qty={"000001.SZ": 100, "600000.SH": 100},
    )

    assert res.ok is False
    assert len(res.diffs) == 2
    assert "000001.SZ: broker=100 local=90 strategy=100" in res.diffs
    assert "600000.SH: broker=0 local=0 strategy=100" in res.diffs

    report = format_reconcile_report(res)
    assert "状态: 不一致" in report
    assert "差异:" in report
