import math

import pytest

from src.trading.ashare_rules import (
    LIMIT_PCT_BJ,
    LIMIT_PCT_CHINEXT_STAR,
    LIMIT_PCT_MAINBOARD,
    LIMIT_PCT_ST,
    calc_trade_fee,
    infer_limit_pct,
    is_limit_down,
    is_limit_up,
    is_open_limit_up,
    is_suspended_or_no_trade,
    participation_ok,
    round_price_a_share,
)


def test_infer_limit_pct_by_code_and_name():
    assert infer_limit_pct("600000.SH", "平安银行") == LIMIT_PCT_MAINBOARD
    assert infer_limit_pct("000001.SZ", "ST测试") == LIMIT_PCT_ST
    assert infer_limit_pct("300750.SZ", "") == LIMIT_PCT_CHINEXT_STAR
    assert infer_limit_pct("688001.SH", "") == LIMIT_PCT_CHINEXT_STAR
    assert infer_limit_pct("430001.BJ", "") == LIMIT_PCT_BJ


def test_round_price_a_share():
    assert round_price_a_share(10.126) == 10.13
    assert round_price_a_share(-1.0) == -1.0
    assert math.isnan(round_price_a_share(float("nan")))


def test_limit_up_and_down_judgement():
    snap_up = {"pre_close": 10.0, "close": 11.0}
    snap_down = {"pre_close": 10.0, "close": 9.0}
    assert is_limit_up(snap_up, 10.0)
    assert is_limit_down(snap_down, 10.0)


@pytest.mark.parametrize(
    "close_px,expected",
    [
        (10.9951, True),  # 略高于阈值，规避浮点边界抖动
        (10.994, False),  # 低于阈值
    ],
)
def test_limit_up_epsilon_boundary(close_px, expected):
    snap = {"pre_close": 10.0, "close": close_px}
    assert is_limit_up(snap, 10.0, eps=0.0005) is expected


@pytest.mark.parametrize(
    "close_px,expected",
    [
        (9.0049, True),  # 略低于阈值，规避浮点边界抖动
        (9.006, False),  # 高于阈值
    ],
)
def test_limit_down_epsilon_boundary(close_px, expected):
    snap = {"pre_close": 10.0, "close": close_px}
    assert is_limit_down(snap, 10.0, eps=0.0005) is expected


def test_limit_judgement_with_invalid_input():
    invalid_snap = {"pre_close": 0.0, "close": 10.0}
    nan_snap = {"pre_close": float("nan"), "close": 10.0}
    assert not is_limit_up(invalid_snap, 10.0)
    assert not is_limit_down(invalid_snap, 10.0)
    assert not is_limit_up(nan_snap, 10.0)
    assert not is_limit_down(nan_snap, 10.0)


def test_open_limit_up():
    assert is_open_limit_up(11.0, 10.0, 10.0)
    assert not is_open_limit_up(10.5, 10.0, 10.0)
    assert not is_open_limit_up(float("nan"), 10.0, 10.0)


@pytest.mark.parametrize(
    "snapshot,expected",
    [
        (None, True),
        ({"vol": 0}, True),
        ({"vol": -1}, True),
        ({}, True),
        ({"vol": "bad_value"}, True),
        ({"vol": 1000}, False),
    ],
)
def test_is_suspended_or_no_trade(snapshot, expected):
    assert is_suspended_or_no_trade(snapshot) is expected


def test_calc_trade_fee_for_buy_and_sell():
    amount = 10000
    commission_rate = 0.0003
    min_commission = 5.0
    transfer_fee_rate = 0.00001
    stamp_tax_rate = 0.001

    buy_fee = calc_trade_fee(
        amount=amount,
        is_sell=False,
        commission_rate=commission_rate,
        min_commission=min_commission,
        transfer_fee_rate=transfer_fee_rate,
        stamp_tax_rate=stamp_tax_rate,
    )
    sell_fee = calc_trade_fee(
        amount=amount,
        is_sell=True,
        commission_rate=commission_rate,
        min_commission=min_commission,
        transfer_fee_rate=transfer_fee_rate,
        stamp_tax_rate=stamp_tax_rate,
    )

    assert buy_fee == pytest.approx(5.1)
    assert sell_fee == pytest.approx(15.1)


def test_participation_ok():
    ok, part = participation_ok(
        order_amount_yuan=10000,
        daily_amount_thousand_yuan=2000,
        max_participation_rate=0.01,
    )
    assert ok
    assert part == pytest.approx(0.005)


def test_participation_not_ok_when_over_threshold():
    ok, part = participation_ok(
        order_amount_yuan=30000,
        daily_amount_thousand_yuan=2000,
        max_participation_rate=0.01,
    )
    assert ok is False
    assert part == pytest.approx(0.015)


def test_participation_ok_with_zero_turnover_fallback():
    ok, part = participation_ok(
        order_amount_yuan=10000,
        daily_amount_thousand_yuan=0,
        max_participation_rate=0.01,
    )
    assert ok
    assert part == 0.0
