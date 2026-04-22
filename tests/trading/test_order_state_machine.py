from src.trading.order_state_machine import OrderLifecycleState, can_transition


def test_valid_transitions():
    assert can_transition(OrderLifecycleState.NEW, OrderLifecycleState.SUBMITTED)
    assert can_transition(OrderLifecycleState.SUBMITTED, OrderLifecycleState.FILLED)
    assert can_transition(OrderLifecycleState.PARTIAL, OrderLifecycleState.PARTIAL)


def test_invalid_transitions():
    assert not can_transition(OrderLifecycleState.NEW, OrderLifecycleState.FILLED)
    assert not can_transition(OrderLifecycleState.FILLED, OrderLifecycleState.CANCELLED)
    assert not can_transition(OrderLifecycleState.CANCELLED, OrderLifecycleState.SUBMITTED)
