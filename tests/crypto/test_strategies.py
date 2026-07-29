from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from quant_rabbit.crypto.fast import FastMarketState
from quant_rabbit.crypto.strategies import (
    load_strategy_profiles,
    strategy_router,
)


CONFIG = Path("config/crypto_strategy_lab_v1.json")
QUEUE_FLOW_CONFIG = Path(
    "config/crypto_bitbank_queue_flow_paper_v1.json"
)


def _state(
    prices: list[str],
    *,
    bid_amount: str = "100",
    ask_amount: str = "1",
) -> FastMarketState:
    state = FastMarketState("btc_jpy", 32)
    assert state.book.apply_whole(
        {
            "bids": [["100", bid_amount]],
            "asks": [["100.001", ask_amount]],
            "timestamp": 1_000,
            "sequenceId": "1",
        }
    )
    state.event_count = 20
    for price in prices:
        state.observe_price(Decimal(price))
    return state


def _decide(
    name: str,
    state: FastMarketState,
    *,
    allow_short: bool = True,
    maker_fee: str = "-0.0002",
    taker_fee: str = "0",
) -> dict[str, object]:
    return strategy_router(
        name,
        config_path=CONFIG,
        warmup_events=1,
        max_data_age_ms=5_000,
    ).decide(
        state,
        position=Decimal("0"),
        average_cost=Decimal("0"),
        maker_fee_rate=Decimal(maker_fee),
        taker_fee_rate=Decimal(taker_fee),
        allow_short=allow_short,
        now_ns=1_000_000_000_000,
        wall_time_ms=1_001,
    )


def test_all_strategy_profiles_are_externalized_and_safe() -> None:
    profiles = load_strategy_profiles(CONFIG)
    assert set(profiles) == {
        "RANGE_MAKER_REVERSION",
        "BREAKOUT_CONFIRMATION",
        "TREND_PULLBACK_MAKER",
        "ORDER_BOOK_FADE",
        "ORDER_BOOK_FADE_COOLDOWN_5S",
        "ORDER_BOOK_FADE_MAKER_EXIT",
    }
    assert all(
        profile.entry_order_style.startswith("PAPER_")
        and profile.exit_order_style.startswith("PAPER_")
        for profile in profiles.values()
    )
    baseline = profiles["ORDER_BOOK_FADE"]
    variant = profiles["ORDER_BOOK_FADE_COOLDOWN_5S"]
    ignored = {"name", "variant_of", "changed_category", "cooldown_ms"}
    for field in baseline.__dataclass_fields__:
        if field not in ignored:
            assert getattr(variant, field) == getattr(baseline, field)
    assert variant.variant_of == baseline.name
    assert variant.changed_category == "cooldown_ms"
    assert variant.cooldown_ms == 5000
    maker_exit = profiles["ORDER_BOOK_FADE_MAKER_EXIT"]
    ignored = {
        "name",
        "variant_of",
        "changed_category",
        "forced_exit_order_style",
    }
    for field in baseline.__dataclass_fields__:
        if field not in ignored:
            assert getattr(maker_exit, field) == getattr(baseline, field)
    assert maker_exit.variant_of == baseline.name
    assert maker_exit.changed_category == "forced_exit_order_style"
    assert maker_exit.forced_exit_order_style == "PAPER_MAKER_LIMIT"


@pytest.mark.parametrize(
    ("name", "state", "expected_side"),
    [
        (
            "RANGE_MAKER_REVERSION",
            _state(["100", "99.998", "99.996", "99.994"]),
            "LONG",
        ),
        (
            "BREAKOUT_CONFIRMATION",
            _state(["100", "100.004", "100.008", "100.012"]),
            "LONG",
        ),
        (
            "TREND_PULLBACK_MAKER",
            _state(
                ["100", "100.004", "100.008", "100.016", "100.014"]
            ),
            "LONG",
        ),
        (
            "ORDER_BOOK_FADE",
            _state(["100", "100.001", "100.002", "100.003"]),
            "SHORT",
        ),
    ],
)
def test_strategy_family_can_emit_paper_entry(
    name: str,
    state: FastMarketState,
    expected_side: str,
) -> None:
    decision = _decide(name, state)
    assert decision["action"] == "ENTER"
    assert decision["position_side"] == expected_side
    assert decision["authority"] == "NONE"
    assert decision["live_permission"] is False
    assert decision["no_future_data"] is True


def test_breakout_remains_blocked_when_taker_cost_exceeds_edge() -> None:
    decision = _decide(
        "BREAKOUT_CONFIRMATION",
        _state(["100", "100.004", "100.008", "100.012"]),
        taker_fee="0.0012",
    )
    assert decision["action"] == "WAIT"
    assert decision["reason"] == "NET_EDGE_BELOW_BUFFER"
    assert Decimal(str(decision["net_edge_bps"])) < 0


def test_spot_variant_cannot_open_short() -> None:
    decision = _decide(
        "ORDER_BOOK_FADE",
        _state(["100", "100.001", "100.002", "100.003"]),
        allow_short=False,
    )
    assert decision["action"] == "WAIT"
    assert decision["reason"] == "SHORT_DISABLED"


def test_variant_fails_closed_on_future_stream_data() -> None:
    decision = strategy_router(
        "RANGE_MAKER_REVERSION",
        config_path=CONFIG,
        warmup_events=1,
        max_data_age_ms=5_000,
    ).decide(
        _state(["100", "99.998", "99.996", "99.994"]),
        position=Decimal("0"),
        average_cost=Decimal("0"),
        maker_fee_rate=Decimal("0"),
        taker_fee_rate=Decimal("0"),
        allow_short=True,
        now_ns=2_000_000_000,
        wall_time_ms=-1,
    )
    assert decision["action"] == "WAIT"
    assert decision["reason"] == "FUTURE_STREAM_DATA"


def test_maker_variant_uses_taker_fallback_at_max_hold() -> None:
    router = strategy_router(
        "ORDER_BOOK_FADE",
        config_path=CONFIG,
        warmup_events=1,
        max_data_age_ms=5_000,
    )
    state = _state(["100", "100.001", "100.002", "100.003"])
    router.opened_ns[state.pair] = 1
    decision = router.decide(
        state,
        position=Decimal("1"),
        average_cost=Decimal("100"),
        maker_fee_rate=Decimal("-0.0002"),
        taker_fee_rate=Decimal("0.001"),
        allow_short=True,
        now_ns=20_000_000_000,
        wall_time_ms=1_001,
    )
    assert decision["action"] == "EXIT"
    assert decision["reason"] == "MAX_HOLD"
    assert decision["exit_order_style"] == "PAPER_TAKER"


def test_maker_exit_variant_keeps_forced_non_stop_exit_maker_only() -> None:
    router = strategy_router(
        "ORDER_BOOK_FADE_MAKER_EXIT",
        config_path=CONFIG,
        warmup_events=1,
        max_data_age_ms=5_000,
    )
    state = _state(["100", "100.001", "100.002", "100.003"])
    router.opened_ns[state.pair] = 1
    decision = router.decide(
        state,
        position=Decimal("1"),
        average_cost=Decimal("100"),
        maker_fee_rate=Decimal("-0.0002"),
        taker_fee_rate=Decimal("0.001"),
        allow_short=True,
        now_ns=20_000_000_000,
        wall_time_ms=1_001,
    )
    assert decision["action"] == "EXIT"
    assert decision["reason"] == "MAX_HOLD"
    assert decision["exit_order_style"] == "PAPER_MAKER_LIMIT"


def test_maker_exit_variant_keeps_stop_loss_taker() -> None:
    router = strategy_router(
        "ORDER_BOOK_FADE_MAKER_EXIT",
        config_path=CONFIG,
        warmup_events=1,
        max_data_age_ms=5_000,
    )
    state = _state(["100", "100.001", "100.002", "100.003"])
    decision = router.decide(
        state,
        position=Decimal("1"),
        average_cost=Decimal("100.1"),
        maker_fee_rate=Decimal("-0.0002"),
        taker_fee_rate=Decimal("0.001"),
        allow_short=True,
        now_ns=2_000_000_000,
        wall_time_ms=1_001,
    )
    assert decision["action"] == "EXIT"
    assert decision["reason"] == "STOP_LOSS"
    assert decision["exit_order_style"] == "PAPER_TAKER"


def test_queue_flow_candidate_requires_three_way_current_data_alignment() -> None:
    state = _state(["100", "100.001", "100.002", "100.003"])
    state.observe_transactions(
        [{"side": "buy", "price": "100.001", "amount": "10"}]
    )
    decision = strategy_router(
        "QUEUE_FLOW_MICROPRICE_MAKER",
        config_path=QUEUE_FLOW_CONFIG,
        warmup_events=1,
        max_data_age_ms=5_000,
    ).decide(
        state,
        position=Decimal("0"),
        average_cost=Decimal("0"),
        maker_fee_rate=Decimal("-0.0002"),
        taker_fee_rate=Decimal("0.0012"),
        allow_short=True,
        now_ns=2_000_000_000,
        wall_time_ms=1_001,
    )
    assert decision["action"] == "ENTER"
    assert decision["position_side"] == "LONG"
    assert decision["entry_order_style"] == "PAPER_MAKER_LIMIT"
    assert decision["authority"] == "NONE"
    assert decision["no_future_data"] is True


def test_queue_flow_candidate_blocks_opposing_trade_flow() -> None:
    state = _state(["100", "100.001", "100.002", "100.003"])
    state.observe_transactions(
        [{"side": "sell", "price": "100", "amount": "10"}]
    )
    decision = strategy_router(
        "QUEUE_FLOW_MICROPRICE_MAKER",
        config_path=QUEUE_FLOW_CONFIG,
        warmup_events=1,
        max_data_age_ms=5_000,
    ).decide(
        state,
        position=Decimal("0"),
        average_cost=Decimal("0"),
        maker_fee_rate=Decimal("-0.0002"),
        taker_fee_rate=Decimal("0.0012"),
        allow_short=True,
        now_ns=2_000_000_000,
        wall_time_ms=1_001,
    )
    assert decision["action"] == "WAIT"
    assert decision["reason"] == "TRADE_FLOW_NOT_ALIGNED"
