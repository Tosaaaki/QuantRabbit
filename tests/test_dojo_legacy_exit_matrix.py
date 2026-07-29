from datetime import datetime, timedelta, timezone

from quant_rabbit.dojo_legacy_exit_matrix import (
    EntrySignal,
    ExitPolicy,
    InventoryPolicy,
    Quote,
    ReplayCosts,
    replay_arm,
    replay_exit_matrix,
)


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def quote(seconds: int, mid: float, spread: float = 0.02) -> Quote:
    return Quote(
        timestamp=BASE + timedelta(seconds=seconds),
        bid=mid - spread / 2,
        ask=mid + spread / 2,
    )


def test_matrix_uses_same_input_entry_cohort_for_all_arms() -> None:
    quotes = [quote(0, 150.00), quote(60, 150.08), quote(120, 149.92)]
    entries = [
        EntrySignal("a", BASE, "long", atr_pips=2.0, take_profit_pips=5.0),
        EntrySignal(
            "b", BASE + timedelta(seconds=30), "short", atr_pips=2.0, take_profit_pips=5.0
        ),
    ]
    results = replay_exit_matrix(
        quotes=quotes,
        entries=entries,
        costs=ReplayCosts(slippage_pips_per_fill=0.0),
    )

    assert len(results) == 10
    assert {result.metrics.entry_cohort_size for result in results} == {2}


def test_future_quotes_do_not_change_an_already_closed_trade() -> None:
    initial_quotes = [quote(0, 150.00), quote(60, 150.10)]
    entries = [
        EntrySignal("a", BASE, "long", atr_pips=2.0, take_profit_pips=5.0)
    ]
    arguments = {
        "entries": entries,
        "policy": ExitPolicy("fixed_sl", fixed_stop_pips=5.0),
        "costs": ReplayCosts(slippage_pips_per_fill=0.0),
        "inventory": InventoryPolicy(False),
    }
    first = replay_arm(quotes=initial_quotes, **arguments)
    extended = replay_arm(
        quotes=initial_quotes + [quote(120, 140.00)],
        **arguments,
    )

    assert first.trades == extended.trades
    assert first.metrics.net_jpy == extended.metrics.net_jpy


def test_spread_slippage_financing_and_ai_cost_are_included() -> None:
    quotes = [quote(0, 150.00), quote(86_400, 150.00)]
    entries = [
        EntrySignal("a", BASE, "long", atr_pips=2.0, take_profit_pips=99.0)
    ]
    result = replay_arm(
        quotes=quotes,
        entries=entries,
        policy=ExitPolicy("no_sl"),
        costs=ReplayCosts(
            units=1_000,
            slippage_pips_per_fill=0.5,
            financing_jpy_per_10k_units_per_day=10.0,
            ai_cost_jpy_per_decision=2.0,
        ),
        inventory=InventoryPolicy(True, checkpoint_seconds=60),
    )

    assert result.metrics.trades == 1
    assert result.trades[0].gross_jpy == -30.0
    assert result.trades[0].financing_jpy == 1.0
    assert result.metrics.ai_decisions == 1
    assert result.metrics.ai_cost_jpy == 2.0
    assert result.metrics.net_jpy == -33.0


def test_period_end_marks_open_position_to_executable_quote() -> None:
    quotes = [quote(0, 150.00), quote(60, 150.02)]
    result = replay_arm(
        quotes=quotes,
        entries=[
            EntrySignal("a", BASE, "long", atr_pips=2.0, take_profit_pips=99.0)
        ],
        policy=ExitPolicy("no_sl"),
        costs=ReplayCosts(slippage_pips_per_fill=0.0),
        inventory=InventoryPolicy(False),
    )

    assert result.trades[0].exit_reason == "period_end_mtm"
    assert result.metrics.net_jpy == 0.0
