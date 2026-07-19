from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from quant_rabbit.dojo_bot_catalog import (
    AUTHORITY_INVARIANTS,
    DojoBotCatalogError,
)
from quant_rabbit.virtual_broker import VirtualBroker, VirtualBrokerError


BOT_PATH = Path(__file__).resolve().parents[1] / "bots" / "lab_bot.py"
SPEC = importlib.util.spec_from_file_location("dojo_new_family_lab_bot", BOT_PATH)
LAB_BOT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(LAB_BOT)
UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")


def _config(signal: str, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "signal": signal,
        "pairs": ["USD_JPY"],
        "tp_pips": 0,
        "sl_pips": 25,
        "ceiling_min": 240,
        "max_concurrent_per_pair": 1,
        "global_max_concurrent": 1,
        "per_pos_lev": 0.1,
        "atr_floor_pips": 0.1,
        "strategy_owner_id": f"test:{signal}",
    }
    values.update(overrides)
    return values


def _bar(
    stamp: datetime,
    *,
    open_mid: float,
    high_mid: float,
    low_mid: float,
    close_mid: float,
    spread: float = 0.02,
) -> dict[str, float | int]:
    half = spread / 2
    return {
        "epoch": int(stamp.timestamp()),
        "bid_o": open_mid - half,
        "bid_h": high_mid - half,
        "bid_l": low_mid - half,
        "bid_c": close_mid - half,
        "ask_o": open_mid + half,
        "ask_h": high_mid + half,
        "ask_l": low_mid + half,
        "ask_c": close_mid + half,
    }


def _complete(
    bot,
    broker: VirtualBroker,
    bar: dict[str, float | int],
    *,
    next_mid: float | None = None,
    next_spread: float = 0.02,
) -> None:
    epoch = int(bar["epoch"])
    executable_mid = float(bar["close_mid"]) if "close_mid" in bar else None
    if next_mid is None:
        executable_mid = (float(bar["bid_c"]) + float(bar["ask_c"])) / 2
    else:
        executable_mid = next_mid
    broker.on_quote(
        "USD_JPY",
        executable_mid - next_spread / 2,
        executable_mid + next_spread / 2,
        datetime.fromtimestamp(epoch + 60, UTC).isoformat(),
    )
    bot.on_bar_closed("USD_JPY", bar, epoch)


def _flat_bar(stamp: datetime, mid: float = 150.0) -> dict[str, float | int]:
    return _bar(
        stamp,
        open_mid=mid,
        high_mid=mid + 0.04,
        low_mid=mid - 0.04,
        close_mid=mid,
    )


def test_session_break_uses_london_local_range_and_next_open(tmp_path: Path) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    bot = LAB_BOT.Bot(
        broker,
        _config(
            "session_open_range_break",
            session_buffer_atr=0.2,
            session_tp_range=1.5,
            session_sl_range=0.75,
        ),
    )
    # 2025-06-02 00:00 Europe/London is 2025-06-01 23:00 UTC (BST).
    range_start = datetime(2025, 6, 1, 23, 0, tzinfo=UTC)
    for minute in range(8 * 60):
        _complete(bot, broker, _flat_bar(range_start + timedelta(minutes=minute)))
    assert broker.positions == {}

    inside_close = _bar(
        range_start + timedelta(hours=8),
        open_mid=150.0,
        high_mid=150.12,
        low_mid=149.98,
        close_mid=150.02,
    )
    _complete(bot, broker, inside_close, next_mid=150.03)
    assert broker.positions == {}

    breakout = _bar(
        range_start + timedelta(hours=8, minutes=1),
        open_mid=150.02,
        high_mid=150.11,
        low_mid=150.01,
        close_mid=150.09,
    )
    _complete(bot, broker, breakout, next_mid=150.07)

    position = next(iter(broker.positions.values()))
    assert position.side == "LONG"
    assert position.entry_price == pytest.approx(150.08)
    assert position.units == pytest.approx(200_000.0 * 0.1 / 150.08)
    assert position.tp_price > position.entry_price
    assert position.sl_price < position.entry_price
    assert bot.state["USD_JPY"].session_range_count == 480


def test_session_break_fails_closed_when_opening_range_has_a_gap(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    bot = LAB_BOT.Bot(broker, _config("session_open_range_break"))
    start = datetime(2025, 1, 6, 0, 0, tzinfo=UTC)
    for minute in range(8 * 60):
        if minute == 200:
            continue
        _complete(bot, broker, _flat_bar(start + timedelta(minutes=minute)))
    breakout = _bar(
        start + timedelta(hours=8),
        open_mid=150.0,
        high_mid=150.20,
        low_mid=149.99,
        close_mid=150.18,
    )
    _complete(bot, broker, breakout)

    assert broker.positions == {}
    state = bot.state["USD_JPY"]
    assert state.session_range_count == 479
    assert state.session_range_contiguous is False


def test_session_break_uses_next_open_spread_for_cost_gate(tmp_path: Path) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    bot = LAB_BOT.Bot(broker, _config("session_open_range_break"))
    start = datetime(2025, 1, 6, 0, 0, tzinfo=UTC)
    for minute in range(8 * 60):
        _complete(bot, broker, _flat_bar(start + timedelta(minutes=minute)))
    breakout = _bar(
        start + timedelta(hours=8),
        open_mid=150.0,
        high_mid=150.30,
        low_mid=149.99,
        close_mid=150.25,
    )

    _complete(bot, broker, breakout, next_mid=150.25, next_spread=1.0)

    assert broker.positions == {}


def test_daily_atr_coverage_rejects_partial_and_gapped_days_but_tolerates_noise(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    bot = LAB_BOT.Bot(broker, _config("weekend_gap_recovery"))
    state = bot.state["USD_JPY"]

    # The replay's first calendar day starts at noon and must never become D1.
    first_day = datetime(2025, 1, 6, tzinfo=UTC)
    for minute in range(12 * 60, 24 * 60):
        bot._update(state, _flat_bar(first_day + timedelta(minutes=minute)))
    bot._update(state, _flat_bar(first_day + timedelta(days=1)))
    assert state.daily_completed_count == 0

    # Density alone is insufficient: an eleven-minute intraday hole fails closed.
    second_day = first_day + timedelta(days=1)
    for minute in range(1, 24 * 60):
        if 12 * 60 <= minute <= 12 * 60 + 10:
            continue
        bot._update(state, _flat_bar(second_day + timedelta(minutes=minute)))
    bot._update(state, _flat_bar(second_day + timedelta(days=1)))
    assert state.daily_completed_count == 0

    # Isolated missing M1s are common in archives.  Edge coverage, 23h density,
    # and a maximum two-minute observed gap still prove a usable full D1.
    third_day = second_day + timedelta(days=1)
    for minute in range(1, 24 * 60):
        if minute % 60 == 30:
            continue
        bot._update(state, _flat_bar(third_day + timedelta(minutes=minute)))
    bot._update(state, _flat_bar(third_day + timedelta(days=1)))
    assert state.daily_completed_count == 1
    assert state.daily_true_ranges


def _prime_friday(
    bot,
    broker: VirtualBroker,
    *,
    observed_day_count: int = 14,
    friday_day: datetime = datetime(2025, 1, 10, tzinfo=UTC),
    missing_friday_minutes: frozenset[int] = frozenset(),
) -> datetime:
    observed_days: list[datetime] = []
    day = friday_day - timedelta(days=1)
    while len(observed_days) < observed_day_count:
        if day.weekday() < 4:
            observed_days.append(day.replace(hour=0, minute=0))
        day -= timedelta(days=1)
    for observed_day in reversed(observed_days):
        for minute in range(24 * 60):
            _complete(
                bot,
                broker,
                _flat_bar(observed_day + timedelta(minutes=minute)),
            )

    friday_local_date = friday_day.date()
    friday_close = datetime(
        friday_local_date.year,
        friday_local_date.month,
        friday_local_date.day,
        17,
        tzinfo=NEW_YORK,
    ).astimezone(UTC)
    friday_minutes = int((friday_close - friday_day).total_seconds() // 60)
    for minute in range(friday_minutes):
        if minute in missing_friday_minutes:
            continue
        last_stamp = friday_day + timedelta(minutes=minute)
        _complete(bot, broker, _flat_bar(last_stamp))
    assert last_stamp == friday_close - timedelta(minutes=1)
    return last_stamp


def _weekend_open(friday_day: datetime) -> datetime:
    sunday_date = friday_day.date() + timedelta(days=2)
    return datetime(
        sunday_date.year,
        sunday_date.month,
        sunday_date.day,
        17,
        tzinfo=NEW_YORK,
    ).astimezone(UTC)


def _seed_weekend_boundary(
    bot,
    broker: VirtualBroker,
    friday_day: datetime,
    *,
    omit_preclose_offset: int | None = None,
) -> datetime:
    state = bot.state["USD_JPY"]
    state.daily_atr = 0.08
    state.daily_completed_count = 14
    expected_days: list[str] = []
    cursor = friday_day.date() - timedelta(days=1)
    while len(expected_days) < 14:
        if cursor.weekday() < 4:
            expected_days.append(cursor.isoformat())
        cursor -= timedelta(days=1)
    state.daily_accepted_dates.extend(reversed(expected_days))
    sunday_open = _weekend_open(friday_day)
    friday_close = datetime(
        friday_day.year,
        friday_day.month,
        friday_day.day,
        17,
        tzinfo=NEW_YORK,
    ).astimezone(UTC)
    for offset in (2, 1):
        if offset == omit_preclose_offset:
            continue
        _complete(
            bot,
            broker,
            _flat_bar(friday_close - timedelta(minutes=offset)),
        )
    return sunday_open


def _sunday_bar(start: datetime, minute: int, *, spread: float = 0.02):
    close = 149.52 + minute * 0.002
    return _bar(
        start + timedelta(minutes=minute),
        open_mid=149.50 if minute == 0 else close - 0.002,
        high_mid=close + 0.01,
        low_mid=close - 0.01,
        close_mid=close,
        spread=spread,
    )


@pytest.mark.parametrize(
    ("friday_day", "expected_utc_hour"),
    [
        (datetime(2025, 1, 10, tzinfo=UTC), 22),
        (datetime(2025, 6, 6, tzinfo=UTC), 21),
    ],
)
def test_weekend_boundary_uses_new_york_17_with_dst(
    tmp_path: Path, friday_day: datetime, expected_utc_hour: int
) -> None:
    broker = VirtualBroker(
        tmp_path / f"{expected_utc_hour}.jsonl",
        balance_jpy=200_000.0,
        fast_ledger=True,
    )
    bot = LAB_BOT.Bot(broker, _config("weekend_gap_recovery"))
    sunday = _seed_weekend_boundary(bot, broker, friday_day)
    assert sunday.hour == expected_utc_hour

    _complete(bot, broker, _sunday_bar(sunday, 0))

    state = bot.state["USD_JPY"]
    assert state.weekend_valid is True
    assert state.weekend_evaluated is False
    assert state.weekend_bar_count == 1
    assert state.weekend_friday_close == pytest.approx(150.0)


def test_weekend_gap_fails_closed_when_preclose_boundary_m1_is_missing(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    bot = LAB_BOT.Bot(broker, _config("weekend_gap_recovery"))
    sunday = _seed_weekend_boundary(
        bot,
        broker,
        datetime(2025, 1, 10, tzinfo=UTC),
        omit_preclose_offset=2,
    )

    _complete(bot, broker, _sunday_bar(sunday, 0))

    state = bot.state["USD_JPY"]
    assert state.weekend_valid is False
    assert state.weekend_evaluated is True
    assert state.weekend_bar_count == 0


def test_weekend_gap_rejects_stale_daily_atr_even_with_exact_m1_boundary(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    bot = LAB_BOT.Bot(broker, _config("weekend_gap_recovery"))
    friday_day = datetime(2025, 1, 10, tzinfo=UTC)
    sunday = _seed_weekend_boundary(bot, broker, friday_day)
    state = bot.state["USD_JPY"]
    stale_days = [
        (datetime.fromisoformat(day).date() - timedelta(days=14)).isoformat()
        for day in state.daily_accepted_dates
    ]
    state.daily_accepted_dates.clear()
    state.daily_accepted_dates.extend(stale_days)

    _complete(bot, broker, _sunday_bar(sunday, 0))

    assert state.weekend_valid is False
    assert state.weekend_evaluated is True
    assert broker.positions == {}


def test_weekend_gap_fails_closed_when_open_or_postopen_m1_is_missing(
    tmp_path: Path,
) -> None:
    friday_day = datetime(2025, 6, 6, tzinfo=UTC)

    missing_open_broker = VirtualBroker(
        tmp_path / "missing-open.jsonl",
        balance_jpy=200_000.0,
        fast_ledger=True,
    )
    missing_open_bot = LAB_BOT.Bot(missing_open_broker, _config("weekend_gap_recovery"))
    sunday = _seed_weekend_boundary(missing_open_bot, missing_open_broker, friday_day)
    _complete(missing_open_bot, missing_open_broker, _sunday_bar(sunday, 1))
    assert missing_open_bot.state["USD_JPY"].weekend_bar_count == 0
    assert missing_open_bot.state["USD_JPY"].weekend_evaluated is True

    missing_post_broker = VirtualBroker(
        tmp_path / "missing-post.jsonl",
        balance_jpy=200_000.0,
        fast_ledger=True,
    )
    missing_post_bot = LAB_BOT.Bot(
        missing_post_broker,
        _config("weekend_gap_recovery", weekend_wait_bars=2),
    )
    _seed_weekend_boundary(missing_post_bot, missing_post_broker, friday_day)
    _complete(missing_post_bot, missing_post_broker, _sunday_bar(sunday, 0))
    _complete(missing_post_bot, missing_post_broker, _sunday_bar(sunday, 2))
    state = missing_post_bot.state["USD_JPY"]
    assert state.weekend_valid is False
    assert state.weekend_evaluated is True
    assert missing_post_broker.positions == {}


def test_weekend_gap_waits_for_completed_bars_then_targets_friday_close(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    bot = LAB_BOT.Bot(
        broker,
        _config(
            "weekend_gap_recovery",
            weekend_gap_atr=4.0,
            weekend_sl_gap=1.0,
            weekend_wait_bars=15,
            weekend_spread_fraction=0.15,
        ),
    )
    friday_last = _prime_friday(bot, broker)
    sunday = _weekend_open(datetime(2025, 1, 10, tzinfo=UTC))
    for minute in range(14):
        _complete(bot, broker, _sunday_bar(sunday, minute))
        assert broker.positions == {}

    final_bar = _sunday_bar(sunday, 14)
    _complete(bot, broker, final_bar, next_mid=149.56)

    position = next(iter(broker.positions.values()))
    assert position.side == "LONG"
    assert position.tp_price == pytest.approx(150.0)
    assert position.sl_price == pytest.approx(149.0)
    entry_events = [
        row
        for row in (
            json.loads(line) for line in broker.ledger_path.read_text().splitlines()
        )
        if row["payload"].get("trade_id") == position.trade_id
    ]
    assert entry_events[0]["event"] == "FILL_MARKET"
    assert entry_events[0]["payload"]["tp"] == pytest.approx(150.0)
    assert entry_events[0]["payload"]["sl"] == pytest.approx(149.0)
    assert all(row["event"] != "SET_EXIT" for row in entry_events)
    state = bot.state["USD_JPY"]
    assert state.weekend_bar_count == 15
    assert state.weekend_evaluated is True
    assert state.daily_completed_count == 14
    assert state.weekend_reference_atr == pytest.approx(state.daily_atr)
    assert state.last_bar_epoch > int(friday_last.timestamp())


def test_weekend_wait_and_spread_fraction_are_enforced(tmp_path: Path) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    bot = LAB_BOT.Bot(
        broker,
        _config(
            "weekend_gap_recovery",
            weekend_wait_bars=3,
            weekend_spread_fraction=0.02,
        ),
    )
    _prime_friday(bot, broker)
    sunday = _weekend_open(datetime(2025, 1, 10, tzinfo=UTC))
    for minute in range(2):
        _complete(bot, broker, _sunday_bar(sunday, minute))
        assert broker.positions == {}
    _complete(bot, broker, _sunday_bar(sunday, 2))

    assert broker.positions == {}
    assert bot.state["USD_JPY"].weekend_evaluated is True


def test_weekend_gap_requires_14_completed_utc_days(tmp_path: Path) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    bot = LAB_BOT.Bot(
        broker,
        _config(
            "weekend_gap_recovery",
            weekend_wait_bars=1,
            weekend_spread_fraction=0.15,
        ),
    )
    _prime_friday(bot, broker, observed_day_count=13)
    sunday = _weekend_open(datetime(2025, 1, 10, tzinfo=UTC))
    _complete(bot, broker, _sunday_bar(sunday, 0))

    assert broker.positions == {}
    state = bot.state["USD_JPY"]
    assert state.daily_completed_count == 13
    assert state.daily_atr is None
    assert state.weekend_bar_count == 0


def test_weekend_position_obeys_existing_terminal_ceiling(tmp_path: Path) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    bot = LAB_BOT.Bot(
        broker,
        _config(
            "weekend_gap_recovery",
            weekend_wait_bars=2,
            weekend_spread_fraction=0.15,
            ceiling_min=1,
        ),
    )
    _prime_friday(bot, broker)
    sunday = _weekend_open(datetime(2025, 1, 10, tzinfo=UTC))
    _complete(bot, broker, _sunday_bar(sunday, 0))
    _complete(bot, broker, _sunday_bar(sunday, 1), next_mid=149.55)
    assert len(broker.positions) == 1

    _complete(bot, broker, _sunday_bar(sunday, 2), next_mid=149.57)
    assert broker.positions == {}


def _prime_overlay(bot, broker: VirtualBroker) -> datetime:
    start = datetime(2025, 1, 8, 12, 0, tzinfo=UTC)
    for minute in range(20):
        _complete(bot, broker, _flat_bar(start + timedelta(minutes=minute)))
    return start + timedelta(minutes=20)


def test_breakeven_overlay_uses_entry_atr_once_and_preserves_tp(
    tmp_path: Path,
) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    bot = LAB_BOT.Bot(
        broker,
        _config(
            "weekend_gap_recovery",
            exit_policy="BREAKEVEN",
            be_trigger_atr=1.0,
            be_offset_pips=1.0,
        ),
    )
    stamp = _prime_overlay(bot, broker)
    trade_id = bot.broker.market_order("USD_JPY", "LONG", 100, tp_pips=50, sl_pips=20)
    original = bot.broker.position(trade_id)
    assert original is not None
    original_tp = original.tp_price

    # First callback only discovers the fill and cannot use its whole bar.
    _complete(
        bot,
        broker,
        _bar(
            stamp,
            open_mid=150.0,
            high_mid=150.08,
            low_mid=149.99,
            close_mid=150.06,
        ),
    )
    discovered = bot.broker.position(trade_id)
    assert discovered is not None
    assert discovered.sl_price == original.sl_price

    _complete(
        bot,
        broker,
        _bar(
            stamp + timedelta(minutes=1),
            open_mid=150.06,
            high_mid=150.16,
            low_mid=150.05,
            close_mid=150.13,
        ),
    )
    protected = bot.broker.position(trade_id)
    assert protected is not None
    assert protected.tp_price == original_tp
    assert protected.sl_price == pytest.approx(original.entry_price + 0.01)


def test_atr_trailing_overlay_never_widens_an_existing_stop(tmp_path: Path) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    bot = LAB_BOT.Bot(
        broker,
        _config(
            "weekend_gap_recovery",
            exit_policy="ATR_TRAILING",
            trail_trigger_atr=0.5,
            trail_distance_atr=1.0,
        ),
    )
    stamp = _prime_overlay(bot, broker)
    trade_id = bot.broker.market_order("USD_JPY", "LONG", 100, tp_pips=100, sl_pips=20)
    _complete(bot, broker, _flat_bar(stamp, mid=150.04))
    _complete(
        bot,
        broker,
        _bar(
            stamp + timedelta(minutes=1),
            open_mid=150.04,
            high_mid=150.20,
            low_mid=150.03,
            close_mid=150.18,
        ),
    )
    trailed = bot.broker.position(trade_id)
    assert trailed is not None
    first_stop = trailed.sl_price
    assert first_stop is not None

    tighter_manual_stop = first_stop + 0.02
    bot.broker.set_exit(
        trade_id, tp_price=trailed.tp_price, sl_price=tighter_manual_stop
    )
    _complete(
        bot,
        broker,
        _bar(
            stamp + timedelta(minutes=2),
            open_mid=150.18,
            high_mid=150.19,
            low_mid=150.10,
            close_mid=150.18,
        ),
    )
    final = bot.broker.position(trade_id)
    assert final is not None
    assert final.sl_price >= tighter_manual_stop


def _trade_ledger_events(path: Path, trade_id: str) -> list[dict[str, object]]:
    return [
        record
        for record in (json.loads(line) for line in path.read_text().splitlines())
        if record["payload"].get("trade_id") == trade_id
    ]


def test_breakeven_long_closes_at_next_quote_gapped_through_new_stop(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    broker = VirtualBroker(ledger_path, balance_jpy=200_000.0, fast_ledger=True)
    bot = LAB_BOT.Bot(
        broker,
        _config(
            "weekend_gap_recovery",
            exit_policy="BREAKEVEN",
            be_trigger_atr=1.0,
            be_offset_pips=0.0,
        ),
    )
    stamp = _prime_overlay(bot, broker)
    trade_id = bot.broker.market_order("USD_JPY", "LONG", 100, tp_pips=100, sl_pips=20)
    _complete(bot, broker, _flat_bar(stamp, mid=150.04))

    # The completed close triggers breakeven, but the already-staged next bid
    # gaps below the new stop while remaining above the original fixed SL.
    _complete(
        bot,
        broker,
        _bar(
            stamp + timedelta(minutes=1),
            open_mid=150.04,
            high_mid=150.18,
            low_mid=150.03,
            close_mid=150.13,
        ),
        next_mid=149.90,
    )

    assert trade_id not in broker.positions
    state = bot.state["USD_JPY"]
    assert trade_id not in state.my_trades
    assert trade_id not in state.trade_entry_atr
    events = _trade_ledger_events(ledger_path, trade_id)
    assert [event["event"] for event in events] == ["FILL_MARKET", "CLOSE"]
    assert events[-1]["payload"]["price"] == pytest.approx(149.89)


def test_atr_trailing_short_closes_at_next_quote_gapped_through_new_stop(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    broker = VirtualBroker(ledger_path, balance_jpy=200_000.0, fast_ledger=True)
    bot = LAB_BOT.Bot(
        broker,
        _config(
            "weekend_gap_recovery",
            exit_policy="ATR_TRAILING",
            trail_trigger_atr=0.5,
            trail_distance_atr=0.5,
        ),
    )
    stamp = _prime_overlay(bot, broker)
    trade_id = bot.broker.market_order("USD_JPY", "SHORT", 100, tp_pips=100, sl_pips=20)
    _complete(bot, broker, _flat_bar(stamp, mid=149.96))

    # The completed low activates the trail.  The next ask gaps through its
    # candidate without reaching the original, much wider fixed SL.
    _complete(
        bot,
        broker,
        _bar(
            stamp + timedelta(minutes=1),
            open_mid=149.96,
            high_mid=149.98,
            low_mid=149.70,
            close_mid=149.74,
        ),
        next_mid=150.00,
    )

    assert trade_id not in broker.positions
    state = bot.state["USD_JPY"]
    assert trade_id not in state.my_trades
    assert trade_id not in state.trade_overlay_extreme
    events = _trade_ledger_events(ledger_path, trade_id)
    assert [event["event"] for event in events] == ["FILL_MARKET", "CLOSE"]
    assert events[-1]["payload"]["price"] == pytest.approx(150.01)


def test_overlay_failed_gap_close_retains_owned_trade_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broker = VirtualBroker(
        tmp_path / "ledger.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    bot = LAB_BOT.Bot(
        broker,
        _config(
            "weekend_gap_recovery",
            exit_policy="BREAKEVEN",
            be_trigger_atr=1.0,
            be_offset_pips=0.0,
        ),
    )
    stamp = _prime_overlay(bot, broker)
    trade_id = bot.broker.market_order("USD_JPY", "LONG", 100, tp_pips=100, sl_pips=20)
    _complete(bot, broker, _flat_bar(stamp, mid=150.04))

    def fail_close(_trade_id: str, units: float | None = None) -> float:
        raise VirtualBrokerError("injected close failure")

    monkeypatch.setattr(broker, "close_trade", fail_close)
    _complete(
        bot,
        broker,
        _bar(
            stamp + timedelta(minutes=1),
            open_mid=150.04,
            high_mid=150.18,
            low_mid=150.03,
            close_mid=150.13,
        ),
        next_mid=149.90,
    )

    assert trade_id in broker.positions
    state = bot.state["USD_JPY"]
    assert trade_id in state.my_trades
    assert trade_id in state.trade_entry_atr


def test_catalog_authority_config_is_revalidated_but_keeps_owner_override(
    tmp_path: Path,
) -> None:
    catalog_config: dict[str, object] = {
        "signal": "weekend_gap_recovery",
        "pairs": ["USD_JPY"],
        "tp_pips": None,
        "tp_atr": None,
        "sl_pips": None,
        "ceiling_min": 240,
        "max_concurrent_per_pair": 1,
        "global_max_concurrent": 1,
        "per_pos_lev": 0.1,
        "atr_floor_pips": 0.1,
        "exit_policy": "FIXED",
        "weekend_gap_atr": 4.0,
        "weekend_sl_gap": 1.0,
        "weekend_wait_bars": 15,
        "weekend_spread_fraction": 0.15,
        "strategy_owner_id": "test:catalog-owner",
        **dict(AUTHORITY_INVARIANTS),
    }
    broker = VirtualBroker(
        tmp_path / "accepted.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    bot = LAB_BOT.Bot(broker, catalog_config)
    assert bot.owner_id == "test:catalog-owner"
    assert bot.sl_pips is None

    drifted = {**catalog_config, "sl_pips": 25}
    other_broker = VirtualBroker(
        tmp_path / "rejected.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    with pytest.raises(DojoBotCatalogError, match="dynamic-stop"):
        LAB_BOT.Bot(other_broker, drifted)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exit_policy", "UNKNOWN"),
        ("weekend_wait_bars", 0),
        ("weekend_wait_bars", 1.5),
        ("weekend_spread_fraction", 0),
        ("session_buffer_atr", float("nan")),
        ("be_offset_pips", -1),
        ("trail_distance_atr", 0),
    ],
)
def test_new_family_and_overlay_config_fail_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    broker = VirtualBroker(
        tmp_path / f"{field}.jsonl", balance_jpy=200_000.0, fast_ledger=True
    )
    with pytest.raises(ValueError):
        LAB_BOT.Bot(
            broker,
            _config("weekend_gap_recovery", **{field: value}),
        )
