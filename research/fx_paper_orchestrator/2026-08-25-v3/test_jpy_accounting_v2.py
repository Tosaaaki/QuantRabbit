from __future__ import annotations

import math

import pytest

import jpy_accounting_reference_v1 as reference
import jpy_accounting_v2 as accounting


T0 = "2026-05-29T12:00:00.000000000Z"
T1 = "2026-05-29T18:00:00.000000000Z"


def bbo(pair: str, time: str, bid: float, ask: float) -> accounting.BBO:
    return accounting.BBO(pair, time, bid, ask)


def test_independent_reference_preserves_nonzero_submicrosecond_elapsed_time() -> None:
    start = "2026-05-29T12:00:00.000000001Z"
    end = "2026-05-29T12:00:00.000000009Z"
    assert reference.parse_utc_nanoseconds(end) - reference.parse_utc_nanoseconds(start) == 8
    with pytest.raises(reference.ReferenceError, match="invalid UTC timestamp"):
        reference.parse_utc_nanoseconds("2026-05-29T12:00:00.000000000+00:00")


def usd_book(
    entry: tuple[float, float] = (150.00, 150.02),
    exit: tuple[float, float] = (151.00, 151.03),
) -> accounting.ConversionBook:
    return accounting.ConversionBook([
        bbo("USD_JPY", T0, *entry),
        bbo("USD_JPY", T1, *exit),
    ])


def test_positive_asset_and_negative_liability_use_opposite_executable_sides() -> None:
    book = usd_book()
    asset = book.convert_to_jpy(100.0, "USD", T0)
    liability = book.convert_to_jpy(-100.0, "USD", T0)
    assert asset.side == "ASSET_SELL"
    assert liability.side == "LIABILITY_BUYBACK"
    assert asset.jpy_amount == 100.0 * 150.00
    assert liability.jpy_amount == -100.0 * 150.02
    assert asset.executable_rate_jpy_per_currency < liability.executable_rate_jpy_per_currency
    assert asset.bid_ask_width_jpy == pytest.approx(2.0)


def test_cad_and_chf_quote_conversion_is_explicit_two_hop() -> None:
    book = accounting.ConversionBook([
        bbo("USD_CAD", T0, 1.3500, 1.3504),
        bbo("USD_CHF", T0, 0.9000, 0.9003),
        bbo("USD_JPY", T0, 150.00, 150.02),
    ])
    cad_asset = book.convert_to_jpy(100.0, "CAD", T0)
    cad_liability = book.convert_to_jpy(-100.0, "CAD", T0)
    chf_asset = book.convert_to_jpy(100.0, "CHF", T0)
    assert cad_asset.path == ("USD_CAD", "USD_JPY")
    assert cad_asset.jpy_amount == pytest.approx(100.0 / 1.3504 * 150.00)
    assert cad_liability.jpy_amount == pytest.approx(-100.0 * 150.02 / 1.3500)
    assert chf_asset.jpy_amount == pytest.approx(100.0 / 0.9003 * 150.00)


@pytest.mark.parametrize("currency", ["USD", "CAD", "CHF", "JPY"])
def test_jpy_conversion_is_scale_invariant_and_roundtrip_loss_nonnegative(currency: str) -> None:
    book = accounting.ConversionBook([
        bbo("USD_CAD", T0, 1.3500, 1.3504),
        bbo("USD_CHF", T0, 0.9000, 0.9003),
        bbo("USD_JPY", T0, 150.00, 150.02),
    ])
    small = book.convert_to_jpy(7.0, currency, T0)
    large = book.convert_to_jpy(70.0, currency, T0)
    assert large.jpy_amount == pytest.approx(10.0 * small.jpy_amount)
    retention = book.roundtrip_retention(100.0, currency, T0)
    assert 0.0 < retention <= 1.0 + 1e-15
    assert 1.0 - retention >= -1e-15


def test_quote_inversion_preserves_executable_conversion() -> None:
    direct = accounting.ConversionBook([bbo("USD_JPY", T0, 150.00, 150.03)])
    inverse_paths = {
        "USD": ((accounting.ConversionHop("JPY_USD", "USD", "JPY"),),),
    }
    inverse = accounting.ConversionBook([
        bbo("JPY_USD", T0, 1.0 / 150.03, 1.0 / 150.00),
    ], path_candidates=inverse_paths)
    for amount in (100.0, -100.0):
        assert inverse.convert_to_jpy(amount, "USD", T0).jpy_amount == pytest.approx(
            direct.convert_to_jpy(amount, "USD", T0).jpy_amount
        )


def test_future_quote_is_ignored_but_missing_stale_invalid_or_ambiguous_fails_closed() -> None:
    future = "2026-05-29T12:10:00.000000000Z"
    query = "2026-05-29T12:04:59.000000000Z"
    first = bbo("USD_JPY", T0, 150.00, 150.02)
    baseline = accounting.ConversionBook([first]).convert_to_jpy(10.0, "USD", query)
    with_future = accounting.ConversionBook([
        first, bbo("USD_JPY", future, 200.00, 200.02),
    ]).convert_to_jpy(10.0, "USD", query)
    assert with_future.jpy_amount == baseline.jpy_amount
    with pytest.raises(accounting.AccountingError, match="stale"):
        accounting.ConversionBook([first]).convert_to_jpy(
            1.0, "USD", "2026-05-29T12:05:01.000000000Z"
        )
    with pytest.raises(accounting.AccountingError, match="missing causal"):
        accounting.ConversionBook([bbo("USD_JPY", future, 150.0, 150.02)]).convert_to_jpy(
            1.0, "USD", T0
        )
    for bid, ask in ((0.0, 1.0), (-1.0, 1.0), (2.0, 1.0)):
        with pytest.raises(accounting.AccountingError):
            bbo("USD_JPY", T0, bid, ask)
    ambiguous = {
        "USD": (
            (accounting.ConversionHop("USD_JPY", "USD", "JPY"),),
            (accounting.ConversionHop("USD_JPY", "USD", "JPY"),),
        )
    }
    with pytest.raises(accounting.AccountingError, match="ambiguous"):
        accounting.ConversionBook([first], path_candidates=ambiguous)
    with pytest.raises(accounting.AccountingError, match="missing conversion path"):
        accounting.ConversionBook([first], path_candidates={}).convert_to_jpy(1.0, "USD", T0)


def test_future_or_stale_pair_bbo_fails_closed() -> None:
    book = usd_book()
    with pytest.raises(accounting.AccountingError, match="future entry pair"):
        accounting.size_position(
            "F", "EUR_USD", 1, 28_000.0, T0,
            bbo("EUR_USD", "2026-05-29T12:00:00.000000001Z", 1.1, 1.1002),
            book,
        )
    stale_time = "2026-05-29T12:05:01.000000000Z"
    stale_book = accounting.ConversionBook([
        bbo("USD_JPY", T0, 150.00, 150.02),
        bbo("USD_JPY", stale_time, 150.00, 150.02),
    ])
    with pytest.raises(accounting.AccountingError, match="stale entry pair"):
        accounting.size_position(
            "S", "EUR_USD", 1, 28_000.0, stale_time,
            bbo("EUR_USD", T0, 1.1, 1.1002), stale_book,
        )


@pytest.mark.parametrize("direction", [1, -1])
def test_long_and_short_linear_pnl_match_independent_reference(direction: int) -> None:
    book = usd_book()
    entry_pair = bbo("EUR_USD", T0, 1.1000, 1.1002)
    exit_pair = bbo("EUR_USD", T1, 1.1010, 1.1012)
    position = accounting.size_position(
        "P", "EUR_USD", direction, 28_000.0, T0, entry_pair, book
    )
    actual = accounting.evaluate_position(
        position, T1, exit_pair, book, accounting.ADVERSE_STRESS
    )
    expected = reference.episode(
        pair="EUR_USD",
        direction=direction,
        notional_jpy=28_000.0,
        entry_bid=entry_pair.bid,
        entry_ask=entry_pair.ask,
        exit_bid=exit_pair.bid,
        exit_ask=exit_pair.ask,
        entry_conversion_quotes={"USD_JPY": (150.00, 150.02)},
        exit_conversion_quotes={"USD_JPY": (151.00, 151.03)},
        quote_to_jpy_path=(("USD_JPY", "USD", "JPY"),),
        entry_time=T0,
        exit_time=T1,
        slippage_pips=0.9,
        commission_bps_per_side=0.2,
        financing_bps_per_day=1.5,
        raw_pair_mid=False,
    )
    for field in (
        "units", "raw_quote_pnl", "gross_jpy", "executable_pair_quote_pnl",
        "executable_pair_jpy", "commission_cost_jpy", "financing_cost_jpy", "net_jpy",
    ):
        assert actual[field] == pytest.approx(expected[field], rel=1e-12, abs=1e-12)
    linear_expected = direction * position.units * (exit_pair.mid - entry_pair.mid)
    assert actual["raw_quote_pnl"] == pytest.approx(linear_expected)
    if direction < 0:
        assert actual["raw_quote_pnl"] == pytest.approx(
            position.units * (entry_pair.mid - exit_pair.mid)
        )
    assert actual["account_currency_midpoint_conversion_used"] is False


def test_usd_jpy_quote_is_direct_jpy_and_conversion_move_is_recorded_without_midpoint() -> None:
    empty_book = accounting.ConversionBook([])
    entry = bbo("USD_JPY", T0, 150.00, 150.02)
    exit_quote = bbo("USD_JPY", T1, 151.00, 151.03)
    position = accounting.size_position("J", "USD_JPY", 1, 28_000.0, T0, entry, empty_book)
    result = accounting.evaluate_position(position, T1, exit_quote, empty_book, accounting.EXECUTABLE_BASE)
    assert result["pair_pnl_conversion"]["path"] == ()
    assert result["pair_pnl_conversion"]["executable_rate_jpy_per_currency"] == 1.0
    assert result["conversion_move_jpy"] == 0.0

    moving = usd_book((150.00, 150.02), (151.00, 151.03))
    eur = accounting.size_position(
        "E", "EUR_USD", 1, 28_000.0, T0,
        bbo("EUR_USD", T0, 1.1000, 1.1002), moving,
    )
    moved = accounting.evaluate_position(
        eur, T1, bbo("EUR_USD", T1, 1.1010, 1.1012),
        moving, accounting.EXECUTABLE_BASE,
    )
    assert moved["conversion_move_jpy"] != 0.0


def test_weekend_financing_uses_actual_elapsed_time() -> None:
    friday = "2026-05-29T21:00:00.000000000Z"
    monday = "2026-06-01T21:00:00.000000000Z"
    book = accounting.ConversionBook([
        bbo("USD_JPY", friday, 150.00, 150.02),
        bbo("USD_JPY", monday, 150.20, 150.23),
    ])
    position = accounting.size_position(
        "W", "EUR_USD", 1, 28_000.0, friday,
        bbo("EUR_USD", friday, 1.1000, 1.1002), book,
    )
    result = accounting.evaluate_position(
        position, monday, bbo("EUR_USD", monday, 1.1010, 1.1012),
        book, accounting.EXECUTABLE_BASE,
    )
    assert result["elapsed_seconds"] == 72 * 60 * 60
    assert result["elapsed_financing_days"] == 3.0
    assert result["financing_cost_jpy"] > 0.0


def test_month_boundary_terminal_liquidation_and_portfolio_sum_reconcile() -> None:
    entry_time = "2026-05-31T23:55:00.000000000Z"
    exit_time = "2026-06-01T00:05:00.000000000Z"
    book = accounting.ConversionBook([
        bbo("USD_JPY", entry_time, 150.00, 150.02),
        bbo("USD_JPY", exit_time, 150.20, 150.23),
    ])
    ledger = accounting.JPYAccountLedger(200_000.0, book, accounting.EXECUTABLE_BASE)
    ledger.open(
        "EUR", "EUR_USD", 1, 28_000.0, entry_time,
        bbo("EUR_USD", entry_time, 1.1000, 1.1002),
    )
    ledger.open(
        "JPY", "USD_JPY", -1, 28_000.0, entry_time,
        bbo("USD_JPY", entry_time, 150.00, 150.02),
    )
    before = ledger.cash_jpy
    terminal = ledger.terminal_liquidate(exit_time, {
        "EUR_USD": bbo("EUR_USD", exit_time, 1.1010, 1.1012),
        "USD_JPY": bbo("USD_JPY", exit_time, 150.20, 150.23),
    })
    close_rows = [row for row in ledger.events if row["event_type"] == "CLOSE"]
    summed = sum(row["realized_net_jpy"] for row in close_rows)
    assert terminal["terminal_open_inventory"] == 0
    assert terminal["terminal_inventory_mtm_jpy"] == 0.0
    assert ledger.positions == {}
    assert terminal["ending_cash_jpy"] == pytest.approx(before + summed)
    assert terminal["realized_net_jpy"] == pytest.approx(summed)
    assert ledger.realized_by_exit_month() == {"2026-06": pytest.approx(summed)}
    assert len({row["event_sha256"] for row in ledger.events}) == len(ledger.events)


def test_fixed_jpy_notional_scale_invariance_for_jpy_and_non_jpy_quotes() -> None:
    for pair, book, entry, exit_quote in (
        (
            "USD_JPY", accounting.ConversionBook([]),
            bbo("USD_JPY", T0, 150.00, 150.02), bbo("USD_JPY", T1, 151.00, 151.03),
        ),
        (
            "EUR_USD", usd_book(),
            bbo("EUR_USD", T0, 1.1000, 1.1002), bbo("EUR_USD", T1, 1.1010, 1.1012),
        ),
    ):
        small = accounting.size_position("S", pair, 1, 10_000.0, T0, entry, book)
        large = accounting.size_position("L", pair, 1, 50_000.0, T0, entry, book)
        small_result = accounting.evaluate_position(
            small, T1, exit_quote, book, accounting.EXECUTABLE_BASE
        )
        large_result = accounting.evaluate_position(
            large, T1, exit_quote, book, accounting.EXECUTABLE_BASE
        )
        assert large.units == pytest.approx(5.0 * small.units)
        assert large_result["net_jpy"] == pytest.approx(5.0 * small_result["net_jpy"])
        assert math.isclose(
            large_result["net_return_on_fixed_notional"],
            small_result["net_return_on_fixed_notional"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
