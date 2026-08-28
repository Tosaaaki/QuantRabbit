from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Callable

import pytest

import paper_research_double_entry_reference_v2 as reference
import paper_research_jpy_oracle_v2 as oracle
import test_paper_research_jpy_oracle_v2 as fixture_support


REFERENCE_INPUTS = (
    "source_blob",
    "source_manifest",
    "proposal",
    "execution_policy",
    "inventory_policy",
    "accounting_policy",
    "evaluation_policy",
    "instrument_registry",
    "authority_policy",
)


def test_contracts_freeze_signed_marked_nodes_and_later_hard_cap_closeout() -> None:
    root = Path(__file__).resolve().parent
    oracle_contract = json.loads(
        (root / "PAPER_RESEARCH_JPY_ORACLE_CONTRACT_V2.json").read_bytes()
    )
    reference_contract = json.loads(
        (root / "PAPER_RESEARCH_DOUBLE_ENTRY_REFERENCE_CONTRACT_V2.json").read_bytes()
    )
    oracle_units = oracle_contract["notional_and_conversion_semantics"]
    oracle_invariants = oracle_contract["corrective_invariants"]
    reference_economics = reference_contract["economic_semantics"]
    reference_risk = reference_contract["risk_and_aggregation"]
    assert oracle_units["cap_margin_and_currency_exposure_basis"] \
        == "ACTUAL_MARKED_NOTIONAL_NOT_PROPOSAL_TARGET"
    assert oracle_units["gross_and_margin_scalar_basis"] == (
        "ACTUAL_EXECUTABLE_MARKED_NOTIONAL_UNCHANGED_BY_SIGNED_CURRENCY_NODE_PROJECTION"
    )
    assert oracle_units["currency_exposure_vector_basis"] == (
        "SIGNED_NATIVE_BASE_UNITS_AND_NEGATED_CURRENT_EXECUTABLE_LIQUIDATION_"
        "QUOTE_COUNTERVALUE_CONVERTED_INDEPENDENTLY"
    )
    assert oracle_units["currency_node_rounding"] == (
        "SUM_EXACT_PER_CURRENCY_THEN_SIGNED_OUTWARD_ONCE_POSITIVE_CEILING_"
        "NEGATIVE_FLOOR"
    )
    assert oracle_invariants["inventory_caps"].startswith(
        "MAX_GROSS_AND_MAX_SIGNED_CURRENCY_NODE_MAGNITUDE_ARE_HARD_GUARDS_AT_ENTRY_"
    )
    assert oracle_invariants["risk_closeout_precedence"] == (
        "EQUITY_FREE_MARGIN_RATIO_OR_EVALUATION_MARGIN_CAP_FIRST_AS_MARGIN_"
        "CLOSEOUT;OTHERWISE_INVENTORY_GROSS_OR_CURRENCY_CAP_AS_INVENTORY_CAP_CLOSEOUT"
    )
    assert reference_economics["cap_margin_currency_exposure_basis"] \
        == "ACTUAL_MARKED_NOTIONAL_NOT_PROPOSAL_TARGET"
    assert reference_economics["currency_exposure_vector_basis"] == (
        "TWO_INDEPENDENT_POSTINGS_PER_LOT_BASE_POSITION_AND_NEGATED_CURRENT_"
        "EXECUTABLE_LIQUIDATION_QUOTE_COUNTERVALUE"
    )
    assert reference_risk["inventory_caps"] == oracle_invariants["inventory_caps"]
    assert reference_risk["risk_closeout_precedence"] \
        == oracle_invariants["risk_closeout_precedence"]
    status = reference_contract["historical_phase_1_test_status"][
        "corrective_fixture_status"
    ]
    assert status[
        "independent_signed_currency_nodes_use_current_executable_liquidation_mark"
    ] is True
    assert status["later_inventory_cap_breach_closeout_and_account_halt"] is True


def _quote(
    bid_ticks: int,
    ask_ticks: int,
    tick_scale: int,
    *,
    source_ts_ns: int = 100,
    arrival_ts_ns: int = 100,
) -> dict[str, int]:
    return {
        "source_ts_ns": source_ts_ns,
        "arrival_ts_ns": arrival_ts_ns,
        "bid_ticks": bid_ticks,
        "ask_ticks": ask_ticks,
        "tick_scale": tick_scale,
    }


def _registry_specs(*instruments: str) -> dict[str, dict[str, int]]:
    return {
        instrument: {
            "price_scale": 100 if "JPY" in instrument else 100_000,
            "pip_ticks": 1,
        }
        for instrument in sorted(instruments)
    }


def _multi_instrument_rows(
    registry: dict[str, dict[str, int]],
) -> list[dict[str, object]]:
    offsets = (0, 1, 2, 301, 302, 360, 361, 662, 900)
    base_prices = {
        "CAD_CHF": 65_000,
        "CAD_JPY": 11_000,
        "CHF_JPY": 17_000,
        "EUR_CAD": 150_000,
        "EUR_CHF": 95_000,
        "EUR_USD": 110_000,
        "USD_CAD": 125_000,
        "USD_CHF": 90_000,
        "USD_JPY": 15_000,
    }
    rows: list[dict[str, object]] = []
    for sequence, seconds in enumerate(offsets, 1):
        source_ts_ns = fixture_support.START_NS + seconds * 1_000_000_000
        for arrival_order, instrument in enumerate(sorted(registry), 1):
            bid_ticks = base_prices[instrument] + sequence * 3
            rows.append({
                "schema_version": 1,
                "provider_id": fixture_support.PROVIDER,
                "instrument": instrument,
                "bid_ticks": bid_ticks,
                "ask_ticks": bid_ticks + 2,
                "tick_scale": registry[instrument]["price_scale"],
                "source_ts_ns": source_ts_ns,
                "arrival_ts_ns": source_ts_ns + arrival_order * 10_000_000,
                "provider_event_id": f"{instrument}-{sequence}",
                "sequence": sequence,
                "heartbeat": False,
                "quality_flags": [],
            })
    return sorted(rows, key=lambda row: (
        row["arrival_ts_ns"], row["source_ts_ns"], row["provider_id"],
        row["instrument"], row["sequence"],
    ))


def _custom_market_fixture(
    root: Path,
    instruments: tuple[str, ...],
    proposal_instrument: str,
    *,
    direction: int = 1,
) -> tuple[dict, dict[str, dict[str, int]]]:
    registry = _registry_specs(*instruments)
    rows = _multi_instrument_rows(registry)
    proposal_index = next(
        index
        for index, row in enumerate(rows)
        if row["instrument"] == proposal_instrument and row["sequence"] == 1
    )

    def replace_registry(payload: dict) -> None:
        payload["instruments"] = registry

    request, _ = fixture_support.fixture(
        root,
        rows=rows,
        proposal_specs=[(proposal_index, direction)],
        registry_mutation=replace_registry,
    )
    return request, registry


def _reference_artifacts(root: Path, request: dict) -> dict[str, bytes]:
    return {
        label: (root / request[label]["relative_path"]).read_bytes()
        for label in REFERENCE_INPUTS
    }


def _node_market(
    specifications: dict[str, tuple[int, int, int]],
) -> tuple[dict, dict[str, list[dict]], reference.ReferenceInput]:
    registry = {
        instrument: {"price_scale": scale, "pip_ticks": 1}
        for instrument, (_, _, scale) in sorted(specifications.items())
    }
    oracle_books: dict[str, list[dict]] = {}
    reference_books: dict[str, tuple[reference.MarketTick, ...]] = {}
    ticks: list[reference.MarketTick] = []
    for sequence, instrument in enumerate(sorted(specifications), 1):
        bid, ask, scale = specifications[instrument]
        oracle_books[instrument] = [{
            "source_ts_ns": 100,
            "arrival_ts_ns": 100,
            "bid_ticks": bid,
            "ask_ticks": ask,
            "tick_scale": scale,
        }]
        tick = reference.MarketTick(
            provider_id="NODE_HAND_FIXTURE",
            instrument=instrument,
            bid_ticks=bid,
            ask_ticks=ask,
            tick_scale=scale,
            source_ts_ns=100,
            arrival_ts_ns=100,
            sequence=sequence,
            source_event_sha256=f"{sequence:064x}",
            source_prefix_root_sha256=f"{sequence + 16:064x}",
        )
        ticks.append(tick)
        reference_books[instrument] = (tick,)
    data = reference.ReferenceInput(
        ticks=tuple(ticks),
        books=reference_books,
        proposals=(),
        candidate_key="NODE-HAND-FIXTURE",
        provenance={},
        arms={},
        max_trade_quote_staleness_ns=1,
        inventory={},
        accounting={
            "supported_quote_currencies": ["CAD", "CHF", "JPY", "USD"],
            "max_conversion_staleness_ns": 1,
        },
        evaluation={},
        authority={},
        registry=registry,
        execution_policy_sha256="f" * 64,
        raw_hashes={},
    )
    return registry, oracle_books, data


def _reference_position(
    data: reference.ReferenceInput,
    instrument: str,
    direction: int,
    units_micros: int,
) -> reference.PositionLot:
    proposal = reference.Proposal(
        ordinal=1,
        decision_source_ts_ns=99,
        decision_arrival_ts_ns=99,
        decision_source_event_sha256="1" * 64,
        completed_data_watermark_source_ts_ns=99,
        completed_data_prefix_root_sha256="2" * 64,
        instrument=instrument,
        direction=direction,
        target_notional_jpy_micros=1,
        max_age_ns=1,
        worker_key="NODE-HAND-FIXTURE",
    )
    return reference.PositionLot(
        arm="RAW_SIGNAL",
        proposal=proposal,
        signal_id="3" * 64,
        economic_lot_id="4" * 64,
        common={},
        entry=data.books[instrument][0],
        entry_price=Fraction(1, 1),
        entry_price_numerator=1,
        entry_price_denominator=1,
        units_micros=units_micros,
        entry_notional_exact=Fraction(1, 1),
        entry_notional_rounded=1,
        due_arrival_ns=101,
    )


def _hand_node_exposure(
    specifications: dict[str, tuple[int, int, int]],
    lots: list[tuple[str, int, int, Fraction]],
) -> tuple[dict[str, int], dict[str, int]]:
    registry, oracle_books, data = _node_market(specifications)
    oracle_positions = [
        {
            "proposal": {"instrument": instrument, "direction": direction},
            "units_micros": units_micros,
        }
        for instrument, direction, units_micros, _ in lots
    ]
    marks = [{"mark_price": mark_price} for *_, mark_price in lots]
    oracle_exposure = oracle._signed_exposure(
        oracle_positions,
        marks,
        100,
        100,
        oracle_books,
        {"max_conversion_staleness_ns": 1},
        registry,
    )
    reference_positions = [
        _reference_position(data, instrument, direction, units_micros)
        for instrument, direction, units_micros, _ in lots
    ]
    reference_exposure = reference._signed_exposure(
        data,
        reference_positions,
        [{"exit_price": mark_price} for *_, mark_price in lots],
        100,
        100,
    )
    return oracle_exposure, reference_exposure


@pytest.mark.parametrize(
    ("direction", "expected"),
    (
        (1, {"EUR": 100_000_000, "USD": -110_000_000}),
        (-1, {"EUR": -110_000_000, "USD": 100_000_000}),
    ),
)
def test_signed_node_wide_spread_long_short_have_independent_hand_values(
    direction: int,
    expected: dict[str, int],
) -> None:
    actual = _hand_node_exposure(
        {"EUR_USD": (1, 1, 1), "USD_JPY": (100, 110, 1)},
        [("EUR_USD", direction, 1_000_000, Fraction(1, 1))],
    )
    assert actual[0] == expected
    assert actual[1] == expected


@pytest.mark.parametrize(
    ("direction", "mark_price", "expected"),
    (
        pytest.param(
            1,
            Fraction(125, 100),
            {"CAD": -150_020_000, "USD": 150_000_000},
            id="long-inverse-multihop",
        ),
        pytest.param(
            -1,
            Fraction(126, 100),
            {"CAD": 150_000_000, "USD": -150_020_000},
            id="short-inverse-multihop",
        ),
    ),
)
def test_signed_node_inverse_multihop_long_short_have_hand_values(
    direction: int,
    mark_price: Fraction,
    expected: dict[str, int],
) -> None:
    actual = _hand_node_exposure(
        {"USD_CAD": (125, 126, 100), "USD_JPY": (15_000, 15_002, 100)},
        [("USD_CAD", direction, 1_000_000, mark_price)],
    )
    assert actual[0] == expected
    assert actual[1] == expected


def test_signed_nodes_prevent_false_cross_pair_netting_and_round_once() -> None:
    specifications = {
        "EUR_USD": (10_000_001, 10_000_001, 10_000_000),
        "USD_JPY": (1_000_000_001, 1_100_000_000, 10_000_000),
    }
    combined = _hand_node_exposure(
        specifications,
        [
            ("EUR_USD", 1, 1_000_000, Fraction(1, 1)),
            ("USD_JPY", 1, 1_000_000, Fraction(100, 1)),
        ],
    )
    assert combined[0] == {
        "EUR": 100_000_011,
        "JPY": -100_000_000,
        "USD": -10_000_000,
    }
    assert combined[1] == combined[0]

    whole = _hand_node_exposure(
        specifications,
        [("EUR_USD", 1, 1_000_000, Fraction(1, 1))],
    )
    split = _hand_node_exposure(
        specifications,
        [
            ("EUR_USD", 1, 500_000, Fraction(1, 1)),
            ("EUR_USD", 1, 500_000, Fraction(1, 1)),
        ],
    )
    assert whole[0] == whole[1] == split[0] == split[1]
    assert whole[0]["EUR"] == 100_000_011


def _wide_spread_currency_cap_fixture(root: Path, cap: int) -> dict:
    rows = fixture_support.source_rows()
    for row in rows:
        if row["instrument"] == "EUR_USD":
            row["bid_ticks"] = 99_999
            row["ask_ticks"] = 100_001
        else:
            row["bid_ticks"] = 10_000
            row["ask_ticks"] = 11_000

    def zero_execution(policy: dict) -> None:
        for arm, terms in policy["arms"].items():
            terms.update({
                "latency_ns": 0,
                "slippage_micropips_per_side": 0,
                "commission_ppm_per_side": (
                    1 if arm == "ADVERSE_STRESS" else 0
                ),
                "financing_ppm_per_day": 0,
                "raw_mid": arm == "RAW_SIGNAL",
            })

    def currency_cap(inventory: dict) -> None:
        inventory["max_currency_notional_jpy_micros"] = cap

    request, values = fixture_support.fixture(
        root,
        rows=rows,
        proposal_specs=[(0, 1)],
        execution_mutation=zero_execution,
        inventory_mutation=currency_cap,
    )
    proposal = values["proposal"]
    proposal.pop("proposal_sha256")
    proposal["rows"][0]["notional_jpy_micros"] = 110_000_000
    fixture_support.seal(proposal, "proposal_sha256")
    proposal_path = fixture_support.write_json(
        root / "inputs" / "proposal.json", proposal
    )
    request["proposal"] = fixture_support.artifact(
        root, proposal_path, "proposal"
    )
    return request


@pytest.mark.parametrize(
    ("cap", "expected_status"),
    ((105_000_000, "CURRENCY_CAP_REJECTED"), (110_000_000, "FILLED_CLOSED")),
)
def test_wide_spread_currency_cap_rejects_105m_and_admits_110m_in_both_engines(
    tmp_path: Path,
    cap: int,
    expected_status: str,
) -> None:
    request = _wide_spread_currency_cap_fixture(tmp_path, cap)
    oracle_result = fixture_support.run(tmp_path, request)
    oracle_rows = fixture_support.ledger(tmp_path)
    reference_result = reference.replay_reference(
        _reference_artifacts(tmp_path, request)
    )
    reference_rows = [
        json.loads(line)
        for line in reference_result["ledger_bytes"].splitlines()
    ]
    assert {row["status"] for row in oracle_rows} == {expected_status}
    assert {row["status"] for row in reference_rows} == {expected_status}
    assert reference_result["oracle_metrics"] == oracle_result["manifest"][
        "oracle_metrics"
    ]
    if expected_status == "FILLED_CLOSED":
        raw = next(row for row in oracle_rows if row["arm"] == "RAW_SIGNAL")
        assert raw["signed_currency_exposure_after_entry_jpy_micros"] == {
            "EUR": 99_999_000,
            "USD": -110_000_000,
        }
        assert raw["gross_open_notional_after_entry_jpy_micros"] \
            == 100_000_000
        assert all(
            max(abs(value) for value in row[
                "signed_currency_exposure_after_entry_jpy_micros"
            ].values()) <= cap
            for row in oracle_rows
        )


def _currency_cap_mark_breach_fixture(root: Path) -> tuple[dict, int, int]:
    rows = fixture_support.source_rows()
    for row in rows:
        if row["instrument"] != "USD_JPY":
            continue
        if row["sequence"] < 4:
            row["bid_ticks"] = 9_999
            row["ask_ticks"] = 10_001
        else:
            row["bid_ticks"] = 11_999
            row["ask_ticks"] = 12_001
    first_decision = next(
        index
        for index, row in enumerate(rows)
        if row["instrument"] == "USD_JPY" and row["sequence"] == 1
    )
    later_decision = next(
        index
        for index, row in enumerate(rows)
        if row["instrument"] == "USD_JPY" and row["sequence"] == 4
    )
    move_arrival_ns = next(
        row["arrival_ts_ns"]
        for row in rows
        if row["instrument"] == "USD_JPY" and row["sequence"] == 4
    )
    currency_cap = 30_000_000_000

    def freeze_currency_cap(inventory: dict) -> None:
        inventory["max_currency_notional_jpy_micros"] = currency_cap

    request, _ = fixture_support.fixture(
        root,
        rows=rows,
        proposal_specs=[(first_decision, 1), (later_decision, 1)],
        inventory_mutation=freeze_currency_cap,
        proposal_max_age_seconds=600,
    )
    return request, currency_cap, move_arrival_ns


def test_causal_mark_currency_cap_breach_closes_and_halts_in_both_engines(
    tmp_path: Path,
) -> None:
    request, currency_cap, move_arrival_ns = _currency_cap_mark_breach_fixture(
        tmp_path
    )
    oracle_result = fixture_support.run(tmp_path, request)
    oracle_rows = fixture_support.ledger(tmp_path)
    reference_result = reference.replay_reference(
        _reference_artifacts(tmp_path, request)
    )
    reference_rows = [
        json.loads(line)
        for line in reference_result["ledger_bytes"].splitlines()
    ]

    assert reference_result["ledger_bytes"] == (
        fixture_support.oracle_output_root(tmp_path)
        / "oracle_output"
        / "oracle_ledger.jsonl"
    ).read_bytes()
    assert reference_result["oracle_metrics"] == oracle_result["manifest"][
        "oracle_metrics"
    ]
    for rows, metrics_by_arm in (
        (oracle_rows, oracle_result["manifest"]["oracle_metrics"]["arms"]),
        (reference_rows, reference_result["oracle_metrics"]["arms"]),
    ):
        for arm in oracle.ARMS:
            arm_rows = sorted(
                (row for row in rows if row["arm"] == arm),
                key=lambda row: row["proposal_ordinal"],
            )
            assert len(arm_rows) == 2
            opened, blocked = arm_rows
            assert max(abs(value) for value in opened[
                "signed_currency_exposure_after_entry_jpy_micros"
            ].values()) <= currency_cap
            assert opened["exit_disposition"] == "INVENTORY_CAP_CLOSEOUT"
            assert opened["exit_arrival_ts_ns"] == move_arrival_ns
            assert blocked["status"] == "ACCOUNT_HALTED"
            metrics = metrics_by_arm[arm]
            assert metrics["max_gross_notional_jpy_micros"] > currency_cap
            assert metrics["max_gross_notional_jpy_micros"] < 200_000_000_000
            assert metrics["margin_guard_pass"] is False
            assert metrics["terminal_open_positions"] == 0


def _cluster_record(
    *,
    arrival_ns: int,
    exact_pnl: Fraction,
    rounded_ledger_pnl: int,
    economic_lot_id: str,
    instrument: str = "EUR_USD",
) -> dict[str, object]:
    """Build only the independently observable inputs to cluster accounting."""
    return {
        "status": "FILLED_CLOSED",
        "instrument": instrument,
        "entry_arrival_ts_ns": arrival_ns,
        "signal_id": economic_lot_id,
        "economic_lot_id": economic_lot_id,
        "net_pnl_jpy_micros": rounded_ledger_pnl,
        "economic_net_pnl_jpy_micros_numerator": exact_pnl.numerator,
        "economic_net_pnl_jpy_micros_denominator": exact_pnl.denominator,
    }


def test_opening_liability_and_liquidation_asset_use_opposite_conversion_sides() -> None:
    # A deliberately wide USD/JPY quote makes BID-vs-ASK use observable without
    # relying on any producer metric.  One EUR at EUR/USD=1 creates a USD cash
    # liability for a long entry and a USD asset at long liquidation; the short
    # side has the exact inverse cash signs.
    usd_jpy = {
        "source_ts_ns": 10,
        "arrival_ts_ns": 10,
        "bid_ticks": 100,
        "ask_ticks": 110,
        "tick_scale": 1,
    }
    books = {"USD_JPY": [usd_jpy]}
    accounting = {"max_conversion_staleness_ns": 1}
    registry = {"USD_JPY": {"price_scale": 1, "pip_ticks": 1}}
    one_unit = oracle.BASE_MICROUNITS_PER_UNIT

    def notional(direction: int, opening: bool) -> Fraction:
        return oracle._position_notional_exact_jpy_micros(
            direction=direction,
            units_micros=one_unit,
            price=Fraction(1, 1),
            quote_currency="USD",
            source_watermark_ns=10,
            arrival_cutoff_ns=10,
            books=books,
            accounting=accounting,
            registry=registry,
            opening=opening,
        )

    assert notional(1, True) == 110 * oracle.JPY_MICROS_PER_YEN
    assert notional(1, False) == 100 * oracle.JPY_MICROS_PER_YEN
    assert notional(-1, True) == 100 * oracle.JPY_MICROS_PER_YEN
    assert notional(-1, False) == 110 * oracle.JPY_MICROS_PER_YEN

    entry = {"source_ts_ns": 10, "arrival_ts_ns": 10}
    target = 110 * oracle.JPY_MICROS_PER_YEN
    long_units = oracle._units_for_actual_entry(
        {"instrument": "EUR_USD", "direction": 1, "notional_jpy_micros": target},
        entry,
        Fraction(1, 1),
        books,
        accounting,
        registry,
    )
    short_units = oracle._units_for_actual_entry(
        {"instrument": "EUR_USD", "direction": -1, "notional_jpy_micros": target},
        entry,
        Fraction(1, 1),
        books,
        accounting,
        registry,
    )
    assert long_units == 1_000_000
    assert short_units == 1_100_000


@pytest.mark.parametrize(
    ("currency", "bridge", "positive_amount", "positive_expected", "negative_amount", "negative_expected"),
    [
        ("CAD", "USD_CAD", 126, 15_000, -125, -15_002),
        ("CHF", "USD_CHF", 91, 15_000, -90, -15_002),
    ],
)
def test_two_hop_assets_and_liabilities_use_exact_side_at_every_edge(
    currency: str,
    bridge: str,
    positive_amount: int,
    positive_expected: int,
    negative_amount: int,
    negative_expected: int,
) -> None:
    bridge_quote = (
        _quote(125, 126, 100)
        if currency == "CAD"
        else _quote(90, 91, 100)
    )
    books = {
        bridge: [bridge_quote],
        "USD_JPY": [_quote(15_000, 15_002, 100)],
    }
    registry = {
        bridge: {"price_scale": 100, "pip_ticks": 1},
        "USD_JPY": {"price_scale": 100, "pip_ticks": 1},
    }
    assert oracle._convert_to_jpy(
        Fraction(positive_amount),
        currency,
        100,
        100,
        books,
        1,
        registry=registry,
    ) == positive_expected
    assert oracle._convert_to_jpy(
        Fraction(negative_amount),
        currency,
        100,
        100,
        books,
        1,
        registry=registry,
    ) == negative_expected


@pytest.mark.parametrize("instrument,currency", [("CAD_JPY", "CAD"), ("CHF_JPY", "CHF")])
def test_direct_registry_path_assets_and_liabilities(
    instrument: str,
    currency: str,
) -> None:
    books = {instrument: [_quote(11_000, 11_010, 100)]}
    registry = {instrument: {"price_scale": 100, "pip_ticks": 1}}
    assert oracle._convert_to_jpy(
        Fraction(1), currency, 100, 100, books, 1, registry=registry,
    ) == 110
    assert oracle._convert_to_jpy(
        Fraction(-1), currency, 100, 100, books, 1, registry=registry,
    ) == Fraction(-1_101, 10)


@pytest.mark.parametrize("instrument,currency", [("JPY_CAD", "CAD"), ("JPY_CHF", "CHF")])
def test_inverted_direct_registry_path_divides_with_exact_side(
    instrument: str,
    currency: str,
) -> None:
    books = {instrument: [_quote(2, 4, 100)]}
    registry = {instrument: {"price_scale": 100, "pip_ticks": 1}}
    assert oracle._convert_to_jpy(
        Fraction(4), currency, 100, 100, books, 1, registry=registry,
    ) == 100
    assert oracle._convert_to_jpy(
        Fraction(-2), currency, 100, 100, books, 1, registry=registry,
    ) == -100


def test_unique_three_hop_non_usd_first_leg_is_supported() -> None:
    books = {
        "CAD_CHF": [_quote(150, 200, 100)],
        "USD_CHF": [_quote(50, 75, 100)],
        "USD_JPY": [_quote(500, 600, 100)],
    }
    registry = {
        instrument: {"price_scale": 100, "pip_ticks": 1}
        for instrument in sorted(books)
    }
    assert oracle._convert_to_jpy(
        Fraction(4), "CAD", 100, 100, books, 1, registry=registry,
    ) == 40
    assert oracle._convert_to_jpy(
        Fraction(-5), "CAD", 100, 100, books, 1, registry=registry,
    ) == -120


def test_fresh_direct_and_fresh_two_hop_triangle_is_ambiguous() -> None:
    books = {
        "CAD_JPY": [_quote(11_000, 11_010, 100)],
        "USD_CAD": [_quote(125, 126, 100)],
        "USD_JPY": [_quote(15_000, 15_002, 100)],
    }
    registry = {
        instrument: {"price_scale": 100, "pip_ticks": 1}
        for instrument in sorted(books)
    }
    for amount in (Fraction(126), Fraction(-125)):
        with pytest.raises(
            oracle.OracleError,
            match="JPY conversion path must be uniquely causal",
        ):
            oracle._convert_to_jpy(
                amount, "CAD", 100, 100, books, 1, registry=registry,
            )


@pytest.mark.parametrize(
    ("alternate_source", "alternate_arrival"),
    [(1, 1), (101, 101)],
)
def test_stale_or_future_alternate_is_not_counted_as_ambiguity(
    alternate_source: int,
    alternate_arrival: int,
) -> None:
    books = {
        "CAD_JPY": [_quote(11_000, 11_010, 100)],
        "USD_CAD": [_quote(
            125, 126, 100,
            source_ts_ns=alternate_source,
            arrival_ts_ns=alternate_arrival,
        )],
        "USD_JPY": [_quote(
            15_000, 15_002, 100,
            source_ts_ns=alternate_source,
            arrival_ts_ns=alternate_arrival,
        )],
    }
    registry = {
        instrument: {"price_scale": 100, "pip_ticks": 1}
        for instrument in sorted(books)
    }
    assert oracle._convert_to_jpy(
        Fraction(1), "CAD", 100, 100, books, 10, registry=registry,
    ) == 110


@pytest.mark.parametrize(
    ("direct_source", "direct_arrival"),
    [(1, 1), (101, 101)],
)
def test_stale_or_future_direct_leaves_unique_fresh_two_hop_route(
    direct_source: int,
    direct_arrival: int,
) -> None:
    books = {
        "CAD_JPY": [_quote(
            11_000, 11_010, 100,
            source_ts_ns=direct_source,
            arrival_ts_ns=direct_arrival,
        )],
        "USD_CAD": [_quote(125, 126, 100)],
        "USD_JPY": [_quote(15_000, 15_002, 100)],
    }
    registry = {
        instrument: {"price_scale": 100, "pip_ticks": 1}
        for instrument in sorted(books)
    }
    assert oracle._convert_to_jpy(
        Fraction(126), "CAD", 100, 100, books, 10, registry=registry,
    ) == 15_000


def test_no_fresh_or_reachable_path_and_cycle_fail_closed() -> None:
    no_jpy_books = {
        "CAD_CHF": [_quote(65, 66, 100)],
        "CHF_USD": [_quote(90, 91, 100)],
        "USD_CAD": [_quote(125, 126, 100)],
    }
    no_jpy_registry = {
        instrument: {"price_scale": 100, "pip_ticks": 1}
        for instrument in sorted(no_jpy_books)
    }
    with pytest.raises(oracle.OracleError, match="uniquely causal"):
        oracle._convert_to_jpy(
            Fraction(1), "CAD", 100, 100, no_jpy_books, 1,
            registry=no_jpy_registry,
        )

    two_route_books = {
        **no_jpy_books,
        "USD_JPY": [_quote(15_000, 15_002, 100)],
    }
    two_route_registry = {
        instrument: {"price_scale": 100, "pip_ticks": 1}
        for instrument in sorted(two_route_books)
    }
    with pytest.raises(oracle.OracleError, match="uniquely causal"):
        oracle._convert_to_jpy(
            Fraction(1), "CAD", 100, 100, two_route_books, 1,
            registry=two_route_registry,
        )


@pytest.mark.parametrize(
    ("instruments", "proposal_instrument"),
    [
        (("CHF_JPY", "EUR_CHF"), "EUR_CHF"),
        (("CAD_CHF", "CHF_JPY", "EUR_CAD"), "EUR_CAD"),
    ],
)
def test_general_direct_and_non_usd_bridge_replay_match_frozen_reference(
    tmp_path: Path,
    instruments: tuple[str, ...],
    proposal_instrument: str,
) -> None:
    request, _ = _custom_market_fixture(
        tmp_path,
        instruments,
        proposal_instrument,
    )
    oracle_result = fixture_support.run(tmp_path, request)
    oracle_ledger = (
        fixture_support.oracle_output_root(tmp_path)
        / "oracle_output"
        / "oracle_ledger.jsonl"
    ).read_bytes()
    reference_result = reference.replay_reference(
        _reference_artifacts(tmp_path, request)
    )
    assert len(oracle_ledger.splitlines()) == 3
    assert reference_result["ledger_bytes"] == oracle_ledger
    assert reference_result["oracle_metrics"] == oracle_result["manifest"][
        "oracle_metrics"
    ]


def test_ambiguous_cad_triangle_rejects_in_oracle_and_frozen_reference(
    tmp_path: Path,
) -> None:
    request, _ = _custom_market_fixture(
        tmp_path,
        ("CAD_JPY", "EUR_USD", "USD_CAD", "USD_JPY"),
        "USD_CAD",
    )
    artifacts = _reference_artifacts(tmp_path, request)
    with pytest.raises(oracle.OracleError, match="uniquely causal"):
        fixture_support.run(tmp_path, request)
    with pytest.raises(reference.ReferenceError, match="uniquely causal"):
        reference.replay_reference(artifacts)


def test_entry_caps_use_actual_fill_and_mark_not_target_notional(tmp_path: Path) -> None:
    # The frozen target is 28,000 JPY, while outward-rounded executable mark
    # risk is slightly smaller after integer-unit sizing.  A cap between them
    # must admit the trade only if the engine uses actual causal notional.
    cap = 27_999_999_960

    def tighten_to_actual_notional(inventory: dict) -> None:
        inventory["max_gross_notional_jpy_micros"] = cap
        inventory["max_currency_notional_jpy_micros"] = cap

    request, _ = fixture_support.fixture(
        tmp_path,
        proposal_specs=[(0, 1)],
        inventory_mutation=tighten_to_actual_notional,
    )
    fixture_support.run(tmp_path, request)
    records = fixture_support.ledger(tmp_path)

    assert {record["status"] for record in records} == {"FILLED_CLOSED"}
    for record in records:
        assert record["target_notional_jpy_micros"] == 28_000_000_000
        assert record["target_notional_jpy_micros"] > cap
        assert record["filled_notional_jpy_micros"] <= record["target_notional_jpy_micros"]
        assert record["gross_open_notional_after_entry_jpy_micros"] <= cap
        exposure = record["signed_currency_exposure_after_entry_jpy_micros"]
        assert max(abs(value) for value in exposure.values()) <= cap
        assert set(exposure) == {"EUR", "USD"}


@pytest.mark.parametrize(
    ("gate", "expected_status"),
    [
        ("gross", "GROSS_CAP_REJECTED"),
        ("currency", "CURRENCY_CAP_REJECTED"),
        ("margin", "MARGIN_ENTRY_REJECTED"),
    ],
)
def test_short_entry_caps_use_same_clock_executable_liquidation_mark(
    tmp_path: Path, gate: str, expected_status: str
) -> None:
    # A short USD/JPY fill receives BID-side JPY at entry but requires the
    # larger ASK-side JPY liability for immediate liquidation.  The common
    # target is exactly the cap, integer-unit entry notionals are just below it,
    # and both executable-arm same-clock liquidation marks are above it.  Each
    # risk gate must therefore reject those arms rather than admit on filled
    # notional; midpoint RAW_SIGNAL remains the control.
    cap = 28_000_000_000

    def tighten_inventory(inventory: dict) -> None:
        if gate == "gross":
            inventory["max_gross_notional_jpy_micros"] = cap
        elif gate == "currency":
            inventory["max_currency_notional_jpy_micros"] = cap

    def tighten_margin(evaluation: dict) -> None:
        if gate == "margin":
            evaluation["margin_notional_cap_jpy_micros"] = cap

    request, _ = fixture_support.fixture(
        tmp_path,
        proposal_specs=[(1, -1)],
        inventory_mutation=tighten_inventory,
        evaluation_mutation=tighten_margin,
    )
    fixture_support.run(tmp_path, request)
    records = fixture_support.ledger(tmp_path)

    assert len(records) == len(oracle.ARMS)
    by_arm = {record["arm"]: record for record in records}
    # RAW_SIGNAL's marked-notional scalar uses midpoint, so its gross and margin
    # controls remain the control.  Its negative USD currency node is still an
    # independently executable liability and therefore uses USD/JPY ASK; the
    # corrected currency cap rejects all three arms.
    assert by_arm["RAW_SIGNAL"]["status"] == (
        expected_status if gate == "currency" else "FILLED_CLOSED"
    )
    for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS"):
        assert by_arm[arm]["status"] == expected_status
        assert by_arm[arm]["target_notional_jpy_micros"] == cap
        assert by_arm[arm]["filled_notional_jpy_micros"] == 0


def test_financing_uses_fill_arrival_elapsed_not_source_clock_elapsed() -> None:
    # Source clocks are 301 seconds apart, but these delayed quotes arrive only
    # two seconds apart.  Financing must charge the two-second holding period.
    entry = {
        "source_ts_ns": 1_000_000_000_000,
        "arrival_ts_ns": 10_000_000_000_000,
        "bid_ticks": 10_000,
        "ask_ticks": 10_002,
        "tick_scale": 100,
    }
    mark = {
        "source_ts_ns": 1_301_000_000_000,
        "arrival_ts_ns": 10_002_000_000_000,
        "bid_ticks": 10_010,
        "ask_ticks": 10_012,
        "tick_scale": 100,
    }
    entry_notional = Fraction(100_020_000, 1)
    position = {
        "proposal": {"instrument": "USD_JPY", "direction": 1},
        "policy": {
            "raw_mid": False,
            "slippage_micropips_per_side": 0,
            "commission_ppm_per_side": 0,
            "financing_ppm_per_day": 1_000_000,
        },
        "entry": entry,
        "entry_price": Fraction(10_002, 100),
        "units_micros": oracle.BASE_MICROUNITS_PER_UNIT,
        "entry_notional_exact_jpy_micros": entry_notional,
    }
    values = oracle._position_value(
        position,
        mark,
        {"USD_JPY": [entry, mark]},
        {"max_conversion_staleness_ns": 10_000_000_000_000},
        {"USD_JPY": {"price_scale": 100, "pip_ticks": 1}},
    )

    arrival_elapsed_ns = 2_000_000_000
    source_elapsed_ns = 301_000_000_000
    expected = (
        entry_notional.numerator * arrival_elapsed_ns
        + oracle.DAY_NS * entry_notional.denominator - 1
    ) // (oracle.DAY_NS * entry_notional.denominator)
    wrong_source_clock_charge = (
        entry_notional.numerator * source_elapsed_ns
        + oracle.DAY_NS * entry_notional.denominator - 1
    ) // (oracle.DAY_NS * entry_notional.denominator)

    assert values["elapsed_ns"] == arrival_elapsed_ns
    assert values["financing_jpy_micros"] == expected
    assert values["financing_jpy_micros"] != wrong_source_clock_charge


def test_cluster_returns_are_signed_for_loss_flat_and_gain() -> None:
    evaluation = {
        "cluster_window_ns": 1_000,
        "initial_equity_jpy_micros": 1_000,
        "cvar_tail_bps": 5_000,
    }
    records = [
        _cluster_record(
            arrival_ns=1,
            exact_pnl=Fraction(-100, 1),
            rounded_ledger_pnl=-100,
            economic_lot_id="a" * 64,
        ),
        _cluster_record(
            arrival_ns=1_001,
            exact_pnl=Fraction(0, 1),
            rounded_ledger_pnl=0,
            economic_lot_id="b" * 64,
        ),
        _cluster_record(
            arrival_ns=2_001,
            exact_pnl=Fraction(100, 1),
            rounded_ledger_pnl=100,
            economic_lot_id="c" * 64,
        ),
    ]
    n_eff, _, _, observations = oracle._cluster_metrics(records, evaluation)

    assert n_eff == 3
    by_bucket = {item["time_bucket"]: item for item in observations}
    assert {
        bucket: item["cluster_risk_net_pnl_jpy_micros"]
        for bucket, item in by_bucket.items()
    } == {0: -100, 1: 0, 2: 100}
    assert {
        bucket: item["signed_return"]
        for bucket, item in by_bucket.items()
    } == {
        0: "-0.100000000000000000",
        1: "0.000000000000000000",
        2: "0.100000000000000000",
    }
    assert [item["cluster_id"] for item in observations] == sorted(
        item["cluster_id"] for item in observations
    )


def test_exact_cluster_cvar_is_ticket_partition_invariant_despite_ledger_rounding() -> None:
    evaluation = {
        "cluster_window_ns": 1_000,
        "initial_equity_jpy_micros": 1_000,
        "cvar_tail_bps": 5_000,
    }
    loss_lot = "a" * 64
    profit_lot = "b" * 64
    profit = _cluster_record(
        arrival_ns=1_001,
        exact_pnl=Fraction(2, 1),
        rounded_ledger_pnl=2,
        economic_lot_id=profit_lot,
    )
    whole = [
        _cluster_record(
            arrival_ns=1,
            exact_pnl=Fraction(-1, 2),
            rounded_ledger_pnl=-1,
            economic_lot_id=loss_lot,
        ),
        profit,
    ]
    partitioned = [
        _cluster_record(
            arrival_ns=1,
            exact_pnl=Fraction(-1, 4),
            rounded_ledger_pnl=-1,
            economic_lot_id=loss_lot,
        ),
        _cluster_record(
            arrival_ns=1,
            exact_pnl=Fraction(-1, 4),
            rounded_ledger_pnl=-1,
            economic_lot_id=loss_lot,
        ),
        profit,
    ]

    assert sum(row["net_pnl_jpy_micros"] for row in whole) == 1
    assert sum(row["net_pnl_jpy_micros"] for row in partitioned) == 0
    whole_result = oracle._cluster_metrics(whole, evaluation)
    partitioned_result = oracle._cluster_metrics(partitioned, evaluation)
    assert whole_result[:3] == partitioned_result[:3]
    assert whole_result[0] == 2
    assert whole_result[1] == -1
    assert whole_result[2] == "-0.000500000000000000"
    whole_observations = whole_result[3]
    partitioned_observations = partitioned_result[3]
    assert [
        {
            key: item[key]
            for key in (
                "cluster_id",
                "time_bucket",
                "currency_nodes",
                "source_signal_set_sha256",
                "cluster_risk_net_pnl_jpy_micros",
                "signed_return",
            )
        }
        for item in whole_observations
    ] == [
        {
            key: item[key]
            for key in (
                "cluster_id",
                "time_bucket",
                "currency_nodes",
                "source_signal_set_sha256",
                "cluster_risk_net_pnl_jpy_micros",
                "signed_return",
            )
        }
        for item in partitioned_observations
    ]
    whole_loss = next(
        item for item in whole_observations if item["time_bucket"] == 0
    )
    partitioned_loss = next(
        item for item in partitioned_observations if item["time_bucket"] == 0
    )
    assert whole_loss["ledger_net_pnl_jpy_micros"] == -1
    assert partitioned_loss["ledger_net_pnl_jpy_micros"] == -2


@pytest.mark.parametrize(
    ("field", "confusable_value"),
    [
        ("paper_only", 1),
        ("live_authority", 0),
        ("external_orders", False),
    ],
)
def test_authority_rejects_bool_int_equality_confusion(
    tmp_path: Path, field: str, confusable_value: object
) -> None:
    request, values = fixture_support.fixture(tmp_path)
    authority = dict(values["authority_policy"])
    authority[field] = confusable_value
    fixture_support.seal(authority, "authority_policy_sha256")
    path = fixture_support.write_json(
        tmp_path / "inputs" / "authority_policy.json", authority
    )
    request["authority_policy"] = fixture_support.artifact(
        tmp_path, path, "authority_policy"
    )

    with pytest.raises(oracle.OracleError, match="paper authority.*mismatch"):
        fixture_support.run(tmp_path, request)


def test_same_pair_cannot_net_but_cross_pair_shared_currency_can(tmp_path: Path) -> None:
    collision_root = tmp_path / "same-pair"
    collision_request, _ = fixture_support.fixture(
        collision_root,
        # First EUR/USD position remains open when the opposite proposal fills.
        proposal_specs=[(0, 1), (2, -1)],
    )
    fixture_support.run(collision_root, collision_request)
    collision_rows = fixture_support.ledger(collision_root)
    for arm in oracle.ARMS:
        rows = [row for row in collision_rows if row["arm"] == arm]
        assert [row["status"] for row in rows] == [
            "FILLED_CLOSED",
            "SAME_PAIR_COLLISION_REJECTED",
        ]

    cross_root = tmp_path / "cross-pair"
    cross_request, _ = fixture_support.fixture(
        cross_root,
        # EUR/USD long contributes -USD; USD/JPY long contributes +USD.
        proposal_specs=[(0, 1), (1, 1)],
    )
    fixture_support.run(cross_root, cross_request)
    cross_rows = fixture_support.ledger(cross_root)
    for arm in oracle.ARMS:
        rows = [row for row in cross_rows if row["arm"] == arm]
        assert [row["status"] for row in rows] == ["FILLED_CLOSED", "FILLED_CLOSED"]
        exposure = rows[1]["signed_currency_exposure_after_entry_jpy_micros"]
        assert set(exposure) == {"EUR", "JPY", "USD"}
        assert abs(exposure["USD"]) < min(abs(exposure["EUR"]), abs(exposure["JPY"]))

def _adverse_below_base(field: str) -> Callable[[dict], None]:
    def mutate(policy: dict) -> None:
        base_value = policy["arms"]["EXECUTABLE_BASE"][field]
        policy["arms"]["ADVERSE_STRESS"][field] = base_value - 1

    return mutate


@pytest.mark.parametrize(
    "mutator",
    [
        _adverse_below_base("latency_ns"),
        _adverse_below_base("slippage_micropips_per_side"),
        _adverse_below_base("commission_ppm_per_side"),
        _adverse_below_base("financing_ppm_per_day"),
        lambda policy: policy["arms"].__setitem__(
            "ADVERSE_STRESS", dict(policy["arms"]["EXECUTABLE_BASE"])
        ),
    ],
    ids=["latency", "slippage", "commission", "financing", "identical"],
)
def test_adverse_policy_is_weakly_worse_in_every_cost_and_strictly_worse_in_one(
    tmp_path: Path, mutator: Callable[[dict], None]
) -> None:
    request, _ = fixture_support.fixture(tmp_path, execution_mutation=mutator)
    with pytest.raises(oracle.OracleError, match="ADVERSE"):
        fixture_support.run(tmp_path, request)
