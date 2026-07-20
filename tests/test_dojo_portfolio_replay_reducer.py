from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from quant_rabbit.dojo_portfolio_replay_reducer import (
    DojoPortfolioReplayError,
    admit_worker_proposals,
    canonical_portfolio_sha256,
    quote_batch_sha256,
    reduce_portfolio_replay,
    seal_portfolio_policy,
    validate_portfolio_replay_result,
)
from quant_rabbit.dojo_shared_worker_protocol import (
    ProtocolViolation,
    seal_post_exit_snapshot,
    seal_worker_proposal,
    seal_worker_proposal_batch,
)


CONFIG_SHA = "c" * 64
WORKERS = [
    {
        "worker_id": "worker-a",
        "owner_id": "owner-a",
        "family_id": "family-a",
        "config_sha256": CONFIG_SHA,
    },
    {
        "worker_id": "worker-b",
        "owner_id": "owner-b",
        "family_id": "family-b",
        "config_sha256": CONFIG_SHA,
    },
]


def make_policy(
    *,
    pairs: tuple[str, ...] = ("USD_JPY",),
    max_total: int = 8,
    max_cluster_fraction: float = 100.0,
    financing_rate: float = 0.0,
    max_pair: int = 8,
    max_family: int = 8,
    max_currency_fraction: float = 100.0,
    max_stop_fraction: float = 0.8,
    max_margin_fraction: float = 0.8,
    max_lock_seconds: int = 86_400,
) -> dict:
    conversion_routes = []
    if "EUR_USD" in pairs:
        conversion_routes.append(
            {"currency": "USD", "pair": "USD_JPY", "orientation": "JPY_PER_CURRENCY"}
        )
    return seal_portfolio_policy(
        {
            "policy_id": "focused-test-policy",
            "expected_quote_pairs": list(pairs),
            "active_worker_bindings": deepcopy(WORKERS),
            "leverage": 20,
            "margin_closeout_fraction": 0.9,
            "max_margin_utilization_fraction": max_margin_fraction,
            "max_portfolio_stop_risk_fraction": max_stop_fraction,
            "max_open_and_pending_total": max_total,
            "max_open_and_pending_per_pair": max_pair,
            "max_open_and_pending_per_family": max_family,
            "max_currency_gross_notional_fraction": max_currency_fraction,
            "max_cluster_gross_notional_fraction": max_cluster_fraction,
            "max_lock_seconds": max_lock_seconds,
            "slippage_by_pair": [
                {
                    "pair": pair,
                    "entry_slippage_price": 0.01,
                    "exit_slippage_price": 0.02,
                }
                for pair in pairs
            ],
            "financing_by_pair": [
                {
                    "pair": pair,
                    "long_cost_jpy_per_unit_day": financing_rate,
                    "short_cost_jpy_per_unit_day": financing_rate,
                }
                for pair in pairs
            ],
            "conversion_routes": conversion_routes,
            "correlation_bindings": [],
        }
    )


def make_quotes(
    epoch: int, phase: str, prices: dict[str, tuple[float, float]]
) -> list[dict]:
    prefix = datetime.fromtimestamp(epoch, timezone.utc).isoformat()
    return [
        {
            "pair": pair,
            "bid": float(bid),
            "ask": float(ask),
            "timestamp": f"{prefix}#{phase}",
        }
        for pair, (bid, ask) in sorted(prices.items())
    ]


def make_snapshot(
    *,
    epoch: int,
    phase: str = "O",
    intrabar: str = "OHLC",
    watermark: int = 1,
    prices: dict[str, tuple[float, float]] | None = None,
    balance: float = 200_000.0,
    equity: float | None = None,
    margin: float = 0.0,
    financing: float = 0.0,
    positions: list[dict] | None = None,
    pending_orders: list[dict] | None = None,
) -> dict:
    if prices is None:
        prices = {"USD_JPY": (145.0, 145.02)}
    quotes = make_quotes(epoch, phase, prices)
    digest = quote_batch_sha256(
        epoch=epoch,
        phase=phase,
        intrabar=intrabar,
        quote_watermark=watermark,
        quotes=quotes,
    )
    return seal_post_exit_snapshot(
        {
            "coordinate_id": f"{epoch}:{phase}",
            "epoch": epoch,
            "phase": phase,
            "intrabar": intrabar,
            "quote_batch_sha256": digest,
            "quote_watermark": watermark,
            "expected_quote_pairs": sorted(prices),
            "active_worker_bindings": deepcopy(WORKERS),
            "account": {
                "balance_jpy": balance,
                "equity_jpy": balance if equity is None else equity,
                "margin_used_jpy": margin,
                "accrued_financing_jpy": financing,
            },
            "quotes": quotes,
            "positions": [] if positions is None else positions,
            "pending_orders": [] if pending_orders is None else pending_orders,
        }
    )


def make_intent(
    *,
    intent_id: str,
    action: str = "MARKET",
    pair: str = "USD_JPY",
    side: str = "LONG",
    units: float = 100.0,
    entry_price: float = 145.02,
    sl_price: float = 144.5,
    tp_price: float | None = 146.0,
    valid_until_epoch: int,
    edge: float = 1.0,
    hard_max_holding_seconds: int = 3600,
) -> dict:
    return {
        "intent_id": intent_id,
        "action": action,
        "parameters": {
            "pair": pair,
            "side": side,
            "units": units,
            "entry_price": entry_price,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "stress_cost_pips": 999_999.0,
            "hard_max_holding_seconds": hard_max_holding_seconds,
            "valid_until_epoch": valid_until_epoch,
            "expected_net_edge_jpy": edge,
        },
        "reason_code": "TEST_ONLY",
    }


def make_batch(
    snapshot: dict,
    *,
    new_by_worker: dict[str, list[dict]] | None = None,
    reducing_by_worker: dict[str, list[dict]] | None = None,
    reverse: bool = False,
) -> dict:
    new_by_worker = {} if new_by_worker is None else new_by_worker
    reducing_by_worker = {} if reducing_by_worker is None else reducing_by_worker
    proposals = []
    for binding in WORKERS:
        proposal = {
            **binding,
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "risk_reducing_intents": deepcopy(
                reducing_by_worker.get(binding["worker_id"], [])
            ),
            "new_risk_intents": deepcopy(new_by_worker.get(binding["worker_id"], [])),
        }
        proposals.append(seal_worker_proposal(snapshot, proposal))
    if reverse:
        proposals.reverse()
    return seal_worker_proposal_batch(snapshot, proposals)


def project_pending(carry: dict) -> list[dict]:
    keys = (
        "order_id",
        "worker_id",
        "owner_id",
        "family_id",
        "pair",
        "side",
        "order_kind",
        "units",
        "trigger_price",
        "tp_price",
        "sl_price",
        "created_epoch",
        "valid_until_epoch",
    )
    return [{key: row[key] for key in keys} for row in carry["pending_orders"]]


def test_admission_is_input_order_independent_and_ignores_claimed_edge() -> None:
    epoch = 1_704_067_200
    snapshot = make_snapshot(epoch=epoch)
    policy = make_policy(max_total=1)
    intents = {
        "worker-a": [
            make_intent(
                intent_id="large-high-claim",
                units=500,
                valid_until_epoch=epoch,
                edge=10_000_000,
            )
        ],
        "worker-b": [
            make_intent(
                intent_id="small-low-claim",
                units=100,
                valid_until_epoch=epoch,
                edge=-10_000_000,
            )
        ],
    }
    first = admit_worker_proposals(
        snapshot, make_batch(snapshot, new_by_worker=intents), policy
    )
    second = admit_worker_proposals(
        snapshot, make_batch(snapshot, new_by_worker=intents, reverse=True), policy
    )

    assert first == second
    assert [row["intent_id"] for row in first["accepted"]] == ["small-low-claim"]
    assert first["rejected"][0]["reason"] == "GLOBAL_COUNT_CAP"
    assert first["ranking_policy"].endswith("NO_EDGE_CLAIM")


@pytest.mark.parametrize(
    ("policy_kwargs", "intent_mode", "expected_reason"),
    [
        ({"max_pair": 1}, "two-workers", "PAIR_COUNT_CAP"),
        ({"max_family": 1}, "same-worker", "FAMILY_COUNT_CAP"),
        ({"max_currency_fraction": 0.05}, "one", "CURRENCY_GROSS_CAP"),
        ({"max_cluster_fraction": 0.1}, "two-workers", "CORRELATION_CLUSTER_CAP"),
        ({"max_stop_fraction": 0.0001}, "one", "STOP_RISK_CAP"),
        ({"max_margin_fraction": 0.003}, "one", "MARGIN_CAP"),
        ({"max_lock_seconds": 1800}, "one", "LOCK_CAP"),
        ({"max_lock_seconds": 4000}, "pending-too-long", "PENDING_LOCK_CAP"),
    ],
)
def test_every_declarative_portfolio_cap_has_an_explicit_rejection(
    policy_kwargs: dict, intent_mode: str, expected_reason: str
) -> None:
    epoch = 1_704_067_200
    snapshot = make_snapshot(epoch=epoch)
    first = make_intent(intent_id="first", units=100, valid_until_epoch=epoch)
    if intent_mode == "pending-too-long":
        first = make_intent(
            intent_id="first",
            action="LIMIT",
            units=100,
            entry_price=144.5,
            sl_price=144.0,
            valid_until_epoch=epoch + 5000,
        )
    new_by_worker = {"worker-a": [first]}
    if intent_mode == "two-workers":
        new_by_worker["worker-b"] = [
            make_intent(intent_id="second", units=100, valid_until_epoch=epoch)
        ]
    elif intent_mode == "same-worker":
        new_by_worker["worker-a"].append(
            make_intent(intent_id="second", units=100, valid_until_epoch=epoch)
        )

    decision = admit_worker_proposals(
        snapshot,
        make_batch(snapshot, new_by_worker=new_by_worker),
        make_policy(**policy_kwargs),
    )

    assert expected_reason in {row["reason"] for row in decision["rejected"]}


def test_reducer_recomputes_market_fill_and_emits_compact_valid_result() -> None:
    epoch = 1_704_067_200
    snapshot = make_snapshot(epoch=epoch)
    intent = make_intent(intent_id="market-a", units=100, valid_until_epoch=epoch)
    batch = make_batch(snapshot, new_by_worker={"worker-a": [intent]})

    result = reduce_portfolio_replay(
        policy=make_policy(),
        frames=[{"post_exit_snapshot": snapshot, "proposal_batch": batch}],
        initial_balance_jpy=200_000,
    )

    # Worker protocol claims executable ask 145.02; reducer-owned slippage fills
    # at 145.03, so terminal bid MTM is -3 JPY, not the worker's -2 JPY.
    assert result["end_equity_jpy"] == pytest.approx(199_997.0)
    assert result["fills"] == 1
    assert result["spread_cost_jpy"] == pytest.approx(2.0)
    assert result["slippage_cost_jpy"] == pytest.approx(1.0)
    assert result["raw_quote_events_included"] is False
    assert "quotes" not in result
    assert validate_portfolio_replay_result(result) == result


def test_quote_batch_is_independently_recomputed() -> None:
    epoch = 1_704_067_200
    snapshot = make_snapshot(epoch=epoch)
    tampered = deepcopy(snapshot)
    tampered["quotes"][0]["bid"] = 144.0
    # Re-sealing makes the worker-protocol hash internally valid while retaining
    # the old reducer quote-batch receipt.  The reducer must still reject it.
    raw_keys = {
        "coordinate_id",
        "epoch",
        "phase",
        "intrabar",
        "quote_batch_sha256",
        "quote_watermark",
        "expected_quote_pairs",
        "active_worker_bindings",
        "account",
        "quotes",
        "positions",
        "pending_orders",
    }
    tampered = seal_post_exit_snapshot({key: tampered[key] for key in raw_keys})
    batch = make_batch(tampered)

    with pytest.raises(DojoPortfolioReplayError, match="quote batch digest"):
        reduce_portfolio_replay(
            policy=make_policy(),
            frames=[{"post_exit_snapshot": tampered, "proposal_batch": batch}],
            initial_balance_jpy=200_000,
        )


def test_risk_release_precedes_same_coordinate_new_admission() -> None:
    epoch = 1_704_067_200
    position = {
        "position_id": "existing-a",
        "worker_id": "worker-a",
        "owner_id": "owner-a",
        "family_id": "family-a",
        "pair": "USD_JPY",
        "side": "LONG",
        "units": 100,
        "entry_price": 145.0,
        "tp_price": 146.0,
        "sl_price": 144.5,
        "opened_epoch": epoch - 60,
        "hard_exit_epoch": epoch + 3600,
    }
    snapshot = make_snapshot(
        epoch=epoch,
        positions=[position],
        equity=200_000,
        margin=100 * 145.01 / 20,
    )
    close = {
        "intent_id": "close-first",
        "action": "CLOSE_POSITION",
        "parameters": {"position_id": "existing-a", "units": None},
        "reason_code": "RELEASE_CAPITAL",
    }
    entry = make_intent(intent_id="replacement", units=100, valid_until_epoch=epoch)
    batch = make_batch(
        snapshot,
        new_by_worker={"worker-b": [entry]},
        reducing_by_worker={"worker-a": [close]},
    )

    decision = admit_worker_proposals(snapshot, batch, make_policy(max_total=1))

    assert decision["accepted_count"] == 1
    assert decision["accepted"][0]["intent_id"] == "replacement"


def test_limit_cannot_fill_at_placement_and_expiry_precedes_later_trigger() -> None:
    epoch = 1_704_067_200
    policy = make_policy()
    snapshot = make_snapshot(epoch=epoch)
    limit = make_intent(
        intent_id="resting-limit",
        action="LIMIT",
        entry_price=144.0,
        sl_price=143.5,
        tp_price=145.0,
        valid_until_epoch=epoch + 60,
    )
    result = reduce_portfolio_replay(
        policy=policy,
        frames=[
            {
                "post_exit_snapshot": snapshot,
                "proposal_batch": make_batch(
                    snapshot, new_by_worker={"worker-a": [limit]}
                ),
            }
        ],
        initial_balance_jpy=200_000,
    )
    assert result["fills"] == 0
    assert result["pending_order_count"] == 1

    carry = result["carry_state"]
    phases = [
        ("H", {"USD_JPY": (146.0, 146.02)}),
        ("L", {"USD_JPY": (144.5, 144.52)}),
        ("C", {"USD_JPY": (145.0, 145.02)}),
    ]
    watermark = 2
    for phase, prices in phases:
        next_snapshot = make_snapshot(
            epoch=epoch,
            phase=phase,
            watermark=watermark,
            prices=prices,
            pending_orders=project_pending(carry),
        )
        segment = reduce_portfolio_replay(
            policy=policy,
            frames=[
                {
                    "post_exit_snapshot": next_snapshot,
                    "proposal_batch": make_batch(next_snapshot),
                }
            ],
            carry_state=carry,
        )
        carry = segment["carry_state"]
        watermark += 1

    # At the next O, ask is below the LIMIT, but valid_until_epoch equals this
    # epoch.  Expiry is processed before trigger evaluation, so no fill exists in
    # the worker-visible post-exit snapshot.
    expired_snapshot = make_snapshot(
        epoch=epoch + 60,
        phase="O",
        watermark=watermark,
        prices={"USD_JPY": (143.98, 144.0)},
        pending_orders=[],
    )
    expired = reduce_portfolio_replay(
        policy=policy,
        frames=[
            {
                "post_exit_snapshot": expired_snapshot,
                "proposal_batch": make_batch(expired_snapshot),
            }
        ],
        carry_state=carry,
    )
    assert expired["orders_expired"] == 1
    assert expired["fills"] == 0
    assert expired["pending_order_count"] == 0


def test_exit_then_triggered_pending_and_fresh_market_share_one_canonical_pool() -> (
    None
):
    epoch = 1_704_067_200
    pairs = ("EUR_USD", "USD_JPY")
    policy = make_policy(pairs=pairs, max_cluster_fraction=0.1)
    first_prices = {"EUR_USD": (1.1, 1.1002), "USD_JPY": (145.0, 145.02)}
    first = make_snapshot(epoch=epoch, prices=first_prices)
    exiting = make_intent(
        intent_id="eur-exits-at-h",
        pair="EUR_USD",
        units=10,
        entry_price=1.1002,
        sl_price=1.09,
        tp_price=1.15,
        valid_until_epoch=epoch,
    )
    resting = make_intent(
        intent_id="usd-resting",
        action="LIMIT",
        units=100,
        entry_price=144.5,
        sl_price=143.5,
        tp_price=146.0,
        valid_until_epoch=epoch + 3600,
    )
    first_result = reduce_portfolio_replay(
        policy=policy,
        frames=[
            {
                "post_exit_snapshot": first,
                "proposal_batch": make_batch(
                    first,
                    new_by_worker={"worker-a": [exiting], "worker-b": [resting]},
                ),
            }
        ],
        initial_balance_jpy=200_000,
    )
    assert first_result["open_position_count"] == 1
    assert first_result["pending_order_count"] == 1

    second_prices = {"EUR_USD": (1.16, 1.1602), "USD_JPY": (144.48, 144.5)}
    # EUR TP realizes 10 * (1.15 - 1.1102) USD at the USD/JPY profit bid 145.
    balance_after_exit = 200_000 + 10 * (1.15 - 1.1102) * 144.48
    second = make_snapshot(
        epoch=epoch,
        phase="H",
        watermark=2,
        prices=second_prices,
        balance=balance_after_exit,
        # The resting order is already trigger-locked before the worker-facing
        # snapshot, so it is no longer present in the cancellable pending set.
        pending_orders=[],
    )
    fresh = make_intent(
        intent_id="usd-fresh",
        units=100,
        entry_price=144.5,
        sl_price=143.5,
        tp_price=146.0,
        valid_until_epoch=epoch,
    )

    forward = reduce_portfolio_replay(
        policy=policy,
        frames=[
            {
                "post_exit_snapshot": second,
                "proposal_batch": make_batch(
                    second, new_by_worker={"worker-a": [fresh]}
                ),
            }
        ],
        carry_state=first_result["carry_state"],
    )
    reversed_input = reduce_portfolio_replay(
        policy=policy,
        frames=[
            {
                "post_exit_snapshot": second,
                "proposal_batch": make_batch(
                    second, new_by_worker={"worker-a": [fresh]}, reverse=True
                ),
            }
        ],
        carry_state=first_result["carry_state"],
    )

    assert forward == reversed_input
    assert forward["rejection_counts"] == {"CORRELATION_CLUSTER_CAP": 1}
    assert forward["pending_order_count"] == 0
    assert forward["orders_triggered"] == 1
    assert forward["open_position_count"] == 1
    assert forward["carry_state"]["positions"][0]["worker_id"] == "worker-b"


def test_trigger_is_frozen_before_snapshot_and_late_cancel_cannot_escape_gap_fill() -> (
    None
):
    epoch = 1_704_067_200
    policy = make_policy()
    first = make_snapshot(epoch=epoch)
    stop = make_intent(
        intent_id="gap-stop",
        action="STOP",
        units=100,
        entry_price=145.5,
        sl_price=144.5,
        tp_price=147.0,
        valid_until_epoch=epoch + 3600,
    )
    first_result = reduce_portfolio_replay(
        policy=policy,
        frames=[
            {
                "post_exit_snapshot": first,
                "proposal_batch": make_batch(first, new_by_worker={"worker-a": [stop]}),
            }
        ],
        initial_balance_jpy=200_000,
    )
    pending = project_pending(first_result["carry_state"])
    order_id = pending[0]["order_id"]
    prices = {"USD_JPY": (145.98, 146.0)}

    # A leaky snapshot that still exposes the now-triggered order can carry a
    # protocol-valid cancel, but reducer state already froze and removed it.  The
    # snapshot mismatch fails closed before worker intents can run.
    leaky = make_snapshot(
        epoch=epoch,
        phase="H",
        watermark=2,
        prices=prices,
        pending_orders=pending,
    )
    late_cancel = {
        "intent_id": "cancel-after-seeing-gap",
        "action": "CANCEL_ORDER",
        "parameters": {"order_id": order_id},
        "reason_code": "ADVERSE_GAP_ESCAPE",
    }
    with pytest.raises(
        DojoPortfolioReplayError, match="pending_orders row count mismatch"
    ):
        reduce_portfolio_replay(
            policy=policy,
            frames=[
                {
                    "post_exit_snapshot": leaky,
                    "proposal_batch": make_batch(
                        leaky, reducing_by_worker={"worker-a": [late_cancel]}
                    ),
                }
            ],
            carry_state=first_result["carry_state"],
        )

    locked = make_snapshot(
        epoch=epoch,
        phase="H",
        watermark=2,
        prices=prices,
        pending_orders=[],
    )
    # Under the correct locked-trigger snapshot the protocol itself refuses the
    # late cancel because the order is no longer worker-addressable.
    with pytest.raises(ProtocolViolation, match="unknown pending order"):
        make_batch(locked, reducing_by_worker={"worker-a": [late_cancel]})

    forward = reduce_portfolio_replay(
        policy=policy,
        frames=[
            {
                "post_exit_snapshot": locked,
                "proposal_batch": make_batch(locked),
            }
        ],
        carry_state=first_result["carry_state"],
    )
    reversed_input = reduce_portfolio_replay(
        policy=policy,
        frames=[
            {
                "post_exit_snapshot": locked,
                "proposal_batch": make_batch(locked, reverse=True),
            }
        ],
        carry_state=first_result["carry_state"],
    )

    assert forward == reversed_input
    assert forward["orders_triggered"] == 1
    assert forward["fills"] == 1
    assert forward["carry_state"]["positions"][0]["entry_price"] == pytest.approx(
        146.01
    )


def test_intrabar_path_requires_exact_next_phase() -> None:
    epoch = 1_704_067_200
    first = make_snapshot(epoch=epoch, intrabar="OLHC")
    initial = reduce_portfolio_replay(
        policy=make_policy(),
        frames=[{"post_exit_snapshot": first, "proposal_batch": make_batch(first)}],
        initial_balance_jpy=200_000,
    )
    skipped_l = make_snapshot(epoch=epoch, phase="H", intrabar="OLHC", watermark=2)

    with pytest.raises(DojoPortfolioReplayError, match="exact next phase"):
        reduce_portfolio_replay(
            policy=make_policy(),
            frames=[
                {
                    "post_exit_snapshot": skipped_l,
                    "proposal_batch": make_batch(skipped_l),
                }
            ],
            carry_state=initial["carry_state"],
        )


@pytest.mark.parametrize(
    ("second_prices", "sl_price", "tp_price", "expected_quote_pnl", "conversion"),
    [
        (
            {"EUR_USD": (1.16, 1.1602), "USD_JPY": (150.0, 150.1)},
            1.09,
            1.15,
            100 * (1.15 - 1.1102),
            150.0,
        ),
        (
            {"EUR_USD": (1.08, 1.0802), "USD_JPY": (150.0, 150.1)},
            1.09,
            1.2,
            100 * ((1.08 - 0.02) - 1.1102),
            150.1,
        ),
    ],
)
def test_signed_conversion_uses_profit_bid_and_loss_ask(
    second_prices: dict[str, tuple[float, float]],
    sl_price: float,
    tp_price: float,
    expected_quote_pnl: float,
    conversion: float,
) -> None:
    epoch = 1_704_067_200
    pairs = ("EUR_USD", "USD_JPY")
    first_prices = {"EUR_USD": (1.1, 1.1002), "USD_JPY": (150.0, 150.1)}
    first = make_snapshot(epoch=epoch, prices=first_prices)
    intent = make_intent(
        intent_id="eur-long",
        pair="EUR_USD",
        side="LONG",
        units=100,
        entry_price=1.1002,
        sl_price=sl_price,
        tp_price=tp_price,
        valid_until_epoch=epoch,
    )
    expected_pnl = expected_quote_pnl * conversion
    second = make_snapshot(
        epoch=epoch,
        phase="H",
        watermark=2,
        prices=second_prices,
        balance=200_000 + expected_pnl,
    )

    result = reduce_portfolio_replay(
        policy=make_policy(pairs=pairs),
        frames=[
            {
                "post_exit_snapshot": first,
                "proposal_batch": make_batch(
                    first, new_by_worker={"worker-a": [intent]}
                ),
            },
            {"post_exit_snapshot": second, "proposal_batch": make_batch(second)},
        ],
        initial_balance_jpy=200_000,
    )

    assert result["realized_pnl_jpy"] == pytest.approx(expected_pnl)
    assert result["end_balance_jpy"] == pytest.approx(200_000 + expected_pnl)


def test_financing_mtm_margin_and_capital_lock_continue_across_coordinates() -> None:
    epoch = 1_704_067_200
    policy = make_policy(financing_rate=0.1)
    first = make_snapshot(epoch=epoch)
    intent = make_intent(
        intent_id="one-hour-hold",
        units=100,
        valid_until_epoch=epoch,
    )
    first_batch = make_batch(first, new_by_worker={"worker-a": [intent]})
    proposal_sha = next(
        row["proposal_sha256"]
        for row in first_batch["proposals"]
        if row["worker_id"] == "worker-a"
    )
    position_id = (
        "P-"
        + canonical_portfolio_sha256(
            {
                "proposal_sha256": proposal_sha,
                "intent_id": "one-hour-hold",
                "epoch": epoch,
                "coordinate_seq": 1,
            }
        )[:24]
    )
    position = {
        "position_id": position_id,
        "worker_id": "worker-a",
        "owner_id": "owner-a",
        "family_id": "family-a",
        "pair": "USD_JPY",
        "side": "LONG",
        "units": 100.0,
        "entry_price": 145.03,
        "tp_price": 146.0,
        "sl_price": 144.5,
        "opened_epoch": epoch,
        "hard_exit_epoch": epoch + 3600,
    }
    margin = 100 * 145.01 / 20
    frames = [{"post_exit_snapshot": first, "proposal_batch": first_batch}]
    for watermark, phase in enumerate(("H", "L", "C"), start=2):
        snapshot = make_snapshot(
            epoch=epoch,
            phase=phase,
            watermark=watermark,
            positions=[position],
            equity=199_997.0,
            margin=margin,
        )
        frames.append(
            {"post_exit_snapshot": snapshot, "proposal_batch": make_batch(snapshot)}
        )
    financing = 100 * 0.1 * 3600 / 86400
    final_balance = 200_000 - 5.0 - financing
    final = make_snapshot(
        epoch=epoch + 3600,
        watermark=5,
        balance=final_balance,
        financing=financing,
    )
    frames.append({"post_exit_snapshot": final, "proposal_batch": make_batch(final)})

    result = reduce_portfolio_replay(
        policy=policy, frames=frames, initial_balance_jpy=200_000
    )

    assert result["realized_pnl_jpy"] == pytest.approx(-5.0)
    assert result["financing_cost_jpy"] == pytest.approx(financing)
    assert result["end_balance_jpy"] == pytest.approx(final_balance)
    assert result["peak_margin_jpy"] == pytest.approx(margin)
    assert result["capital_lock_margin_jpy_hours"] == pytest.approx(margin)


def test_margin_closeout_is_economic_failure_not_silent_success() -> None:
    epoch = 1_704_067_200
    policy = make_policy(max_stop_fraction=100.0)
    first = make_snapshot(epoch=epoch)
    intent = make_intent(
        intent_id="leveraged-long",
        units=20_000,
        sl_price=1.0,
        tp_price=200.0,
        valid_until_epoch=epoch,
    )
    # At H the stop is not touched, but equity is negative.  Reducer liquidation
    # sells at bid 100 minus sealed exit slippage .02.
    closeout_balance = 200_000 + 20_000 * (99.98 - 145.03)
    second = make_snapshot(
        epoch=epoch,
        phase="H",
        watermark=2,
        prices={"USD_JPY": (100.0, 100.02)},
        balance=closeout_balance,
    )

    result = reduce_portfolio_replay(
        policy=policy,
        frames=[
            {
                "post_exit_snapshot": first,
                "proposal_batch": make_batch(
                    first, new_by_worker={"worker-a": [intent]}
                ),
            },
            {"post_exit_snapshot": second, "proposal_batch": make_batch(second)},
        ],
        initial_balance_jpy=200_000,
    )

    assert result["margin_closeouts"] == 1
    assert result["end_balance_jpy"] == pytest.approx(closeout_balance)
    assert "MARGIN_CLOSEOUT_OCCURRED" in result["economic_failure_codes"]
    assert "NON_POSITIVE_END_EQUITY" in result["economic_failure_codes"]
    assert result["status"] == "COMPLETE_WITH_ECONOMIC_FAILURES"


def test_result_hash_detects_terminal_metric_tampering() -> None:
    epoch = 1_704_067_200
    snapshot = make_snapshot(epoch=epoch)
    result = reduce_portfolio_replay(
        policy=make_policy(),
        frames=[
            {"post_exit_snapshot": snapshot, "proposal_batch": make_batch(snapshot)}
        ],
        initial_balance_jpy=200_000,
    )
    tampered = deepcopy(result)
    tampered["end_equity_jpy"] += 1
    assert canonical_portfolio_sha256(tampered) != result["result_sha256"]
    with pytest.raises(DojoPortfolioReplayError, match="result_sha256"):
        validate_portfolio_replay_result(tampered)
