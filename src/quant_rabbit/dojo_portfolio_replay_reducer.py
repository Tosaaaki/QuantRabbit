"""Deterministic shared-account reducer for causal DOJO portfolio replay.

This module is deliberately pure data.  It cannot call a broker, a model, or a
worker.  At every coordinate it independently advances the shared account to a
post-exit state, verifies the worker-facing snapshot, then consumes the exact
all-worker proposal batch defined by :mod:`dojo_shared_worker_protocol`.

Worker supplied ``entry_price`` is only an order trigger for LIMIT/STOP.  A
MARKET fill, conversion rate, costs, cluster identity, margin and stop risk are
always recomputed from sealed reducer policy plus the verified quote batch.
``expected_net_edge_jpy`` and ``stress_cost_pips`` are never used for economics
or admission ranking.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Final, Mapping, Sequence

from quant_rabbit.dojo_shared_worker_protocol import (
    ProtocolViolation,
    seal_post_exit_snapshot,
    verify_post_exit_snapshot,
    verify_worker_proposal_batch,
)


PORTFOLIO_POLICY_CONTRACT: Final = "QR_DOJO_SHARED_PORTFOLIO_POLICY_V1"
PORTFOLIO_REPLAY_CONTRACT: Final = "QR_DOJO_SHARED_ACCOUNT_PORTFOLIO_REPLAY_V1"
PORTFOLIO_CARRY_CONTRACT: Final = "QR_DOJO_SHARED_ACCOUNT_CARRY_V1"
PORTFOLIO_CHECKPOINT_CONTRACT: Final = (
    "QR_DOJO_SHARED_ACCOUNT_INTRA_JOB_CHECKPOINT_V1"
)
PORTFOLIO_COORDINATE_RECEIPT_CONTRACT: Final = (
    "QR_DOJO_SHARED_PORTFOLIO_COORDINATE_RECEIPT_V1"
)
QUOTE_BATCH_CONTRACT: Final = "QR_DOJO_EXACT_QUOTE_BATCH_V1"
SCHEMA_VERSION: Final = 1
MONTH_SECONDS: Final = 30 * 24 * 60 * 60
UNVERIFIED_CLUSTER: Final = "UNVERIFIED"
GENESIS_EVENT_SHA256: Final = "0" * 64
MONTH_END_FLAT_SETTLEMENT: Final = "MONTH_END_FLAT_SETTLEMENT"
MONTH_END_MTM_WITH_STATE_HANDOFF: Final = "MONTH_END_MTM_WITH_STATE_HANDOFF"

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_INTRABAR_PHASE_ORDER = {
    "OHLC": {"O": 0, "H": 1, "L": 2, "C": 3},
    "OLHC": {"O": 0, "L": 1, "H": 2, "C": 3},
}
_POLICY_RAW_KEYS = frozenset(
    {
        "policy_id",
        "expected_quote_pairs",
        "tradable_pairs",
        "active_worker_bindings",
        "leverage",
        "margin_closeout_fraction",
        "max_margin_utilization_fraction",
        "max_portfolio_stop_risk_fraction",
        "max_open_and_pending_total",
        "max_open_and_pending_per_pair",
        "max_open_and_pending_per_family",
        "max_currency_gross_notional_fraction",
        "max_cluster_gross_notional_fraction",
        "max_lock_seconds",
        "slippage_by_pair",
        "financing_by_pair",
        "conversion_routes",
        "correlation_bindings",
    }
)
_POLICY_SEALED_EXTRA_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "live_permission",
        "broker_mutation_allowed",
        "worker_economic_claims_authoritative",
        "ranking_policy",
        "unverified_cluster_id",
        "policy_sha256",
    }
)


class DojoPortfolioReplayError(ValueError):
    """Raised when replay evidence is incomplete, inconsistent or non-causal."""


def canonical_portfolio_sha256(value: Any) -> str:
    """Return the canonical JSON SHA-256 used by reducer-owned artifacts."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoPortfolioReplayError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise DojoPortfolioReplayError(f"{path} must be a string-keyed mapping")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DojoPortfolioReplayError(f"{path} must be a sequence")
    return value


def _exact(value: Mapping[str, Any], expected: frozenset[str], path: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise DojoPortfolioReplayError(
            f"{path} schema mismatch: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )


def _finite(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoPortfolioReplayError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise DojoPortfolioReplayError(f"{path} must be finite and >= {minimum}")
    return result


def _positive(value: Any, path: str) -> float:
    result = _finite(value, path)
    if result <= 0:
        raise DojoPortfolioReplayError(f"{path} must be > 0")
    return result


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DojoPortfolioReplayError(f"{path} must be an integer >= {minimum}")
    return value


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DojoPortfolioReplayError(f"{path} must be a non-empty trimmed string")
    return value


def _sha(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise DojoPortfolioReplayError(f"{path} must be a lowercase SHA-256 digest")
    return value


def _pair(value: Any, path: str) -> str:
    result = _identifier(value, path)
    if _PAIR_RE.fullmatch(result) is None:
        raise DojoPortfolioReplayError(f"{path} must be AAA_BBB")
    return result


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _sorted_unique_rows(
    value: Any, path: str, keys: frozenset[str], identity: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(_sequence(value, path)):
        row = _mapping(raw, f"{path}[{index}]")
        _exact(row, keys, f"{path}[{index}]")
        ident = _identifier(row[identity], f"{path}[{index}].{identity}")
        if ident in seen:
            raise DojoPortfolioReplayError(f"duplicate {path}.{identity}: {ident}")
        seen.add(ident)
        rows.append(dict(row))
    return sorted(rows, key=lambda row: row[identity])


def seal_portfolio_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and seal every reducer-owned economic and portfolio constraint."""

    raw = _mapping(policy, "policy")
    _exact(raw, _POLICY_RAW_KEYS, "policy")

    expected_pairs = sorted(
        {
            _pair(item, "policy.expected_quote_pairs")
            for item in _sequence(
                raw["expected_quote_pairs"], "policy.expected_quote_pairs"
            )
        }
    )
    if not expected_pairs:
        raise DojoPortfolioReplayError("policy.expected_quote_pairs must not be empty")
    tradable_pairs = sorted(
        {
            _pair(item, "policy.tradable_pairs")
            for item in _sequence(raw["tradable_pairs"], "policy.tradable_pairs")
        }
    )
    if not tradable_pairs or not set(tradable_pairs).issubset(expected_pairs):
        raise DojoPortfolioReplayError(
            "policy.tradable_pairs must be a non-empty subset of expected_quote_pairs"
        )

    bindings = _sorted_unique_rows(
        raw["active_worker_bindings"],
        "policy.active_worker_bindings",
        frozenset({"worker_id", "owner_id", "family_id", "config_sha256"}),
        "worker_id",
    )
    if not bindings:
        raise DojoPortfolioReplayError("policy requires at least one active worker")
    for index, row in enumerate(bindings):
        for key in ("owner_id", "family_id"):
            row[key] = _identifier(
                row[key], f"policy.active_worker_bindings[{index}].{key}"
            )
        row["config_sha256"] = _sha(
            row["config_sha256"],
            f"policy.active_worker_bindings[{index}].config_sha256",
        )

    slippage = _sorted_unique_rows(
        raw["slippage_by_pair"],
        "policy.slippage_by_pair",
        frozenset({"pair", "entry_slippage_price", "exit_slippage_price"}),
        "pair",
    )
    financing = _sorted_unique_rows(
        raw["financing_by_pair"],
        "policy.financing_by_pair",
        frozenset(
            {"pair", "long_cost_jpy_per_unit_day", "short_cost_jpy_per_unit_day"}
        ),
        "pair",
    )
    if {row["pair"] for row in slippage} != set(expected_pairs):
        raise DojoPortfolioReplayError(
            "slippage_by_pair must cover expected_quote_pairs exactly"
        )
    if {row["pair"] for row in financing} != set(expected_pairs):
        raise DojoPortfolioReplayError(
            "financing_by_pair must cover expected_quote_pairs exactly"
        )
    for index, row in enumerate(slippage):
        row["pair"] = _pair(row["pair"], f"policy.slippage_by_pair[{index}].pair")
        row["entry_slippage_price"] = _finite(
            row["entry_slippage_price"],
            f"policy.slippage_by_pair[{index}].entry_slippage_price",
            minimum=0,
        )
        row["exit_slippage_price"] = _finite(
            row["exit_slippage_price"],
            f"policy.slippage_by_pair[{index}].exit_slippage_price",
            minimum=0,
        )
    for index, row in enumerate(financing):
        row["pair"] = _pair(row["pair"], f"policy.financing_by_pair[{index}].pair")
        row["long_cost_jpy_per_unit_day"] = _finite(
            row["long_cost_jpy_per_unit_day"],
            f"policy.financing_by_pair[{index}].long_cost_jpy_per_unit_day",
            minimum=0,
        )
        row["short_cost_jpy_per_unit_day"] = _finite(
            row["short_cost_jpy_per_unit_day"],
            f"policy.financing_by_pair[{index}].short_cost_jpy_per_unit_day",
            minimum=0,
        )

    routes = _sorted_unique_rows(
        raw["conversion_routes"],
        "policy.conversion_routes",
        frozenset({"currency", "pair", "orientation"}),
        "currency",
    )
    for index, row in enumerate(routes):
        currency = _identifier(
            row["currency"], f"policy.conversion_routes[{index}].currency"
        )
        if len(currency) != 3 or not currency.isupper() or currency == "JPY":
            raise DojoPortfolioReplayError(
                "conversion route currency must be a non-JPY ISO-like code"
            )
        row["pair"] = _pair(row["pair"], f"policy.conversion_routes[{index}].pair")
        if row["pair"] not in expected_pairs:
            raise DojoPortfolioReplayError(
                "conversion route pair must be in expected_quote_pairs"
            )
        if row["orientation"] not in {"JPY_PER_CURRENCY", "CURRENCY_PER_JPY"}:
            raise DojoPortfolioReplayError("unsupported conversion route orientation")

    correlation_rows: list[dict[str, Any]] = []
    seen_correlation: set[tuple[str, str]] = set()
    worker_ids = {row["worker_id"] for row in bindings}
    for index, raw_row in enumerate(
        _sequence(raw["correlation_bindings"], "policy.correlation_bindings")
    ):
        row = _mapping(raw_row, f"policy.correlation_bindings[{index}]")
        _exact(
            row,
            frozenset({"worker_id", "pair", "cluster_id", "verified"}),
            f"policy.correlation_bindings[{index}]",
        )
        worker_id = _identifier(
            row["worker_id"], f"policy.correlation_bindings[{index}].worker_id"
        )
        pair = _pair(row["pair"], f"policy.correlation_bindings[{index}].pair")
        if worker_id not in worker_ids or pair not in expected_pairs:
            raise DojoPortfolioReplayError(
                "correlation binding references an inactive worker or unknown pair"
            )
        if not isinstance(row["verified"], bool):
            raise DojoPortfolioReplayError("correlation binding verified must be bool")
        key = (worker_id, pair)
        if key in seen_correlation:
            raise DojoPortfolioReplayError(f"duplicate correlation binding: {key}")
        seen_correlation.add(key)
        correlation_rows.append(
            {
                "worker_id": worker_id,
                "pair": pair,
                "cluster_id": _identifier(
                    row["cluster_id"],
                    f"policy.correlation_bindings[{index}].cluster_id",
                ),
                "verified": row["verified"],
            }
        )
    correlation_rows.sort(key=lambda row: (row["worker_id"], row["pair"]))

    leverage = _positive(raw["leverage"], "policy.leverage")
    admission_margin = _positive(
        raw["max_margin_utilization_fraction"], "policy.max_margin_utilization_fraction"
    )
    closeout_margin = _positive(
        raw["margin_closeout_fraction"], "policy.margin_closeout_fraction"
    )
    if (
        admission_margin > 1
        or closeout_margin > 1
        or admission_margin > closeout_margin
    ):
        raise DojoPortfolioReplayError(
            "margin fractions must be <= 1 and admission <= closeout"
        )

    body: dict[str, Any] = {
        "contract": PORTFOLIO_POLICY_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "policy_id": _identifier(raw["policy_id"], "policy.policy_id"),
        "expected_quote_pairs": expected_pairs,
        "tradable_pairs": tradable_pairs,
        "active_worker_bindings": bindings,
        "leverage": leverage,
        "margin_closeout_fraction": closeout_margin,
        "max_margin_utilization_fraction": admission_margin,
        "max_portfolio_stop_risk_fraction": _positive(
            raw["max_portfolio_stop_risk_fraction"],
            "policy.max_portfolio_stop_risk_fraction",
        ),
        "max_open_and_pending_total": _integer(
            raw["max_open_and_pending_total"],
            "policy.max_open_and_pending_total",
            minimum=1,
        ),
        "max_open_and_pending_per_pair": _integer(
            raw["max_open_and_pending_per_pair"],
            "policy.max_open_and_pending_per_pair",
            minimum=1,
        ),
        "max_open_and_pending_per_family": _integer(
            raw["max_open_and_pending_per_family"],
            "policy.max_open_and_pending_per_family",
            minimum=1,
        ),
        "max_currency_gross_notional_fraction": _positive(
            raw["max_currency_gross_notional_fraction"],
            "policy.max_currency_gross_notional_fraction",
        ),
        "max_cluster_gross_notional_fraction": _positive(
            raw["max_cluster_gross_notional_fraction"],
            "policy.max_cluster_gross_notional_fraction",
        ),
        "max_lock_seconds": _integer(
            raw["max_lock_seconds"], "policy.max_lock_seconds", minimum=1
        ),
        "slippage_by_pair": slippage,
        "financing_by_pair": financing,
        "conversion_routes": routes,
        "correlation_bindings": correlation_rows,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "worker_economic_claims_authoritative": False,
        "ranking_policy": "LOWEST_STOP_RISK_THEN_MARGIN_THEN_CANONICAL_ID_NO_EDGE_CLAIM",
        "unverified_cluster_id": UNVERIFIED_CLUSTER,
    }
    body["policy_sha256"] = canonical_portfolio_sha256(body)
    return _copy(body)


def verify_portfolio_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a sealed policy and return a detached canonical copy."""

    sealed = _mapping(policy, "sealed_policy")
    _exact(sealed, _POLICY_RAW_KEYS | _POLICY_SEALED_EXTRA_KEYS, "sealed_policy")
    raw = {key: sealed[key] for key in _POLICY_RAW_KEYS}
    rebuilt = seal_portfolio_policy(raw)
    if _copy(sealed) != rebuilt:
        raise DojoPortfolioReplayError(
            "sealed policy content or policy_sha256 is invalid"
        )
    return rebuilt


def quote_batch_sha256(
    *,
    epoch: int,
    phase: str,
    intrabar: str,
    quote_watermark: int,
    quotes: Sequence[Mapping[str, Any]],
) -> str:
    """Digest an exact, sorted full-coverage quote coordinate."""

    normalized = []
    for index, raw in enumerate(_sequence(quotes, "quotes")):
        row = _mapping(raw, f"quotes[{index}]")
        _exact(row, frozenset({"pair", "bid", "ask", "timestamp"}), f"quotes[{index}]")
        bid = _positive(row["bid"], f"quotes[{index}].bid")
        ask = _positive(row["ask"], f"quotes[{index}].ask")
        if ask <= bid:
            raise DojoPortfolioReplayError("quote ask must be greater than bid")
        normalized.append(
            {
                "pair": _pair(row["pair"], f"quotes[{index}].pair"),
                "bid": bid,
                "ask": ask,
                "timestamp": _identifier(
                    row["timestamp"], f"quotes[{index}].timestamp"
                ),
            }
        )
    normalized.sort(key=lambda row: row["pair"])
    if len({row["pair"] for row in normalized}) != len(normalized):
        raise DojoPortfolioReplayError("quote batch contains duplicate pair")
    return canonical_portfolio_sha256(
        {
            "contract": QUOTE_BATCH_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "epoch": _integer(epoch, "epoch"),
            "phase": _identifier(phase, "phase"),
            "intrabar": _identifier(intrabar, "intrabar"),
            "quote_watermark": _integer(quote_watermark, "quote_watermark"),
            "quotes": normalized,
        }
    )


def _policy_indexes(policy: Mapping[str, Any]) -> dict[str, Any]:
    correlations = {
        (row["worker_id"], row["pair"]): row for row in policy["correlation_bindings"]
    }
    return {
        "slippage": {row["pair"]: row for row in policy["slippage_by_pair"]},
        "financing": {row["pair"]: row for row in policy["financing_by_pair"]},
        "routes": {row["currency"]: row for row in policy["conversion_routes"]},
        "correlations": correlations,
    }


def _cluster(indexes: Mapping[str, Any], worker_id: str, pair: str) -> str:
    row = indexes["correlations"].get((worker_id, pair))
    if row is None or not row["verified"]:
        return UNVERIFIED_CLUSTER
    return row["cluster_id"]


def _quote_map(snapshot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["pair"]: row for row in snapshot["quotes"]}


def _signed_quote_to_jpy(
    amount: float,
    currency: str,
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
) -> float:
    """Convert signed quote PnL: profit at bid, loss at ask (inverse symmetrically)."""

    if amount == 0:
        return 0.0
    if currency == "JPY":
        return amount
    route = indexes["routes"].get(currency)
    if route is None:
        raise DojoPortfolioReplayError(
            f"missing sealed JPY conversion route for {currency}"
        )
    quote = quotes.get(route["pair"])
    if quote is None:
        raise DojoPortfolioReplayError(f"missing conversion quote {route['pair']}")
    if route["orientation"] == "JPY_PER_CURRENCY":
        factor = quote["bid"] if amount > 0 else quote["ask"]
    else:
        denominator = quote["ask"] if amount > 0 else quote["bid"]
        factor = 1.0 / denominator
    return amount * factor


def _positive_quote_to_jpy(
    amount: float,
    currency: str,
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
) -> float:
    if amount < 0:
        raise DojoPortfolioReplayError("positive conversion received negative amount")
    return _signed_quote_to_jpy(amount, currency, quotes, indexes)


def _currencies(pair: str) -> tuple[str, str]:
    base, quote = pair.split("_", 1)
    return base, quote


def _executable_price(
    pair: str, side: str, quotes: Mapping[str, Mapping[str, Any]]
) -> float:
    quote = quotes[pair]
    return float(quote["ask"] if side == "LONG" else quote["bid"])


def _market_fill(
    pair: str,
    side: str,
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
    *,
    exit_fill: bool,
) -> tuple[float, float]:
    executable = _executable_price(pair, side, quotes)
    slip_row = indexes["slippage"][pair]
    slip = float(
        slip_row["exit_slippage_price" if exit_fill else "entry_slippage_price"]
    )
    price = executable + slip if side == "LONG" else executable - slip
    if price <= 0:
        raise DojoPortfolioReplayError("sealed slippage produces a non-positive fill")
    return price, slip


def _pending_fill(
    order: Mapping[str, Any], quote: Mapping[str, Any], indexes: Mapping[str, Any]
) -> tuple[float, float] | None:
    side = order["side"]
    trigger = float(order["trigger_price"])
    if order["order_kind"] == "LIMIT":
        executable = float(quote["ask"] if side == "LONG" else quote["bid"])
        triggered = (side == "LONG" and executable <= trigger) or (
            side == "SHORT" and executable >= trigger
        )
        if not triggered:
            return None
        # Apply the sealed adverse entry-slippage offset to favorable gap price
        # improvement, capped at the limit so a LIMIT is never filled worse
        # than its trigger.  The returned cost is the effective offset actually
        # consumed; an exact touch therefore fills at the trigger with zero
        # additional slippage, while a wider favorable gap pays the sealed cost.
        sealed_slip = float(
            indexes["slippage"][order["pair"]]["entry_slippage_price"]
        )
        if side == "LONG":
            price = min(trigger, executable + sealed_slip)
            effective_slip = price - executable
        else:
            price = max(trigger, executable - sealed_slip)
            effective_slip = executable - price
        return price, effective_slip
    triggered = (side == "LONG" and quote["ask"] >= trigger) or (
        side == "SHORT" and quote["bid"] <= trigger
    )
    if not triggered:
        return None
    slip = float(indexes["slippage"][order["pair"]]["entry_slippage_price"])
    executable = (
        max(trigger, float(quote["ask"]))
        if side == "LONG"
        else min(trigger, float(quote["bid"]))
    )
    price = executable + slip if side == "LONG" else executable - slip
    if price <= 0:
        raise DojoPortfolioReplayError(
            "sealed slippage produces a non-positive pending fill"
        )
    return price, slip


def _position_unrealized(
    position: Mapping[str, Any],
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
) -> float:
    quote = quotes[position["pair"]]
    exit_price = float(quote["bid"] if position["side"] == "LONG" else quote["ask"])
    signed_quote = float(position["units"]) * (
        exit_price - float(position["entry_price"])
        if position["side"] == "LONG"
        else float(position["entry_price"]) - exit_price
    )
    return _signed_quote_to_jpy(
        signed_quote, _currencies(position["pair"])[1], quotes, indexes
    )


def _notional_jpy(
    item: Mapping[str, Any],
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
) -> float:
    pair = item["pair"]
    mid = (float(quotes[pair]["bid"]) + float(quotes[pair]["ask"])) / 2.0
    quote_amount = float(item["units"]) * mid
    return _positive_quote_to_jpy(quote_amount, _currencies(pair)[1], quotes, indexes)


def _margin(
    item: Mapping[str, Any],
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> float:
    return _notional_jpy(item, quotes, indexes) / float(policy["leverage"])


def _stop_risk(
    item: Mapping[str, Any],
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
) -> float:
    pair = item["pair"]
    side = item["side"]
    entry = float(item.get("entry_price", item.get("trigger_price")))
    stop = float(item["sl_price"])
    units = float(item["units"])
    loss_quote = max(
        0.0, units * ((entry - stop) if side == "LONG" else (stop - entry))
    )
    slip_quote = units * float(indexes["slippage"][pair]["exit_slippage_price"])
    return abs(
        _signed_quote_to_jpy(
            -(loss_quote + slip_quote), _currencies(pair)[1], quotes, indexes
        )
    )


def _account_equity(
    state: Mapping[str, Any],
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
) -> float:
    return float(state["balance_jpy"]) + sum(
        _position_unrealized(position, quotes, indexes)
        for position in state["positions"].values()
    )


def _exposure(
    positions: Sequence[Mapping[str, Any]],
    pending: Sequence[Mapping[str, Any]],
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    items = list(positions) + list(pending)
    counts_pair: Counter[str] = Counter()
    counts_family: Counter[str] = Counter()
    currency: defaultdict[str, float] = defaultdict(float)
    clusters: defaultdict[str, float] = defaultdict(float)
    margin = 0.0
    stop_risk = 0.0
    for item in items:
        notional = _notional_jpy(item, quotes, indexes)
        counts_pair[item["pair"]] += 1
        counts_family[item["family_id"]] += 1
        base, quote = _currencies(item["pair"])
        currency[base] += notional
        currency[quote] += notional
        clusters[_cluster(indexes, item["worker_id"], item["pair"])] += notional
        margin += _margin(item, quotes, indexes, policy)
        stop_risk += _stop_risk(item, quotes, indexes)
    return {
        "total": len(items),
        "pair": counts_pair,
        "family": counts_family,
        "currency": currency,
        "clusters": clusters,
        "margin": margin,
        "stop_risk": stop_risk,
    }


def _candidate_from_intent(
    proposal: Mapping[str, Any],
    intent: Mapping[str, Any],
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
    policy: Mapping[str, Any],
    epoch: int,
) -> dict[str, Any]:
    params = intent["parameters"]
    action = intent["action"]
    if action == "MARKET":
        price, slip = _market_fill(
            params["pair"], params["side"], quotes, indexes, exit_fill=False
        )
        trigger = None
    else:
        trigger = float(params["entry_price"])
        if action == "STOP":
            # Reserve risk using at least sealed entry slippage.  A later gap can
            # still fill worse and is independently recomputed at activation.
            slip = float(indexes["slippage"][params["pair"]]["entry_slippage_price"])
            price = trigger + slip if params["side"] == "LONG" else trigger - slip
        else:
            price = trigger
            slip = 0.0
    candidate = {
        "worker_id": proposal["worker_id"],
        "owner_id": proposal["owner_id"],
        "family_id": proposal["family_id"],
        "config_sha256": proposal["config_sha256"],
        "proposal_sha256": proposal["proposal_sha256"],
        "intent_id": intent["intent_id"],
        "action": action,
        "pair": params["pair"],
        "side": params["side"],
        "units": float(params["units"]),
        "entry_price": price,
        "trigger_price": trigger,
        "tp_price": params["tp_price"],
        "sl_price": float(params["sl_price"]),
        "entry_slippage_price": slip,
        "hard_max_holding_seconds": int(params["hard_max_holding_seconds"]),
        "valid_until_epoch": int(params["valid_until_epoch"]),
        "created_epoch": epoch,
        "cluster_id": _cluster(indexes, proposal["worker_id"], params["pair"]),
        "source": "FRESH_WORKER_INTENT",
        "worker_claims_ignored": ["expected_net_edge_jpy", "stress_cost_pips"],
    }
    candidate["stop_risk_jpy"] = _stop_risk(candidate, quotes, indexes)
    candidate["margin_jpy"] = _margin(candidate, quotes, indexes, policy)
    return candidate


def _allocation_rejection(
    candidate: Mapping[str, Any],
    accepted: Sequence[Mapping[str, Any]],
    existing_positions: Sequence[Mapping[str, Any]],
    existing_pending: Sequence[Mapping[str, Any]],
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
    policy: Mapping[str, Any],
    equity_jpy: float,
) -> str | None:
    if equity_jpy <= 0:
        return "NON_POSITIVE_EQUITY"
    if candidate["pair"] not in policy["tradable_pairs"]:
        return "PAIR_NOT_TRADABLE"
    if candidate["hard_max_holding_seconds"] > policy["max_lock_seconds"]:
        return "LOCK_CAP"
    if (
        candidate["action"] != "MARKET"
        and candidate["valid_until_epoch"] - candidate["created_epoch"]
        > policy["max_lock_seconds"]
    ):
        return "PENDING_LOCK_CAP"
    shadow_pending = list(existing_pending) + list(accepted) + [candidate]
    exposure = _exposure(existing_positions, shadow_pending, quotes, indexes, policy)
    if exposure["total"] > policy["max_open_and_pending_total"]:
        return "GLOBAL_COUNT_CAP"
    if exposure["pair"][candidate["pair"]] > policy["max_open_and_pending_per_pair"]:
        return "PAIR_COUNT_CAP"
    if (
        exposure["family"][candidate["family_id"]]
        > policy["max_open_and_pending_per_family"]
    ):
        return "FAMILY_COUNT_CAP"
    if exposure["margin"] / equity_jpy > policy["max_margin_utilization_fraction"]:
        return "MARGIN_CAP"
    if exposure["stop_risk"] / equity_jpy > policy["max_portfolio_stop_risk_fraction"]:
        return "STOP_RISK_CAP"
    if any(
        amount / equity_jpy > policy["max_currency_gross_notional_fraction"]
        for amount in exposure["currency"].values()
    ):
        return "CURRENCY_GROSS_CAP"
    if any(
        amount / equity_jpy > policy["max_cluster_gross_notional_fraction"]
        for amount in exposure["clusters"].values()
    ):
        return "CORRELATION_CLUSTER_CAP"
    return None


def _place_accepted_candidate(
    state: dict[str, Any],
    candidate: Mapping[str, Any],
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
    epoch: int,
    coordinate_seq: int,
) -> None:
    if (
        candidate["action"] == "MARKET"
        or candidate["source"] == "RESTING_PENDING_TRIGGER"
    ):
        _open_position(
            state,
            candidate,
            candidate["entry_price"],
            candidate["entry_slippage_price"],
            quotes,
            indexes,
            epoch,
            coordinate_seq,
        )
        return
    seed = {
        "proposal_sha256": candidate["proposal_sha256"],
        "intent_id": candidate["intent_id"],
        "epoch": epoch,
        "coordinate_seq": coordinate_seq,
    }
    order_id = "O-" + canonical_portfolio_sha256(seed)[:24]
    order = {
        "order_id": order_id,
        "worker_id": candidate["worker_id"],
        "owner_id": candidate["owner_id"],
        "family_id": candidate["family_id"],
        "pair": candidate["pair"],
        "side": candidate["side"],
        "order_kind": candidate["action"],
        "units": candidate["units"],
        "entry_price": candidate["entry_price"],
        "trigger_price": candidate["trigger_price"],
        "tp_price": candidate["tp_price"],
        "sl_price": candidate["sl_price"],
        "created_epoch": epoch,
        "valid_until_epoch": candidate["valid_until_epoch"],
        "created_coordinate_seq": coordinate_seq,
        "hard_max_holding_seconds": candidate["hard_max_holding_seconds"],
        "proposal_sha256": candidate["proposal_sha256"],
        "intent_id": candidate["intent_id"],
    }
    if order_id in state["pending_orders"]:
        raise DojoPortfolioReplayError(f"deterministic order id collision: {order_id}")
    state["pending_orders"][order_id] = order
    state["metrics"]["orders_placed"] += 1
    _event(
        state,
        "ORDER_PLACE",
        {
            "order_id": order_id,
            "order_kind": order["order_kind"],
            "trigger_price": order["trigger_price"],
            "activation": "NEXT_COORDINATE_OR_LATER",
        },
    )


def admit_worker_proposals(
    snapshot: Mapping[str, Any],
    proposal_batch: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify and order-independently admit all new-risk proposals.

    The returned rows are declarative decisions only.  No order is sent and no
    worker-claimed edge is treated as proof.  Risk-reducing intents are applied
    logically before multi-admission so released capacity is visible.
    """

    verified_policy = verify_portfolio_policy(policy)
    try:
        verified_snapshot = verify_post_exit_snapshot(snapshot)
        verified_batch = verify_worker_proposal_batch(verified_snapshot, proposal_batch)
    except ProtocolViolation as exc:
        raise DojoPortfolioReplayError(f"worker protocol violation: {exc}") from exc
    if (
        verified_snapshot["expected_quote_pairs"]
        != verified_policy["expected_quote_pairs"]
    ):
        raise DojoPortfolioReplayError(
            "snapshot quote coverage does not match sealed policy"
        )
    if (
        verified_snapshot["active_worker_bindings"]
        != verified_policy["active_worker_bindings"]
    ):
        raise DojoPortfolioReplayError("snapshot workers do not match sealed policy")
    recomputed_batch = quote_batch_sha256(
        epoch=verified_snapshot["epoch"],
        phase=verified_snapshot["phase"],
        intrabar=verified_snapshot["intrabar"],
        quote_watermark=verified_snapshot["quote_watermark"],
        quotes=verified_snapshot["quotes"],
    )
    if recomputed_batch != verified_snapshot["quote_batch_sha256"]:
        raise DojoPortfolioReplayError(
            "quote_batch_sha256 is not the reducer-recomputed digest"
        )

    indexes = _policy_indexes(verified_policy)
    quotes = _quote_map(verified_snapshot)
    # Use the same executable-price close/cancel/tighten path as full replay.
    # This matters when an exit's sealed slippage lowers equity before a second
    # same-coordinate MARKET candidate is tested.
    decision_state = {
        "balance_jpy": float(verified_snapshot["account"]["balance_jpy"]),
        "accrued_financing_jpy": float(
            verified_snapshot["account"]["accrued_financing_jpy"]
        ),
        "positions": {
            row["position_id"]: dict(row) for row in verified_snapshot["positions"]
        },
        "pending_orders": {
            row["order_id"]: dict(row) for row in verified_snapshot["pending_orders"]
        },
        "metrics": _empty_metrics(),
        "event_chain_sha256": GENESIS_EVENT_SHA256,
        "event_count": 0,
    }
    _apply_actual_risk_reductions(decision_state, verified_batch, quotes, indexes)
    risk_reductions = [
        {**intent, "worker_id": proposal["worker_id"]}
        for proposal in verified_batch["proposals"]
        for intent in proposal["risk_reducing_intents"]
    ]
    risk_reductions.sort(
        key=lambda row: (row["action"], row["worker_id"], row["intent_id"])
    )
    candidates = [
        _candidate_from_intent(
            proposal,
            intent,
            quotes,
            indexes,
            verified_policy,
            verified_snapshot["epoch"],
        )
        for proposal in verified_batch["proposals"]
        for intent in proposal["new_risk_intents"]
    ]
    candidates.sort(
        key=lambda row: (
            row["stop_risk_jpy"],
            row["margin_jpy"],
            row["family_id"],
            row["worker_id"],
            row["intent_id"],
        )
    )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for candidate in candidates:
        equity = _account_equity(decision_state, quotes, indexes)
        reason = _allocation_rejection(
            candidate,
            [],
            list(decision_state["positions"].values()),
            list(decision_state["pending_orders"].values()),
            quotes,
            indexes,
            verified_policy,
            equity,
        )
        if reason is None:
            accepted.append(candidate)
            _place_accepted_candidate(
                decision_state,
                candidate,
                quotes,
                indexes,
                verified_snapshot["epoch"],
                1,
            )
        else:
            rejected.append(
                {
                    "worker_id": candidate["worker_id"],
                    "intent_id": candidate["intent_id"],
                    "reason": reason,
                }
            )
    body = {
        "contract": "QR_DOJO_SHARED_ADMISSION_DECISION_V1",
        "schema_version": SCHEMA_VERSION,
        "policy_sha256": verified_policy["policy_sha256"],
        "snapshot_sha256": verified_snapshot["snapshot_sha256"],
        "proposal_batch_sha256": verified_batch["batch_sha256"],
        "ranking_policy": verified_policy["ranking_policy"],
        "risk_reducing_intents": risk_reductions,
        "accepted": accepted,
        "rejected": rejected,
        "attempt_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    body["decision_sha256"] = canonical_portfolio_sha256(body)
    return body


def _empty_metrics() -> dict[str, Any]:
    return {
        "attempts": 0,
        "fills": 0,
        "rejections": 0,
        "orders_placed": 0,
        "orders_triggered": 0,
        "orders_expired": 0,
        "orders_cancelled": 0,
        "position_closes": 0,
        "margin_closeouts": 0,
        "margin_reject_count": 0,
        "ruin_event_count": 0,
        "realized_pnl_jpy": 0.0,
        "financing_cost_jpy": 0.0,
        "spread_cost_jpy": 0.0,
        "slippage_cost_jpy": 0.0,
        "capital_lock_margin_jpy_seconds": 0.0,
        "peak_margin_jpy": 0.0,
        "peak_margin_usage_fraction": 0.0,
        "minimum_mtm_equity_jpy": None,
        "minimum_free_margin_jpy": None,
        "max_drawdown_fraction": 0.0,
        "rejection_counts": {},
        "pair_pnl": {},
        "family_pnl": {},
    }


def _pnl_bucket(metrics: dict[str, Any], dimension: str, key: str) -> dict[str, float]:
    bucket = metrics[dimension].setdefault(
        key, {"realized_pnl_jpy": 0.0, "financing_cost_jpy": 0.0, "net_pnl_jpy": 0.0}
    )
    return bucket


def _event(state: dict[str, Any], kind: str, payload: Mapping[str, Any]) -> None:
    body = {
        "previous_event_sha256": state["event_chain_sha256"],
        "event_index": state["event_count"] + 1,
        "kind": kind,
        "payload": payload,
    }
    state["event_chain_sha256"] = canonical_portfolio_sha256(body)
    state["event_count"] += 1


def _record_equity(state: dict[str, Any], equity: float) -> None:
    state["peak_equity_jpy"] = max(float(state["peak_equity_jpy"]), equity)
    if state["peak_equity_jpy"] > 0:
        drawdown = max(
            0.0, (state["peak_equity_jpy"] - equity) / state["peak_equity_jpy"]
        )
        state["metrics"]["max_drawdown_fraction"] = max(
            float(state["metrics"]["max_drawdown_fraction"]), drawdown
        )


def _record_account_state(
    state: dict[str, Any],
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[float, float, float]:
    """Record streaming MTM, reserved margin and free-margin extrema."""

    equity = _account_equity(state, quotes, indexes)
    reserved_margin = _exposure(
        list(state["positions"].values()),
        list(state["pending_orders"].values()),
        quotes,
        indexes,
        policy,
    )["margin"]
    free_margin = equity - reserved_margin
    _record_equity(state, equity)
    metrics = state["metrics"]
    current_min_equity = metrics["minimum_mtm_equity_jpy"]
    metrics["minimum_mtm_equity_jpy"] = (
        equity if current_min_equity is None else min(float(current_min_equity), equity)
    )
    current_min_free = metrics["minimum_free_margin_jpy"]
    metrics["minimum_free_margin_jpy"] = (
        free_margin
        if current_min_free is None
        else min(float(current_min_free), free_margin)
    )
    margin_usage = reserved_margin / equity if equity > 0 else 1.0
    metrics["peak_margin_usage_fraction"] = max(
        float(metrics["peak_margin_usage_fraction"]), margin_usage
    )
    metrics["peak_margin_jpy"] = max(
        float(metrics["peak_margin_jpy"]), reserved_margin
    )
    ruin_active = bool(state.get("ruin_active", False))
    if equity <= 0 and not ruin_active:
        metrics["ruin_event_count"] += 1
        state["ruin_active"] = True
        _event(
            state,
            "RUIN_ENTER",
            {
                "equity_jpy": equity,
                "free_margin_jpy": free_margin,
            },
        )
    elif equity > 0 and ruin_active:
        state["ruin_active"] = False
    return equity, reserved_margin, free_margin


def _state_from_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "positions": {row["position_id"]: dict(row) for row in snapshot["positions"]},
        "pending_orders": {
            row["order_id"]: dict(row) for row in snapshot["pending_orders"]
        },
    }


def _project_position(position: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "position_id",
        "worker_id",
        "owner_id",
        "family_id",
        "pair",
        "side",
        "units",
        "entry_price",
        "tp_price",
        "sl_price",
        "opened_epoch",
        "hard_exit_epoch",
    )
    return {key: position[key] for key in keys}


def _project_order(order: Mapping[str, Any]) -> dict[str, Any]:
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
    return {key: order[key] for key in keys}


def _assert_close(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isclose(
            float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-7
        ):
            raise DojoPortfolioReplayError(
                f"{path} mismatch: reducer={actual}, snapshot={expected}"
            )
    elif actual != expected:
        raise DojoPortfolioReplayError(
            f"{path} mismatch: reducer={actual}, snapshot={expected}"
        )


def _assert_rows(
    actual: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    identity: str,
    path: str,
) -> None:
    left = sorted(actual, key=lambda row: row[identity])
    right = sorted(expected, key=lambda row: row[identity])
    if len(left) != len(right):
        raise DojoPortfolioReplayError(f"{path} row count mismatch")
    for arow, erow in zip(left, right):
        if set(arow) != set(erow):
            raise DojoPortfolioReplayError(f"{path} row schema mismatch")
        for key in arow:
            _assert_close(arow[key], erow[key], f"{path}.{arow[identity]}.{key}")


def _financing_charges(
    state: Mapping[str, Any], epoch: int, indexes: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Price carry for positions held since the prior coordinate.

    Charges are captured before current-coordinate activation, then booked after
    deterministic price exits.  This prevents a newly triggered order from
    receiving past carry while still charging a position that exits now.
    """

    last_epoch = state["last_epoch"]
    if last_epoch is None:
        return []
    elapsed = epoch - int(last_epoch)
    if elapsed < 0:
        raise DojoPortfolioReplayError("coordinate epoch moved backwards")
    if elapsed == 0:
        return []
    charges: list[dict[str, Any]] = []
    for position in state["positions"].values():
        rates = indexes["financing"][position["pair"]]
        key = (
            "long_cost_jpy_per_unit_day"
            if position["side"] == "LONG"
            else "short_cost_jpy_per_unit_day"
        )
        cost = float(position["units"]) * float(rates[key]) * elapsed / 86400.0
        charges.append(
            {
                "position_id": position["position_id"],
                "pair": position["pair"],
                "family_id": position["family_id"],
                "elapsed_seconds": elapsed,
                "cost_jpy": cost,
            }
        )
    return charges


def _book_financing(
    state: dict[str, Any], charges: Sequence[Mapping[str, Any]]
) -> None:
    for charge in charges:
        cost = float(charge["cost_jpy"])
        state["balance_jpy"] -= cost
        state["accrued_financing_jpy"] += cost
        state["metrics"]["financing_cost_jpy"] += cost
        pair_bucket = _pnl_bucket(state["metrics"], "pair_pnl", charge["pair"])
        family_bucket = _pnl_bucket(state["metrics"], "family_pnl", charge["family_id"])
        for bucket in (pair_bucket, family_bucket):
            bucket["financing_cost_jpy"] += cost
            bucket["net_pnl_jpy"] -= cost
        _event(
            state,
            "FINANCING",
            {
                "position_id": charge["position_id"],
                "elapsed_seconds": charge["elapsed_seconds"],
                "cost_jpy": cost,
            },
        )


def _open_position(
    state: dict[str, Any],
    candidate: Mapping[str, Any],
    fill_price: float,
    slip: float,
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
    epoch: int,
    coordinate_seq: int,
) -> None:
    seed = {
        "proposal_sha256": candidate["proposal_sha256"],
        "intent_id": candidate["intent_id"],
        "epoch": epoch,
        "coordinate_seq": coordinate_seq,
    }
    position_id = "P-" + canonical_portfolio_sha256(seed)[:24]
    if position_id in state["positions"]:
        raise DojoPortfolioReplayError(
            f"deterministic position id collision: {position_id}"
        )
    position = {
        "position_id": position_id,
        "worker_id": candidate["worker_id"],
        "owner_id": candidate["owner_id"],
        "family_id": candidate["family_id"],
        "pair": candidate["pair"],
        "side": candidate["side"],
        "units": candidate["units"],
        "entry_price": fill_price,
        "tp_price": candidate["tp_price"],
        "sl_price": candidate["sl_price"],
        "opened_epoch": epoch,
        "hard_exit_epoch": epoch + candidate["hard_max_holding_seconds"],
        "entry_coordinate_seq": coordinate_seq,
    }
    state["positions"][position_id] = position
    _, quote_currency = _currencies(position["pair"])
    spread_quote = position["units"] * (
        quotes[position["pair"]]["ask"] - quotes[position["pair"]]["bid"]
    )
    spread_cost = abs(
        _signed_quote_to_jpy(-spread_quote, quote_currency, quotes, indexes)
    )
    slip_cost = abs(
        _signed_quote_to_jpy(
            -(position["units"] * slip), quote_currency, quotes, indexes
        )
    )
    state["metrics"]["fills"] += 1
    state["metrics"]["spread_cost_jpy"] += spread_cost
    state["metrics"]["slippage_cost_jpy"] += slip_cost
    _event(
        state,
        "POSITION_OPEN",
        {
            "position_id": position_id,
            "pair": position["pair"],
            "side": position["side"],
            "units": position["units"],
            "fill_price": fill_price,
            "entry_slippage_cost_jpy": slip_cost,
            "spread_cost_jpy": spread_cost,
        },
    )


def _close_position(
    state: dict[str, Any],
    position_id: str,
    units: float | None,
    reason: str,
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
    *,
    protected_price: float | None = None,
) -> None:
    position = state["positions"][position_id]
    close_units = float(position["units"] if units is None else units)
    if close_units <= 0 or close_units > float(position["units"]):
        raise DojoPortfolioReplayError("invalid close quantity")
    if protected_price is None:
        exit_side = "SHORT" if position["side"] == "LONG" else "LONG"
        fill, slip = _market_fill(
            position["pair"], exit_side, quotes, indexes, exit_fill=True
        )
    else:
        fill, slip = float(protected_price), 0.0
    quote_pnl = close_units * (
        fill - float(position["entry_price"])
        if position["side"] == "LONG"
        else float(position["entry_price"]) - fill
    )
    quote_currency = _currencies(position["pair"])[1]
    pnl_jpy = _signed_quote_to_jpy(quote_pnl, quote_currency, quotes, indexes)
    slip_cost = abs(
        _signed_quote_to_jpy(-(close_units * slip), quote_currency, quotes, indexes)
    )
    state["balance_jpy"] += pnl_jpy
    state["metrics"]["realized_pnl_jpy"] += pnl_jpy
    state["metrics"]["slippage_cost_jpy"] += slip_cost
    state["metrics"]["position_closes"] += 1
    for bucket in (
        _pnl_bucket(state["metrics"], "pair_pnl", position["pair"]),
        _pnl_bucket(state["metrics"], "family_pnl", position["family_id"]),
    ):
        bucket["realized_pnl_jpy"] += pnl_jpy
        bucket["net_pnl_jpy"] += pnl_jpy
    position["units"] = float(position["units"]) - close_units
    if position["units"] <= 1e-12:
        del state["positions"][position_id]
    _event(
        state,
        "POSITION_CLOSE",
        {
            "position_id": position_id,
            "reason": reason,
            "units": close_units,
            "fill_price": fill,
            "pnl_jpy": pnl_jpy,
            "exit_slippage_cost_jpy": slip_cost,
        },
    )


def _expire_old_pending(state: dict[str, Any], epoch: int) -> None:
    for order_id in sorted(list(state["pending_orders"])):
        order = state["pending_orders"].get(order_id)
        if order is None:
            continue
        # Expiry is fail-closed and processed before any possible trigger fill.
        if epoch >= int(order["valid_until_epoch"]):
            del state["pending_orders"][order_id]
            state["metrics"]["orders_expired"] += 1
            _event(state, "ORDER_EXPIRE", {"order_id": order_id, "epoch": epoch})


def _triggered_pending_candidates(
    state: dict[str, Any],
    *,
    epoch: int,
    coordinate_seq: int,
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Remove causally triggered resting orders and return central-pool candidates."""

    candidates: list[dict[str, Any]] = []
    for order_id in sorted(list(state["pending_orders"])):
        order = state["pending_orders"][order_id]
        if coordinate_seq <= int(order["created_coordinate_seq"]):
            continue
        fill = _pending_fill(order, quotes[order["pair"]], indexes)
        if fill is None:
            continue
        del state["pending_orders"][order_id]
        candidate = {
            "worker_id": order["worker_id"],
            "owner_id": order["owner_id"],
            "family_id": order["family_id"],
            "proposal_sha256": order["proposal_sha256"],
            "intent_id": order["intent_id"],
            "action": order["order_kind"],
            "pair": order["pair"],
            "side": order["side"],
            "units": float(order["units"]),
            "entry_price": float(fill[0]),
            "trigger_price": float(order["trigger_price"]),
            "tp_price": order["tp_price"],
            "sl_price": float(order["sl_price"]),
            "entry_slippage_price": float(fill[1]),
            "hard_max_holding_seconds": int(order["hard_max_holding_seconds"]),
            "valid_until_epoch": int(order["valid_until_epoch"]),
            "created_epoch": int(order["created_epoch"]),
            "cluster_id": _cluster(indexes, order["worker_id"], order["pair"]),
            "source": "RESTING_PENDING_TRIGGER",
            "resting_order_id": order_id,
            "worker_claims_ignored": ["expected_net_edge_jpy", "stress_cost_pips"],
        }
        candidate["stop_risk_jpy"] = _stop_risk(candidate, quotes, indexes)
        candidate["margin_jpy"] = _margin(candidate, quotes, indexes, policy)
        candidates.append(candidate)
        state["metrics"]["orders_triggered"] += 1
        _event(
            state,
            "ORDER_TRIGGER_READY",
            {
                "order_id": order_id,
                "fill_price": fill[0],
                "coordinate_seq": coordinate_seq,
            },
        )
    return candidates


def _process_system_exits(
    state: dict[str, Any],
    epoch: int,
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
) -> None:
    for position_id in sorted(list(state["positions"])):
        position = state["positions"].get(position_id)
        if position is None:
            continue
        quote = quotes[position["pair"]]
        if position["side"] == "LONG":
            stop_hit = quote["bid"] <= position["sl_price"]
            tp_hit = (
                position["tp_price"] is not None
                and quote["bid"] >= position["tp_price"]
            )
        else:
            stop_hit = quote["ask"] >= position["sl_price"]
            tp_hit = (
                position["tp_price"] is not None
                and quote["ask"] <= position["tp_price"]
            )
        if stop_hit:
            _close_position(state, position_id, None, "STOP_LOSS", quotes, indexes)
        elif tp_hit:
            _close_position(
                state,
                position_id,
                None,
                "TAKE_PROFIT",
                quotes,
                indexes,
                protected_price=float(position["tp_price"]),
            )
        elif epoch >= int(position["hard_exit_epoch"]):
            _close_position(state, position_id, None, "HARD_EXIT", quotes, indexes)


def _process_margin_closeout(
    state: dict[str, Any],
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> None:
    while state["positions"]:
        equity = _account_equity(state, quotes, indexes)
        margin = sum(
            _margin(row, quotes, indexes, policy) for row in state["positions"].values()
        )
        if equity <= 0 or margin / equity >= policy["margin_closeout_fraction"]:
            victim = min(
                state["positions"].values(),
                key=lambda row: (
                    _position_unrealized(row, quotes, indexes),
                    row["position_id"],
                ),
            )
            state["metrics"]["margin_closeouts"] += 1
            _close_position(
                state, victim["position_id"], None, "MARGIN_CLOSEOUT", quotes, indexes
            )
        else:
            break


def _apply_actual_risk_reductions(
    state: dict[str, Any],
    batch: Mapping[str, Any],
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
) -> None:
    intents = []
    for proposal in batch["proposals"]:
        for intent in proposal["risk_reducing_intents"]:
            intents.append(
                (intent["action"], proposal["worker_id"], intent["intent_id"], intent)
            )
    for _, _, _, intent in sorted(intents):
        params = intent["parameters"]
        if intent["action"] == "CANCEL_ORDER":
            if params["order_id"] not in state["pending_orders"]:
                raise DojoPortfolioReplayError(
                    "risk reduction target disappeared before application"
                )
            del state["pending_orders"][params["order_id"]]
            state["metrics"]["orders_cancelled"] += 1
            _event(
                state,
                "ORDER_CANCEL",
                {"order_id": params["order_id"], "intent_id": intent["intent_id"]},
            )
        elif intent["action"] == "CLOSE_POSITION":
            if params["position_id"] not in state["positions"]:
                raise DojoPortfolioReplayError(
                    "risk reduction target disappeared before application"
                )
            _close_position(
                state,
                params["position_id"],
                params["units"],
                "WORKER_CLOSE",
                quotes,
                indexes,
            )
        else:
            if params["position_id"] not in state["positions"]:
                raise DojoPortfolioReplayError(
                    "risk reduction target disappeared before application"
                )
            state["positions"][params["position_id"]]["sl_price"] = params["sl_price"]
            _event(
                state,
                "STOP_TIGHTEN",
                {
                    "position_id": params["position_id"],
                    "sl_price": params["sl_price"],
                    "intent_id": intent["intent_id"],
                },
            )


def _new_state(initial_balance_jpy: float, policy_sha256: str) -> dict[str, Any]:
    metrics = _empty_metrics()
    metrics["minimum_mtm_equity_jpy"] = initial_balance_jpy
    metrics["minimum_free_margin_jpy"] = initial_balance_jpy
    return {
        "policy_sha256": policy_sha256,
        "balance_jpy": initial_balance_jpy,
        "accrued_financing_jpy": 0.0,
        "positions": {},
        "pending_orders": {},
        "last_quotes": {},
        "last_epoch": None,
        "last_coordinate": None,
        "last_quote_watermark": None,
        "coordinate_seq": 0,
        "start_epoch": None,
        "start_balance_jpy": initial_balance_jpy,
        "start_equity_jpy": initial_balance_jpy,
        "peak_equity_jpy": initial_balance_jpy,
        "ruin_active": False,
        "event_chain_sha256": GENESIS_EVENT_SHA256,
        "event_count": 0,
        "metrics": metrics,
    }


def _state_from_carry(
    carry: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    reset_reporting_window: bool = True,
) -> dict[str, Any]:
    row = _mapping(carry, "carry_state")
    expected = frozenset(
        {
            "contract",
            "schema_version",
            "policy_sha256",
            "balance_jpy",
            "equity_jpy",
            "accrued_financing_jpy",
            "positions",
            "pending_orders",
            "last_quotes",
            "last_epoch",
            "last_coordinate",
            "last_quote_watermark",
            "coordinate_seq",
            "start_epoch",
            "start_balance_jpy",
            "start_equity_jpy",
            "peak_equity_jpy",
            "event_chain_sha256",
            "event_count",
            "metrics",
            "carry_state_sha256",
        }
    )
    _exact(row, expected, "carry_state")
    unsigned = {key: row[key] for key in expected if key != "carry_state_sha256"}
    if (
        row["contract"] != PORTFOLIO_CARRY_CONTRACT
        or row["schema_version"] != SCHEMA_VERSION
    ):
        raise DojoPortfolioReplayError("unsupported carry state contract")
    if (
        _sha(row["policy_sha256"], "carry_state.policy_sha256")
        != policy["policy_sha256"]
    ):
        raise DojoPortfolioReplayError("carry state policy mismatch")
    if canonical_portfolio_sha256(unsigned) != _sha(
        row["carry_state_sha256"], "carry_state.carry_state_sha256"
    ):
        raise DojoPortfolioReplayError("carry_state_sha256 mismatch")
    bindings = {item["worker_id"]: item for item in policy["active_worker_bindings"]}
    position_keys = frozenset(
        {
            "position_id",
            "worker_id",
            "owner_id",
            "family_id",
            "pair",
            "side",
            "units",
            "entry_price",
            "tp_price",
            "sl_price",
            "opened_epoch",
            "hard_exit_epoch",
            "entry_coordinate_seq",
        }
    )
    pending_keys = frozenset(
        {
            "order_id",
            "worker_id",
            "owner_id",
            "family_id",
            "pair",
            "side",
            "order_kind",
            "units",
            "entry_price",
            "trigger_price",
            "tp_price",
            "sl_price",
            "created_epoch",
            "valid_until_epoch",
            "created_coordinate_seq",
            "hard_max_holding_seconds",
            "proposal_sha256",
            "intent_id",
        }
    )
    positions: dict[str, dict[str, Any]] = {}
    for index, raw_position in enumerate(
        _sequence(row["positions"], "carry_state.positions")
    ):
        position = dict(_mapping(raw_position, f"carry_state.positions[{index}]"))
        _exact(position, position_keys, f"carry_state.positions[{index}]")
        position_id = _identifier(
            position["position_id"], f"carry_state.positions[{index}].position_id"
        )
        if position_id in positions:
            raise DojoPortfolioReplayError(
                f"duplicate carry position_id: {position_id}"
            )
        binding = bindings.get(position["worker_id"])
        if binding is None or any(
            position[key] != binding[key] for key in ("owner_id", "family_id")
        ):
            raise DojoPortfolioReplayError(
                "carry position owner is not an active sealed worker"
            )
        if position["pair"] not in policy["tradable_pairs"] or position[
            "side"
        ] not in {"LONG", "SHORT"}:
            raise DojoPortfolioReplayError("carry position has unknown pair or side")
        _positive(position["units"], "carry position units")
        _positive(position["entry_price"], "carry position entry_price")
        _positive(position["sl_price"], "carry position sl_price")
        if position["tp_price"] is not None:
            _positive(position["tp_price"], "carry position tp_price")
        _integer(position["opened_epoch"], "carry position opened_epoch")
        _integer(position["hard_exit_epoch"], "carry position hard_exit_epoch")
        _integer(
            position["entry_coordinate_seq"],
            "carry position entry_coordinate_seq",
            minimum=1,
        )
        positions[position_id] = position
    pending_orders: dict[str, dict[str, Any]] = {}
    for index, raw_order in enumerate(
        _sequence(row["pending_orders"], "carry_state.pending_orders")
    ):
        order = dict(_mapping(raw_order, f"carry_state.pending_orders[{index}]"))
        _exact(order, pending_keys, f"carry_state.pending_orders[{index}]")
        order_id = _identifier(
            order["order_id"], f"carry_state.pending_orders[{index}].order_id"
        )
        if order_id in pending_orders:
            raise DojoPortfolioReplayError(f"duplicate carry order_id: {order_id}")
        binding = bindings.get(order["worker_id"])
        if binding is None or any(
            order[key] != binding[key] for key in ("owner_id", "family_id")
        ):
            raise DojoPortfolioReplayError(
                "carry pending order owner is not an active sealed worker"
            )
        if order["pair"] not in policy["tradable_pairs"] or order["side"] not in {
            "LONG",
            "SHORT",
        }:
            raise DojoPortfolioReplayError(
                "carry pending order has unknown pair or side"
            )
        if order["order_kind"] not in {"LIMIT", "STOP"}:
            raise DojoPortfolioReplayError("carry pending order has invalid kind")
        for key in ("units", "entry_price", "trigger_price", "sl_price"):
            _positive(order[key], f"carry pending order {key}")
        if order["tp_price"] is not None:
            _positive(order["tp_price"], "carry pending order tp_price")
        _integer(order["created_epoch"], "carry pending order created_epoch")
        _integer(order["valid_until_epoch"], "carry pending order valid_until_epoch")
        _integer(
            order["created_coordinate_seq"],
            "carry pending order created_coordinate_seq",
            minimum=1,
        )
        _integer(
            order["hard_max_holding_seconds"],
            "carry pending order hard_max_holding_seconds",
            minimum=1,
        )
        _sha(order["proposal_sha256"], "carry pending order proposal_sha256")
        _identifier(order["intent_id"], "carry pending order intent_id")
        pending_orders[order_id] = order
    last_quotes: dict[str, dict[str, Any]] = {}
    for index, raw_quote in enumerate(
        _sequence(row["last_quotes"], "carry_state.last_quotes")
    ):
        quote = dict(_mapping(raw_quote, f"carry_state.last_quotes[{index}]"))
        _exact(
            quote,
            frozenset({"pair", "bid", "ask", "timestamp"}),
            f"carry_state.last_quotes[{index}]",
        )
        pair = _pair(quote["pair"], f"carry_state.last_quotes[{index}].pair")
        if pair in last_quotes:
            raise DojoPortfolioReplayError(f"duplicate carry quote pair: {pair}")
        bid = _positive(quote["bid"], "carry quote bid")
        ask = _positive(quote["ask"], "carry quote ask")
        if ask <= bid:
            raise DojoPortfolioReplayError("carry quote ask must exceed bid")
        _identifier(quote["timestamp"], "carry quote timestamp")
        last_quotes[pair] = quote
    if set(last_quotes) != set(policy["expected_quote_pairs"]):
        raise DojoPortfolioReplayError(
            "carry quote coverage differs from sealed policy"
        )
    metric_keys = frozenset(_empty_metrics())
    metrics = dict(_mapping(row["metrics"], "carry_state.metrics"))
    _exact(metrics, metric_keys, "carry_state.metrics")
    signed_metric_keys = {
        "realized_pnl_jpy",
        "minimum_mtm_equity_jpy",
        "minimum_free_margin_jpy",
    }
    for key in metric_keys - {"rejection_counts", "pair_pnl", "family_pnl"}:
        _finite(
            metrics[key],
            f"carry_state.metrics.{key}",
            minimum=None if key in signed_metric_keys else 0,
        )
    for dimension in ("pair_pnl", "family_pnl"):
        dimension_rows = _mapping(
            metrics[dimension], f"carry_state.metrics.{dimension}"
        )
        for key, raw_bucket in dimension_rows.items():
            _identifier(key, f"carry_state.metrics.{dimension}.key")
            bucket = _mapping(raw_bucket, f"carry_state.metrics.{dimension}.{key}")
            _exact(
                bucket,
                frozenset({"realized_pnl_jpy", "financing_cost_jpy", "net_pnl_jpy"}),
                f"carry_state.metrics.{dimension}.{key}",
            )
            _finite(bucket["realized_pnl_jpy"], "carry pnl realized")
            _finite(bucket["financing_cost_jpy"], "carry pnl financing", minimum=0)
            _finite(bucket["net_pnl_jpy"], "carry pnl net")
            _assert_close(
                bucket["net_pnl_jpy"],
                bucket["realized_pnl_jpy"] - bucket["financing_cost_jpy"],
                "carry pnl bucket accounting",
            )
    rejection_counts = _mapping(
        metrics["rejection_counts"], "carry_state.metrics.rejection_counts"
    )
    for key, value in rejection_counts.items():
        _identifier(key, "carry rejection reason")
        _integer(value, "carry rejection count")
    last_coordinate = _sequence(row["last_coordinate"], "carry_state.last_coordinate")
    if len(last_coordinate) != 3:
        raise DojoPortfolioReplayError(
            "carry last_coordinate must be [epoch, rank, intrabar]"
        )
    _integer(last_coordinate[0], "carry last coordinate epoch")
    rank = _integer(last_coordinate[1], "carry last coordinate rank")
    if rank > 3 or last_coordinate[2] not in _INTRABAR_PHASE_ORDER:
        raise DojoPortfolioReplayError("carry last coordinate rank/path is invalid")
    _integer(row["last_quote_watermark"], "carry last_quote_watermark")
    _integer(row["coordinate_seq"], "carry coordinate_seq", minimum=1)
    _integer(row["last_epoch"], "carry last_epoch")
    _integer(row["start_epoch"], "carry start_epoch")
    _positive(row["start_balance_jpy"], "carry start_balance_jpy")
    _positive(row["start_equity_jpy"], "carry start_equity_jpy")
    _positive(row["peak_equity_jpy"], "carry peak_equity_jpy")
    _sha(row["event_chain_sha256"], "carry event_chain_sha256")
    _integer(row["event_count"], "carry event_count")
    state = {
        "policy_sha256": row["policy_sha256"],
        "balance_jpy": _finite(row["balance_jpy"], "carry_state.balance_jpy"),
        "accrued_financing_jpy": _finite(
            row["accrued_financing_jpy"], "carry_state.accrued_financing_jpy", minimum=0
        ),
        "positions": positions,
        "pending_orders": pending_orders,
        "last_quotes": last_quotes,
        "last_epoch": row["last_epoch"],
        "last_coordinate": row["last_coordinate"],
        "last_quote_watermark": row["last_quote_watermark"],
        "coordinate_seq": row["coordinate_seq"],
        "start_epoch": row["start_epoch"],
        "start_balance_jpy": row["start_balance_jpy"],
        "start_equity_jpy": row["start_equity_jpy"],
        "peak_equity_jpy": row["peak_equity_jpy"],
        # Ruin is a transition, not a sticky reporting boolean.  Its active
        # state is determined by the carried executable MTM at the handoff.
        "ruin_active": float(row["equity_jpy"]) <= 0,
        "event_chain_sha256": row["event_chain_sha256"],
        "event_count": row["event_count"],
        "metrics": _copy(metrics),
    }
    _assert_close(
        state["balance_jpy"],
        state["start_balance_jpy"]
        + state["metrics"]["realized_pnl_jpy"]
        - state["metrics"]["financing_cost_jpy"],
        "carry balance accounting",
    )
    equity = _account_equity(state, state["last_quotes"], _policy_indexes(policy))
    _assert_close(equity, row["equity_jpy"], "carry_state.equity_jpy")
    if not reset_reporting_window:
        # Intra-job resume must preserve the exact reporting denominator and
        # accumulated economics.  The ordinary month-to-month carry path below
        # intentionally resets them; keeping this opt-in branch separate
        # preserves that V1 behavior.
        return state
    # A carry transfers economic state, not the preceding reporting window.
    # Long-horizon month cells must start at their own month-open MTM equity and
    # report only this segment's fills/costs/drawdown while preserving positions,
    # orders, financing balance and the append-only event chain.
    segment_metrics = _empty_metrics()
    segment_metrics["minimum_mtm_equity_jpy"] = equity
    month_open_exposure = _exposure(
        list(state["positions"].values()),
        list(state["pending_orders"].values()),
        state["last_quotes"],
        _policy_indexes(policy),
        policy,
    )
    month_open_margin = float(month_open_exposure["margin"])
    segment_metrics["minimum_free_margin_jpy"] = equity - month_open_margin
    segment_metrics["peak_margin_jpy"] = month_open_margin
    segment_metrics["peak_margin_usage_fraction"] = (
        month_open_margin / equity if equity > 0 else 1.0
    )
    state["start_epoch"] = None
    state["start_balance_jpy"] = state["balance_jpy"]
    state["start_equity_jpy"] = equity
    state["peak_equity_jpy"] = equity
    state["metrics"] = segment_metrics
    return state


def _carry_artifact(
    state: Mapping[str, Any],
    quotes: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "contract": PORTFOLIO_CARRY_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "policy_sha256": state["policy_sha256"],
        "balance_jpy": state["balance_jpy"],
        "equity_jpy": _account_equity(state, quotes, indexes),
        "accrued_financing_jpy": state["accrued_financing_jpy"],
        "positions": sorted(
            (_copy(row) for row in state["positions"].values()),
            key=lambda row: row["position_id"],
        ),
        "pending_orders": sorted(
            (_copy(row) for row in state["pending_orders"].values()),
            key=lambda row: row["order_id"],
        ),
        "last_quotes": sorted(
            (_copy(row) for row in quotes.values()), key=lambda row: row["pair"]
        ),
        "last_epoch": state["last_epoch"],
        "last_coordinate": state["last_coordinate"],
        "last_quote_watermark": state["last_quote_watermark"],
        "coordinate_seq": state["coordinate_seq"],
        "start_epoch": state["start_epoch"],
        "start_balance_jpy": state["start_balance_jpy"],
        "start_equity_jpy": state["start_equity_jpy"],
        "peak_equity_jpy": state["peak_equity_jpy"],
        "event_chain_sha256": state["event_chain_sha256"],
        "event_count": state["event_count"],
        "metrics": _copy(state["metrics"]),
    }
    body["carry_state_sha256"] = canonical_portfolio_sha256(body)
    return body


class PortfolioReplaySession:
    """Incremental, bounded-memory shared-account replay session.

    A session validates policy/carry and builds policy indexes once.  Each quote
    coordinate is a strict two-step transaction: ``prepare_coordinate`` advances
    reducer-owned economics and returns the exact worker snapshot;
    ``consume_proposal_batch`` verifies the all-worker acknowledgement and
    completes allocation.  Only one prepared coordinate may exist at a time.
    """

    def __init__(
        self,
        *,
        policy: Mapping[str, Any],
        initial_balance_jpy: int | float | None = None,
        carry_state: Mapping[str, Any] | None = None,
    ) -> None:
        self.policy = verify_portfolio_policy(policy)
        self.indexes = _policy_indexes(self.policy)
        if carry_state is None:
            if initial_balance_jpy is None:
                raise DojoPortfolioReplayError(
                    "initial_balance_jpy is required without carry_state"
                )
            self.state = _new_state(
                _positive(initial_balance_jpy, "initial_balance_jpy"),
                self.policy["policy_sha256"],
            )
        else:
            if initial_balance_jpy is not None:
                raise DojoPortfolioReplayError(
                    "initial_balance_jpy and carry_state are mutually exclusive"
                )
            self.state = _state_from_carry(carry_state, self.policy)
        self._prepared: dict[str, Any] | None = None
        self._processed_coordinate_count = 0
        self._final_result: dict[str, Any] | None = None
        self._terminal_policy: str | None = None

    def prepare_coordinate(
        self,
        *,
        coordinate_id: str,
        epoch: int,
        phase: str,
        intrabar: str,
        quote_watermark: int,
        quotes: Sequence[Mapping[str, Any]],
        quote_batch_sha256_value: str | None = None,
    ) -> dict[str, Any]:
        """Advance through exits and return the reducer-owned worker snapshot."""

        if self._final_result is not None:
            raise DojoPortfolioReplayError("a finalized session cannot consume quotes")
        if self._prepared is not None:
            raise DojoPortfolioReplayError(
                "the prior prepared coordinate requires one proposal batch"
            )
        epoch_value = _integer(epoch, "coordinate.epoch")
        if phase not in {"O", "H", "L", "C"}:
            raise DojoPortfolioReplayError("coordinate.phase is unsupported")
        if intrabar not in _INTRABAR_PHASE_ORDER:
            raise DojoPortfolioReplayError("coordinate.intrabar is unsupported")
        watermark = _integer(quote_watermark, "coordinate.quote_watermark")
        coordinate_name = _identifier(coordinate_id, "coordinate.coordinate_id")
        computed_digest = quote_batch_sha256(
            epoch=epoch_value,
            phase=phase,
            intrabar=intrabar,
            quote_watermark=watermark,
            quotes=quotes,
        )
        if quote_batch_sha256_value is not None and _sha(
            quote_batch_sha256_value, "coordinate.quote_batch_sha256"
        ) != computed_digest:
            raise DojoPortfolioReplayError("quote batch digest mismatch")

        # Validate the complete mark set and timestamp coordinate before state
        # mutation.  The placeholder account/state are intentionally empty;
        # reducer-owned economic state is sealed only after exits below.
        try:
            market_guard = seal_post_exit_snapshot(
                {
                    "coordinate_id": coordinate_name,
                    "epoch": epoch_value,
                    "phase": phase,
                    "intrabar": intrabar,
                    "quote_batch_sha256": computed_digest,
                    "quote_watermark": watermark,
                    "expected_quote_pairs": self.policy["expected_quote_pairs"],
                    "active_worker_bindings": self.policy["active_worker_bindings"],
                    "account": {
                        "balance_jpy": 0.0,
                        "equity_jpy": 0.0,
                        "margin_used_jpy": 0.0,
                        "accrued_financing_jpy": 0.0,
                    },
                    "quotes": list(quotes),
                    "positions": [],
                    "pending_orders": [],
                }
            )
        except ProtocolViolation as exc:
            raise DojoPortfolioReplayError(f"invalid quote coordinate: {exc}") from exc
        normalized_quotes = _quote_map(market_guard)

        phase_rank = _INTRABAR_PHASE_ORDER[intrabar][phase]
        last_coordinate = self.state["last_coordinate"]
        if last_coordinate is None:
            if phase != "O":
                raise DojoPortfolioReplayError(
                    "a replay without prior coordinate must begin at O"
                )
        else:
            last_epoch, last_rank, last_intrabar = last_coordinate
            if int(last_epoch) == epoch_value and last_intrabar != intrabar:
                raise DojoPortfolioReplayError(
                    "intrabar path changed within one candle epoch"
                )
            if int(last_epoch) == epoch_value and phase_rank != int(last_rank) + 1:
                raise DojoPortfolioReplayError(
                    "coordinate phase must be the exact next phase of the sealed intrabar path"
                )
            if int(last_epoch) < epoch_value and (
                int(last_rank) != 3 or phase != "O"
            ):
                raise DojoPortfolioReplayError(
                    "a new candle requires a completed prior C phase and must begin at O"
                )
            if int(last_epoch) > epoch_value:
                raise DojoPortfolioReplayError("coordinate epoch moved backwards")
        if (
            self.state["last_quote_watermark"] is not None
            and watermark <= self.state["last_quote_watermark"]
        ):
            raise DojoPortfolioReplayError(
                "quote watermark must be strictly increasing"
            )

        self.state["coordinate_seq"] += 1
        previous_epoch = self.state["last_epoch"]
        if previous_epoch is not None:
            elapsed = epoch_value - int(previous_epoch)
            locked = _exposure(
                list(self.state["positions"].values()),
                list(self.state["pending_orders"].values()),
                self.state["last_quotes"],
                self.indexes,
                self.policy,
            )["margin"]
            self.state["metrics"]["capital_lock_margin_jpy_seconds"] += (
                locked * elapsed
            )
        carry_charges = _financing_charges(self.state, epoch_value, self.indexes)
        _expire_old_pending(self.state, epoch_value)
        _process_system_exits(
            self.state, epoch_value, normalized_quotes, self.indexes
        )
        frozen_trigger_candidates = _triggered_pending_candidates(
            self.state,
            epoch=epoch_value,
            coordinate_seq=self.state["coordinate_seq"],
            quotes=normalized_quotes,
            indexes=self.indexes,
            policy=self.policy,
        )
        _book_financing(self.state, carry_charges)
        # Preserve the actual stressed state which causes forced liquidation.
        # Recording only after closeout would erase the low equity/free margin
        # and peak usage observed at the executable current quote.
        _record_account_state(
            self.state, normalized_quotes, self.indexes, self.policy
        )
        _process_margin_closeout(
            self.state, normalized_quotes, self.indexes, self.policy
        )
        _record_account_state(
            self.state, normalized_quotes, self.indexes, self.policy
        )

        computed_equity = _account_equity(
            self.state, normalized_quotes, self.indexes
        )
        open_margin = sum(
            _margin(row, normalized_quotes, self.indexes, self.policy)
            for row in self.state["positions"].values()
        )
        try:
            snapshot = seal_post_exit_snapshot(
                {
                    "coordinate_id": coordinate_name,
                    "epoch": epoch_value,
                    "phase": phase,
                    "intrabar": intrabar,
                    "quote_batch_sha256": computed_digest,
                    "quote_watermark": watermark,
                    "expected_quote_pairs": self.policy["expected_quote_pairs"],
                    "active_worker_bindings": self.policy["active_worker_bindings"],
                    "account": {
                        "balance_jpy": self.state["balance_jpy"],
                        "equity_jpy": computed_equity,
                        "margin_used_jpy": open_margin,
                        "accrued_financing_jpy": self.state[
                            "accrued_financing_jpy"
                        ],
                    },
                    "quotes": list(market_guard["quotes"]),
                    "positions": [
                        _project_position(row)
                        for row in self.state["positions"].values()
                    ],
                    "pending_orders": [
                        _project_order(row)
                        for row in self.state["pending_orders"].values()
                    ],
                }
            )
        except ProtocolViolation as exc:  # pragma: no cover - internal invariant
            raise DojoPortfolioReplayError(
                f"reducer produced an invalid post-exit snapshot: {exc}"
            ) from exc
        self._prepared = {
            "snapshot": snapshot,
            "quotes": normalized_quotes,
            "epoch": epoch_value,
            "phase_rank": phase_rank,
            "intrabar": intrabar,
            "watermark": watermark,
            "frozen_trigger_candidates": frozen_trigger_candidates,
        }
        return _copy(snapshot)

    def consume_proposal_batch(
        self, proposal_batch: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Atomically consume the exact all-worker batch for the prepared view."""

        if self._prepared is None:
            raise DojoPortfolioReplayError(
                "prepare_coordinate must precede proposal consumption"
            )
        prepared = self._prepared
        snapshot = prepared["snapshot"]
        quotes = prepared["quotes"]
        try:
            batch = verify_worker_proposal_batch(snapshot, proposal_batch)
        except ProtocolViolation as exc:
            raise DojoPortfolioReplayError(
                f"invalid all-worker proposal batch: {exc}"
            ) from exc
        epoch = int(prepared["epoch"])
        if self.state["start_epoch"] is None:
            self.state["start_epoch"] = epoch
        before = {
            key: self.state["metrics"][key]
            for key in ("attempts", "fills", "rejections")
        }
        _record_account_state(self.state, quotes, self.indexes, self.policy)
        _event(
            self.state,
            "POST_EXIT_SNAPSHOT_ACK",
            {
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "batch_sha256": batch["batch_sha256"],
            },
        )
        _apply_actual_risk_reductions(self.state, batch, quotes, self.indexes)
        candidates = list(prepared["frozen_trigger_candidates"])
        candidates.extend(
            _candidate_from_intent(
                proposal, intent, quotes, self.indexes, self.policy, epoch
            )
            for proposal in batch["proposals"]
            for intent in proposal["new_risk_intents"]
        )
        candidates.sort(
            key=lambda row: (
                row["stop_risk_jpy"],
                row["margin_jpy"],
                row["family_id"],
                row["worker_id"],
                row["intent_id"],
            )
        )
        self.state["metrics"]["attempts"] += len(candidates)
        for candidate in candidates:
            equity = _account_equity(self.state, quotes, self.indexes)
            reason = _allocation_rejection(
                candidate,
                [],
                list(self.state["positions"].values()),
                list(self.state["pending_orders"].values()),
                quotes,
                self.indexes,
                self.policy,
                equity,
            )
            if reason is not None:
                metrics = self.state["metrics"]
                metrics["rejections"] += 1
                if reason == "MARGIN_CAP":
                    metrics["margin_reject_count"] += 1
                counts = metrics["rejection_counts"]
                counts[reason] = counts.get(reason, 0) + 1
                _event(
                    self.state,
                    "ADMISSION_REJECT",
                    {
                        "worker_id": candidate["worker_id"],
                        "intent_id": candidate["intent_id"],
                        "source": candidate["source"],
                        "reason": reason,
                    },
                )
                continue
            _place_accepted_candidate(
                self.state,
                candidate,
                quotes,
                self.indexes,
                epoch,
                self.state["coordinate_seq"],
            )
        equity, reserved_margin, free_margin = _record_account_state(
            self.state, quotes, self.indexes, self.policy
        )
        self.state["last_quotes"] = quotes
        self.state["last_epoch"] = epoch
        self.state["last_coordinate"] = [
            epoch,
            prepared["phase_rank"],
            prepared["intrabar"],
        ]
        self.state["last_quote_watermark"] = prepared["watermark"]
        self._processed_coordinate_count += 1
        self._prepared = None
        body = {
            "contract": PORTFOLIO_COORDINATE_RECEIPT_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "policy_sha256": self.policy["policy_sha256"],
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "proposal_batch_sha256": batch["batch_sha256"],
            "quote_batch_sha256": snapshot["quote_batch_sha256"],
            "epoch": epoch,
            "phase": snapshot["phase"],
            "quote_watermark": snapshot["quote_watermark"],
            "ending_balance_jpy": self.state["balance_jpy"],
            "ending_equity_jpy": equity,
            "reserved_margin_jpy": reserved_margin,
            "free_margin_jpy": free_margin,
            "attempt_delta": self.state["metrics"]["attempts"]
            - before["attempts"],
            "entry_fill_delta": self.state["metrics"]["fills"] - before["fills"],
            "rejection_delta": self.state["metrics"]["rejections"]
            - before["rejections"],
            "event_count": self.state["event_count"],
            "event_chain_sha256": self.state["event_chain_sha256"],
            "live_permission": False,
            "broker_mutation_allowed": False,
        }
        body["coordinate_receipt_sha256"] = canonical_portfolio_sha256(body)
        return body

    def export_checkpoint(self) -> dict[str, Any]:
        """Export an exact, causal intra-job resume boundary.

        Unlike the month-to-month carry contract, this checkpoint deliberately
        preserves the current reporting window, accumulated metrics, event
        chain, and processed-coordinate denominator.  Export is allowed only
        after a complete coordinate transaction, never between quote
        preparation and proposal allocation and never after finalization.
        """

        if self._prepared is not None:
            raise DojoPortfolioReplayError(
                "cannot checkpoint an unacknowledged prepared snapshot"
            )
        if self._final_result is not None:
            raise DojoPortfolioReplayError("cannot checkpoint a finalized session")
        state_kind = "ACTIVE_EXACT_STATE"
        initial_balance: float | None = None
        carry: dict[str, Any] | None
        if self.state["last_coordinate"] is None:
            if self._processed_coordinate_count != 0:
                raise DojoPortfolioReplayError(
                    "fresh checkpoint has a non-zero processed denominator"
                )
            state_kind = "FRESH_INITIAL_BALANCE"
            initial_balance = float(self.state["balance_jpy"])
            expected = _new_state(initial_balance, self.policy["policy_sha256"])
            if self.state != expected:
                raise DojoPortfolioReplayError(
                    "fresh checkpoint state differs from the canonical origin"
                )
            carry = None
        else:
            carry = _carry_artifact(
                self.state,
                self.state["last_quotes"],
                self.indexes,
            )
        body: dict[str, Any] = {
            "contract": PORTFOLIO_CHECKPOINT_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "policy_sha256": self.policy["policy_sha256"],
            "processed_coordinate_count": self._processed_coordinate_count,
            "state_kind": state_kind,
            "initial_balance_jpy": initial_balance,
            "exact_carry_state": carry,
            "terminal": False,
            "live_permission": False,
            "broker_mutation_allowed": False,
            "order_authority": "NONE",
        }
        body["checkpoint_sha256"] = canonical_portfolio_sha256(body)
        return _copy(body)

    @classmethod
    def restore_checkpoint(
        cls,
        *,
        policy: Mapping[str, Any],
        checkpoint: Mapping[str, Any],
    ) -> PortfolioReplaySession:
        """Restore exactly the state exported by :meth:`export_checkpoint`."""

        verified_policy = verify_portfolio_policy(policy)
        verified_checkpoint = verify_portfolio_replay_checkpoint(
            policy=verified_policy,
            checkpoint=checkpoint,
        )
        return _restore_portfolio_replay_checkpoint(
            policy=verified_policy,
            checkpoint=verified_checkpoint,
        )

    def finalize(
        self,
        *,
        terminal_policy: str = MONTH_END_MTM_WITH_STATE_HANDOFF,
    ) -> dict[str, Any]:
        """Finalize current segment, optionally flattening at the last mark."""

        if terminal_policy not in {
            MONTH_END_FLAT_SETTLEMENT,
            MONTH_END_MTM_WITH_STATE_HANDOFF,
        }:
            raise DojoPortfolioReplayError("terminal_policy is unsupported")
        if self._prepared is not None:
            raise DojoPortfolioReplayError(
                "cannot finalize with an unacknowledged prepared snapshot"
            )
        if self._processed_coordinate_count == 0:
            raise DojoPortfolioReplayError("cannot finalize an empty replay segment")
        if self._final_result is not None:
            if terminal_policy != self._terminal_policy:
                raise DojoPortfolioReplayError(
                    "a finalized session cannot change terminal policy"
                )
            return _copy(self._final_result)
        quotes = self.state["last_quotes"]
        if terminal_policy == MONTH_END_FLAT_SETTLEMENT:
            for order_id in sorted(list(self.state["pending_orders"])):
                del self.state["pending_orders"][order_id]
                self.state["metrics"]["orders_cancelled"] += 1
                _event(
                    self.state,
                    "ORDER_CANCEL",
                    {"order_id": order_id, "intent_id": "TERMINAL_SETTLEMENT"},
                )
            for position_id in sorted(list(self.state["positions"])):
                _close_position(
                    self.state,
                    position_id,
                    None,
                    "TERMINAL_SETTLEMENT",
                    quotes,
                    self.indexes,
                )
            _record_account_state(self.state, quotes, self.indexes, self.policy)
        self._terminal_policy = terminal_policy
        self._final_result = self._build_result(terminal_policy=terminal_policy)
        return _copy(self._final_result)

    def _build_result(self, *, terminal_policy: str) -> dict[str, Any]:
        state = self.state
        quotes = state["last_quotes"]
        end_equity = _account_equity(state, quotes, self.indexes)
        period_seconds = int(state["last_epoch"] - state["start_epoch"])
        period_multiple = end_equity / float(state["start_equity_jpy"])
        monthly_multiple = (
            period_multiple ** (MONTH_SECONDS / period_seconds)
            if period_seconds > 0 and period_multiple > 0
            else None
        )
        metrics = state["metrics"]
        carry = _carry_artifact(state, quotes, self.indexes)
        economic_failures = []
        if metrics["fills"] == 0:
            economic_failures.append("ZERO_FILLS")
        if metrics["margin_closeouts"] > 0:
            economic_failures.append("MARGIN_CLOSEOUT_OCCURRED")
        if end_equity <= 0:
            economic_failures.append("NON_POSITIVE_END_EQUITY")
        transaction_cost = metrics["spread_cost_jpy"] + metrics["slippage_cost_jpy"]
        result = {
            "contract": PORTFOLIO_REPLAY_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE"
            if not economic_failures
            else "COMPLETE_WITH_ECONOMIC_FAILURES",
            "economic_failure_codes": economic_failures,
            "policy_sha256": self.policy["policy_sha256"],
            "terminal_policy": terminal_policy,
            "processed_coordinate_count": self._processed_coordinate_count,
            "start_epoch": state["start_epoch"],
            "end_epoch": state["last_epoch"],
            "duration_seconds": period_seconds,
            "start_balance_jpy": state["start_balance_jpy"],
            "end_balance_jpy": state["balance_jpy"],
            "start_equity_jpy": state["start_equity_jpy"],
            "end_equity_jpy": end_equity,
            "minimum_mtm_equity_jpy": metrics["minimum_mtm_equity_jpy"],
            "minimum_free_margin_jpy": metrics["minimum_free_margin_jpy"],
            "period_multiple_mtm": period_multiple,
            "monthly_multiple_mtm": monthly_multiple,
            "max_drawdown_fraction": metrics["max_drawdown_fraction"],
            "peak_margin_jpy": metrics["peak_margin_jpy"],
            "peak_margin_usage_fraction": metrics["peak_margin_usage_fraction"],
            "margin_closeouts": metrics["margin_closeouts"],
            "margin_reject_count": metrics["margin_reject_count"],
            "ruin_event_count": metrics["ruin_event_count"],
            "realized_pnl_jpy": metrics["realized_pnl_jpy"],
            "financing_cost_jpy": metrics["financing_cost_jpy"],
            "spread_cost_jpy": metrics["spread_cost_jpy"],
            "slippage_cost_jpy": metrics["slippage_cost_jpy"],
            "transaction_cost_jpy": transaction_cost,
            "attempts": metrics["attempts"],
            "fills": metrics["fills"],
            "execution_fill_count": metrics["fills"] + metrics["position_closes"],
            "trade_count": metrics["position_closes"],
            "rejections": metrics["rejections"],
            "rejection_counts": dict(sorted(metrics["rejection_counts"].items())),
            "orders_placed": metrics["orders_placed"],
            "orders_triggered": metrics["orders_triggered"],
            "orders_expired": metrics["orders_expired"],
            "orders_cancelled": metrics["orders_cancelled"],
            "position_closes": metrics["position_closes"],
            "pair_pnl_jpy": [
                dict({"pair": key}, **value)
                for key, value in sorted(metrics["pair_pnl"].items())
            ],
            "family_pnl_jpy": [
                dict({"family_id": key}, **value)
                for key, value in sorted(metrics["family_pnl"].items())
            ],
            "capital_lock_margin_jpy_hours": metrics[
                "capital_lock_margin_jpy_seconds"
            ]
            / 3600.0,
            "open_position_count": len(state["positions"]),
            "pending_order_count": len(state["pending_orders"]),
            "event_count": state["event_count"],
            "event_chain_sha256": state["event_chain_sha256"],
            "carry_state_sha256": carry["carry_state_sha256"],
            "carry_state": carry,
            "raw_quote_events_included": False,
            "live_permission": False,
            "broker_mutation_allowed": False,
            "three_x_guaranteed": False,
        }
        result["result_sha256"] = canonical_portfolio_sha256(result)
        return result


_CHECKPOINT_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "policy_sha256",
        "processed_coordinate_count",
        "state_kind",
        "initial_balance_jpy",
        "exact_carry_state",
        "terminal",
        "live_permission",
        "broker_mutation_allowed",
        "order_authority",
        "checkpoint_sha256",
    }
)


def _restore_portfolio_replay_checkpoint(
    *,
    policy: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> PortfolioReplaySession:
    """Construct a session from an already verified exact checkpoint."""

    row = checkpoint
    if row["state_kind"] == "FRESH_INITIAL_BALANCE":
        session = PortfolioReplaySession(
            policy=policy,
            initial_balance_jpy=row["initial_balance_jpy"],
        )
    else:
        session = PortfolioReplaySession.__new__(PortfolioReplaySession)
        session.policy = verify_portfolio_policy(policy)
        session.indexes = _policy_indexes(session.policy)
        session.state = _state_from_carry(
            row["exact_carry_state"],
            session.policy,
            reset_reporting_window=False,
        )
        session._prepared = None
        session._final_result = None
        session._terminal_policy = None
    session._processed_coordinate_count = row["processed_coordinate_count"]
    return session


def verify_portfolio_replay_checkpoint(
    *,
    policy: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify one exact intra-job checkpoint without changing V1 carry semantics."""

    verified_policy = verify_portfolio_policy(policy)
    row = dict(_mapping(checkpoint, "checkpoint"))
    _exact(row, _CHECKPOINT_KEYS, "checkpoint")
    unsigned = {key: item for key, item in row.items() if key != "checkpoint_sha256"}
    if (
        row["contract"] != PORTFOLIO_CHECKPOINT_CONTRACT
        or row["schema_version"] != SCHEMA_VERSION
        or _sha(row["policy_sha256"], "checkpoint.policy_sha256")
        != verified_policy["policy_sha256"]
        or canonical_portfolio_sha256(unsigned)
        != _sha(row["checkpoint_sha256"], "checkpoint.checkpoint_sha256")
        or row["terminal"] is not False
        or row["live_permission"] is not False
        or row["broker_mutation_allowed"] is not False
        or row["order_authority"] != "NONE"
    ):
        raise DojoPortfolioReplayError(
            "checkpoint contract, policy, digest, or authority boundary is invalid"
        )
    processed = _integer(
        row["processed_coordinate_count"],
        "checkpoint.processed_coordinate_count",
    )
    state_kind = row["state_kind"]
    if state_kind == "FRESH_INITIAL_BALANCE":
        if row["exact_carry_state"] is not None or processed != 0:
            raise DojoPortfolioReplayError("fresh checkpoint shape is invalid")
        _positive(row["initial_balance_jpy"], "checkpoint.initial_balance_jpy")
    elif state_kind == "ACTIVE_EXACT_STATE":
        if row["initial_balance_jpy"] is not None:
            raise DojoPortfolioReplayError("active checkpoint shape is invalid")
        state = _state_from_carry(
            _mapping(row["exact_carry_state"], "checkpoint.exact_carry_state"),
            verified_policy,
            reset_reporting_window=False,
        )
        if state["last_coordinate"] is None or processed > state["coordinate_seq"]:
            raise DojoPortfolioReplayError(
                "active checkpoint coordinate denominator is impossible"
            )
    else:
        raise DojoPortfolioReplayError("checkpoint.state_kind is unsupported")
    restored = _restore_portfolio_replay_checkpoint(
        policy=verified_policy,
        checkpoint=row,
    )
    rebuilt = restored.export_checkpoint()
    if rebuilt != row:
        raise DojoPortfolioReplayError("checkpoint is not the canonical exact state")
    return _copy(row)


def _legacy_reduce_portfolio_replay(
    *,
    policy: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
    initial_balance_jpy: int | float | None = None,
    carry_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reduce ordered quote/snapshot/proposal frames into one compact account result."""

    verified_policy = verify_portfolio_policy(policy)
    if carry_state is None:
        if initial_balance_jpy is None:
            raise DojoPortfolioReplayError(
                "initial_balance_jpy is required without carry_state"
            )
        state = _new_state(
            _positive(initial_balance_jpy, "initial_balance_jpy"),
            verified_policy["policy_sha256"],
        )
    else:
        if initial_balance_jpy is not None:
            raise DojoPortfolioReplayError(
                "initial_balance_jpy and carry_state are mutually exclusive"
            )
        state = _state_from_carry(carry_state, verified_policy)
    frame_rows = _sequence(frames, "frames")
    if not frame_rows:
        raise DojoPortfolioReplayError("frames must not be empty")
    indexes = _policy_indexes(verified_policy)

    for frame_index, raw_frame in enumerate(frame_rows):
        frame = _mapping(raw_frame, f"frames[{frame_index}]")
        _exact(
            frame,
            frozenset({"post_exit_snapshot", "proposal_batch"}),
            f"frames[{frame_index}]",
        )
        try:
            snapshot = verify_post_exit_snapshot(frame["post_exit_snapshot"])
        except ProtocolViolation as exc:
            raise DojoPortfolioReplayError(
                f"invalid post-exit snapshot: {exc}"
            ) from exc
        if snapshot["expected_quote_pairs"] != verified_policy["expected_quote_pairs"]:
            raise DojoPortfolioReplayError(
                "snapshot quote coverage differs from policy"
            )
        if (
            snapshot["active_worker_bindings"]
            != verified_policy["active_worker_bindings"]
        ):
            raise DojoPortfolioReplayError("snapshot active workers differ from policy")
        digest = quote_batch_sha256(
            epoch=snapshot["epoch"],
            phase=snapshot["phase"],
            intrabar=snapshot["intrabar"],
            quote_watermark=snapshot["quote_watermark"],
            quotes=snapshot["quotes"],
        )
        if digest != snapshot["quote_batch_sha256"]:
            raise DojoPortfolioReplayError("quote batch digest mismatch")
        epoch = int(snapshot["epoch"])
        phase_rank = _INTRABAR_PHASE_ORDER[snapshot["intrabar"]][snapshot["phase"]]
        if state["last_coordinate"] is None:
            if snapshot["phase"] != "O":
                raise DojoPortfolioReplayError(
                    "a replay without prior coordinate must begin at O"
                )
        else:
            last_epoch, last_rank, last_intrabar = state["last_coordinate"]
            if int(last_epoch) == epoch and last_intrabar != snapshot["intrabar"]:
                raise DojoPortfolioReplayError(
                    "intrabar path changed within one candle epoch"
                )
            if int(last_epoch) == epoch and phase_rank != int(last_rank) + 1:
                raise DojoPortfolioReplayError(
                    "coordinate phase must be the exact next phase of the sealed intrabar path"
                )
            if int(last_epoch) < epoch and (
                int(last_rank) != 3 or snapshot["phase"] != "O"
            ):
                raise DojoPortfolioReplayError(
                    "a new candle requires a completed prior C phase and must begin at O"
                )
            if int(last_epoch) > epoch:
                raise DojoPortfolioReplayError("coordinate epoch moved backwards")
        if (
            state["last_quote_watermark"] is not None
            and snapshot["quote_watermark"] <= state["last_quote_watermark"]
        ):
            raise DojoPortfolioReplayError(
                "quote watermark must be strictly increasing"
            )
        state["coordinate_seq"] += 1
        quotes = _quote_map(snapshot)

        previous_epoch = state["last_epoch"]
        previous_quotes = state["last_quotes"]
        if previous_epoch is not None:
            elapsed = epoch - int(previous_epoch)
            locked = _exposure(
                list(state["positions"].values()),
                list(state["pending_orders"].values()),
                previous_quotes,
                indexes,
                verified_policy,
            )["margin"]
            state["metrics"]["capital_lock_margin_jpy_seconds"] += locked * elapsed
        carry_charges = _financing_charges(state, epoch, indexes)
        _expire_old_pending(state, epoch)
        _process_system_exits(state, epoch, quotes, indexes)
        # Trigger causality is frozen before anything worker-visible exists at
        # this coordinate.  Triggered orders are removed from the cancellable
        # snapshot and their exact reducer-priced candidate is committed to the
        # event chain.  Workers can neither observe the trigger and escape it nor
        # mutate its fill claim; only central admission remains pending.
        frozen_trigger_candidates = _triggered_pending_candidates(
            state,
            epoch=epoch,
            coordinate_seq=state["coordinate_seq"],
            quotes=quotes,
            indexes=indexes,
            policy=verified_policy,
        )
        _book_financing(state, carry_charges)
        _process_margin_closeout(state, quotes, indexes, verified_policy)

        computed_equity = _account_equity(state, quotes, indexes)
        computed_margin = sum(
            _margin(row, quotes, indexes, verified_policy)
            for row in state["positions"].values()
        )
        _assert_close(
            state["balance_jpy"],
            snapshot["account"]["balance_jpy"],
            "snapshot.account.balance_jpy",
        )
        _assert_close(
            computed_equity,
            snapshot["account"]["equity_jpy"],
            "snapshot.account.equity_jpy",
        )
        _assert_close(
            computed_margin,
            snapshot["account"]["margin_used_jpy"],
            "snapshot.account.margin_used_jpy",
        )
        _assert_close(
            state["accrued_financing_jpy"],
            snapshot["account"]["accrued_financing_jpy"],
            "snapshot.account.accrued_financing_jpy",
        )
        _assert_rows(
            [_project_position(row) for row in state["positions"].values()],
            snapshot["positions"],
            "position_id",
            "snapshot.positions",
        )
        _assert_rows(
            [_project_order(row) for row in state["pending_orders"].values()],
            snapshot["pending_orders"],
            "order_id",
            "snapshot.pending_orders",
        )
        try:
            batch = verify_worker_proposal_batch(snapshot, frame["proposal_batch"])
        except ProtocolViolation as exc:
            raise DojoPortfolioReplayError(
                f"invalid all-worker proposal batch: {exc}"
            ) from exc

        if state["start_epoch"] is None:
            state["start_epoch"] = epoch
        _record_equity(state, computed_equity)
        state["metrics"]["peak_margin_jpy"] = max(
            state["metrics"]["peak_margin_jpy"], computed_margin
        )
        _event(
            state,
            "POST_EXIT_SNAPSHOT_ACK",
            {
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "batch_sha256": batch["batch_sha256"],
            },
        )

        _apply_actual_risk_reductions(state, batch, quotes, indexes)
        # Frozen triggers and fresh intents enter one canonical pool only after
        # exits and worker risk reductions.  Trigger rows were already removed
        # before the worker snapshot, so same-coordinate cancellation is
        # structurally impossible.
        candidates = frozen_trigger_candidates
        candidates.extend(
            [
                _candidate_from_intent(
                    proposal, intent, quotes, indexes, verified_policy, epoch
                )
                for proposal in batch["proposals"]
                for intent in proposal["new_risk_intents"]
            ]
        )
        candidates.sort(
            key=lambda row: (
                row["stop_risk_jpy"],
                row["margin_jpy"],
                row["family_id"],
                row["worker_id"],
                row["intent_id"],
            )
        )
        state["metrics"]["attempts"] += len(candidates)
        for candidate in candidates:
            equity = _account_equity(state, quotes, indexes)
            reason = _allocation_rejection(
                candidate,
                [],
                list(state["positions"].values()),
                list(state["pending_orders"].values()),
                quotes,
                indexes,
                verified_policy,
                equity,
            )
            if reason is not None:
                state["metrics"]["rejections"] += 1
                counts = state["metrics"]["rejection_counts"]
                counts[reason] = counts.get(reason, 0) + 1
                _event(
                    state,
                    "ADMISSION_REJECT",
                    {
                        "worker_id": candidate["worker_id"],
                        "intent_id": candidate["intent_id"],
                        "source": candidate["source"],
                        "reason": reason,
                    },
                )
                continue
            _place_accepted_candidate(
                state,
                candidate,
                quotes,
                indexes,
                epoch,
                state["coordinate_seq"],
            )

        ending_equity_at_coordinate = _account_equity(state, quotes, indexes)
        ending_margin = sum(
            _margin(row, quotes, indexes, verified_policy)
            for row in state["positions"].values()
        )
        _record_equity(state, ending_equity_at_coordinate)
        state["metrics"]["peak_margin_jpy"] = max(
            state["metrics"]["peak_margin_jpy"], ending_margin
        )
        state["last_quotes"] = quotes
        state["last_epoch"] = epoch
        state["last_coordinate"] = [epoch, phase_rank, snapshot["intrabar"]]
        state["last_quote_watermark"] = snapshot["quote_watermark"]

    final_quotes = state["last_quotes"]
    end_equity = _account_equity(state, final_quotes, indexes)
    period_seconds = int(state["last_epoch"] - state["start_epoch"])
    period_multiple = end_equity / float(state["start_equity_jpy"])
    monthly_multiple = (
        period_multiple ** (MONTH_SECONDS / period_seconds)
        if period_seconds > 0 and period_multiple > 0
        else None
    )
    metrics = state["metrics"]
    carry = _carry_artifact(state, final_quotes, indexes)
    economic_failures = []
    if metrics["fills"] == 0:
        economic_failures.append("ZERO_FILLS")
    if metrics["margin_closeouts"] > 0:
        economic_failures.append("MARGIN_CLOSEOUT_OCCURRED")
    if end_equity <= 0:
        economic_failures.append("NON_POSITIVE_END_EQUITY")
    result = {
        "contract": PORTFOLIO_REPLAY_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE"
        if not economic_failures
        else "COMPLETE_WITH_ECONOMIC_FAILURES",
        "economic_failure_codes": economic_failures,
        "policy_sha256": verified_policy["policy_sha256"],
        "start_epoch": state["start_epoch"],
        "end_epoch": state["last_epoch"],
        "duration_seconds": period_seconds,
        "start_balance_jpy": state["start_balance_jpy"],
        "end_balance_jpy": state["balance_jpy"],
        "start_equity_jpy": state["start_equity_jpy"],
        "end_equity_jpy": end_equity,
        "period_multiple_mtm": period_multiple,
        "monthly_multiple_mtm": monthly_multiple,
        "max_drawdown_fraction": metrics["max_drawdown_fraction"],
        "peak_margin_jpy": metrics["peak_margin_jpy"],
        "margin_closeouts": metrics["margin_closeouts"],
        "realized_pnl_jpy": metrics["realized_pnl_jpy"],
        "financing_cost_jpy": metrics["financing_cost_jpy"],
        "spread_cost_jpy": metrics["spread_cost_jpy"],
        "slippage_cost_jpy": metrics["slippage_cost_jpy"],
        "attempts": metrics["attempts"],
        "fills": metrics["fills"],
        "rejections": metrics["rejections"],
        "rejection_counts": dict(sorted(metrics["rejection_counts"].items())),
        "orders_placed": metrics["orders_placed"],
        "orders_triggered": metrics["orders_triggered"],
        "orders_expired": metrics["orders_expired"],
        "orders_cancelled": metrics["orders_cancelled"],
        "position_closes": metrics["position_closes"],
        "pair_pnl_jpy": [
            dict({"pair": key}, **value)
            for key, value in sorted(metrics["pair_pnl"].items())
        ],
        "family_pnl_jpy": [
            dict({"family_id": key}, **value)
            for key, value in sorted(metrics["family_pnl"].items())
        ],
        "capital_lock_margin_jpy_hours": metrics["capital_lock_margin_jpy_seconds"]
        / 3600.0,
        "open_position_count": len(state["positions"]),
        "pending_order_count": len(state["pending_orders"]),
        "event_count": state["event_count"],
        "event_chain_sha256": state["event_chain_sha256"],
        "carry_state_sha256": carry["carry_state_sha256"],
        "carry_state": carry,
        "raw_quote_events_included": False,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "three_x_guaranteed": False,
    }
    result["result_sha256"] = canonical_portfolio_sha256(result)
    return result


def reduce_portfolio_replay(
    *,
    policy: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
    initial_balance_jpy: int | float | None = None,
    carry_state: Mapping[str, Any] | None = None,
    terminal_policy: str = MONTH_END_MTM_WITH_STATE_HANDOFF,
) -> dict[str, Any]:
    """Batch compatibility wrapper over :class:`PortfolioReplaySession`.

    Supplied post-exit snapshots are compared byte-semantically with the
    reducer-owned streaming snapshot before their proposal batch is consumed.
    This makes batch and incremental outputs exactly identical.
    """

    frame_rows = _sequence(frames, "frames")
    if not frame_rows:
        raise DojoPortfolioReplayError("frames must not be empty")
    session = PortfolioReplaySession(
        policy=policy,
        initial_balance_jpy=initial_balance_jpy,
        carry_state=carry_state,
    )
    for frame_index, raw_frame in enumerate(frame_rows):
        frame = _mapping(raw_frame, f"frames[{frame_index}]")
        _exact(
            frame,
            frozenset({"post_exit_snapshot", "proposal_batch"}),
            f"frames[{frame_index}]",
        )
        try:
            supplied = verify_post_exit_snapshot(frame["post_exit_snapshot"])
        except ProtocolViolation as exc:
            raise DojoPortfolioReplayError(
                f"invalid post-exit snapshot: {exc}"
            ) from exc
        prepared = session.prepare_coordinate(
            coordinate_id=supplied["coordinate_id"],
            epoch=supplied["epoch"],
            phase=supplied["phase"],
            intrabar=supplied["intrabar"],
            quote_watermark=supplied["quote_watermark"],
            quotes=supplied["quotes"],
            quote_batch_sha256_value=supplied["quote_batch_sha256"],
        )
        prepared_semantic = {
            key: value for key, value in prepared.items() if key != "snapshot_sha256"
        }
        supplied_semantic = {
            key: value for key, value in supplied.items() if key != "snapshot_sha256"
        }
        if prepared_semantic != supplied_semantic:
            raise DojoPortfolioReplayError(
                "supplied post-exit snapshot differs from reducer-owned state"
            )
        # Protocol normalization permits JSON integer/float representations that
        # are numerically equal but hash differently.  The compatibility wrapper
        # retains the already-verified caller representation so existing sealed
        # proposal batches remain bound, after reducer economics matched exactly.
        assert session._prepared is not None
        session._prepared["snapshot"] = supplied
        session.consume_proposal_batch(frame["proposal_batch"])
    return session.finalize(terminal_policy=terminal_policy)


def validate_portfolio_replay_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate hashes and terminal accounting invariants of a replay result."""

    row = _mapping(result, "result")
    required = frozenset(
        {
            "contract",
            "schema_version",
            "status",
            "economic_failure_codes",
            "policy_sha256",
            "terminal_policy",
            "processed_coordinate_count",
            "start_epoch",
            "end_epoch",
            "duration_seconds",
            "start_balance_jpy",
            "end_balance_jpy",
            "start_equity_jpy",
            "end_equity_jpy",
            "minimum_mtm_equity_jpy",
            "minimum_free_margin_jpy",
            "period_multiple_mtm",
            "monthly_multiple_mtm",
            "max_drawdown_fraction",
            "peak_margin_jpy",
            "peak_margin_usage_fraction",
            "margin_closeouts",
            "margin_reject_count",
            "ruin_event_count",
            "realized_pnl_jpy",
            "financing_cost_jpy",
            "spread_cost_jpy",
            "slippage_cost_jpy",
            "transaction_cost_jpy",
            "attempts",
            "fills",
            "execution_fill_count",
            "trade_count",
            "rejections",
            "rejection_counts",
            "orders_placed",
            "orders_triggered",
            "orders_expired",
            "orders_cancelled",
            "position_closes",
            "pair_pnl_jpy",
            "family_pnl_jpy",
            "capital_lock_margin_jpy_hours",
            "open_position_count",
            "pending_order_count",
            "event_count",
            "event_chain_sha256",
            "carry_state_sha256",
            "carry_state",
            "raw_quote_events_included",
            "live_permission",
            "broker_mutation_allowed",
            "three_x_guaranteed",
            "result_sha256",
        }
    )
    _exact(row, required, "result")
    unsigned = {key: row[key] for key in required if key != "result_sha256"}
    if (
        row["contract"] != PORTFOLIO_REPLAY_CONTRACT
        or row["schema_version"] != SCHEMA_VERSION
    ):
        raise DojoPortfolioReplayError("unsupported replay result contract")
    if canonical_portfolio_sha256(unsigned) != _sha(
        row["result_sha256"], "result.result_sha256"
    ):
        raise DojoPortfolioReplayError("result_sha256 mismatch")
    if row["terminal_policy"] not in {
        MONTH_END_FLAT_SETTLEMENT,
        MONTH_END_MTM_WITH_STATE_HANDOFF,
    }:
        raise DojoPortfolioReplayError("result terminal_policy is unsupported")
    carry = _mapping(row["carry_state"], "result.carry_state")
    if carry.get("carry_state_sha256") != row["carry_state_sha256"]:
        raise DojoPortfolioReplayError("result carry_state_sha256 mismatch")
    carry_unsigned = {
        key: value for key, value in carry.items() if key != "carry_state_sha256"
    }
    if canonical_portfolio_sha256(carry_unsigned) != row["carry_state_sha256"]:
        raise DojoPortfolioReplayError("embedded carry state hash mismatch")
    _assert_close(
        row["end_balance_jpy"],
        row["start_balance_jpy"] + row["realized_pnl_jpy"] - row["financing_cost_jpy"],
        "result balance accounting",
    )
    _assert_close(
        row["period_multiple_mtm"],
        row["end_equity_jpy"] / row["start_equity_jpy"],
        "result period_multiple_mtm",
    )
    _assert_close(
        row["transaction_cost_jpy"],
        row["spread_cost_jpy"] + row["slippage_cost_jpy"],
        "result transaction cost accounting",
    )
    if row["execution_fill_count"] != row["fills"] + row["position_closes"]:
        raise DojoPortfolioReplayError("result execution fill count is inconsistent")
    if row["trade_count"] != row["position_closes"]:
        raise DojoPortfolioReplayError("result trade count is inconsistent")
    if row["ruin_event_count"] > 0 and row["minimum_mtm_equity_jpy"] > 0:
        raise DojoPortfolioReplayError("result ruin metric is inconsistent")
    if (
        row["start_equity_jpy"] > 0
        and row["minimum_mtm_equity_jpy"] <= 0
        and row["ruin_event_count"] == 0
    ):
        raise DojoPortfolioReplayError("result ruin metric is inconsistent")
    if row["terminal_policy"] == MONTH_END_FLAT_SETTLEMENT and (
        row["open_position_count"] != 0 or row["pending_order_count"] != 0
    ):
        raise DojoPortfolioReplayError("flat settlement retained risk")
    if (
        row["raw_quote_events_included"]
        or row["live_permission"]
        or row["broker_mutation_allowed"]
        or row["three_x_guaranteed"]
    ):
        raise DojoPortfolioReplayError("result grants authority or claims a guarantee")
    return _copy(row)


__all__ = [
    "DojoPortfolioReplayError",
    "MONTH_END_FLAT_SETTLEMENT",
    "MONTH_END_MTM_WITH_STATE_HANDOFF",
    "PORTFOLIO_CARRY_CONTRACT",
    "PORTFOLIO_CHECKPOINT_CONTRACT",
    "PORTFOLIO_COORDINATE_RECEIPT_CONTRACT",
    "PORTFOLIO_POLICY_CONTRACT",
    "PORTFOLIO_REPLAY_CONTRACT",
    "PortfolioReplaySession",
    "admit_worker_proposals",
    "canonical_portfolio_sha256",
    "quote_batch_sha256",
    "reduce_portfolio_replay",
    "seal_portfolio_policy",
    "validate_portfolio_replay_result",
    "verify_portfolio_replay_checkpoint",
    "verify_portfolio_policy",
]
