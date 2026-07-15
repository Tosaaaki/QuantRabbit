"""Capture-economics audit: is the exit machinery paying for the entries?

Reads trader-attributed realized outcomes from `data/execution_ledger.db`
(same gateway-entry attribution CTE as lane-history scoring, so manual /
tagless closes are excluded) and publishes the payoff arithmetic the daily
5% / 10% campaign actually depends on:

- win rate `p`, average win `W`, average loss `L`, payoff ratio `W/L`
- the breakeven payoff requirement `(1 - p) / p` at the observed win rate
- expectancy per trade in JPY and in % of the campaign per-trade budget
- the same metrics split by exit reason and by ISO week
- pair/side/method repair priorities that separate scoped broker-TP proof from
  MARKET_ORDER_TRADE_CLOSE leakage, so high rotation preserves the paying
  capture shape instead of scaling the lossy close path

This is first an audit surface: it does not select sides or grant permission.
When it reports NEGATIVE_EXPECTANCY with average losses larger than average
wins, intent generation consumes the observed average winner as a temporary
fresh-entry loss cap. That loss-asymmetry guard is the bounded-risk repair for
"one loss erases multiple wins"; it still cannot override forecast, spread,
strategy, margin, or gateway gates. It exists because the 2026-05-14→06-08 ledger showed 55 wins
averaging +376 JPY against 24 losses averaging -1,437 JPY (payoff 0.26 vs
breakeven 0.43 at the observed 70% win rate) — an asymmetry no forecast
hit-rate can outrun. The trader and the operator must see this number move
toward/over breakeven, or the +5% pace/protection marker (§5) has no arithmetic
route on days where a valid edge exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.forecast_precision import hit_rate_wilson_lower
from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.paths import ROOT

DEFAULT_CAPTURE_ECONOMICS = ROOT / "data" / "capture_economics.json"
DEFAULT_CAPTURE_ECONOMICS_REPORT = ROOT / "docs" / "capture_economics_report.md"

# Realized outcomes below this many attributed closes cannot produce a stable
# proportion estimate; the audit still reports them but flags LOW_SAMPLE.
# 20 keeps the binomial standard error under ~11pp at p=0.5 — a documented
# statistical floor, not a tuned market threshold.
MIN_SAMPLE_FOR_VERDICT = 20

# Lifetime capture economics is the safety truth, but it moves slowly after a
# repair.  Keep a fixed, adjacent seven-day comparison so the trader can see
# whether new outcomes improved without laundering a three-trade winning
# streak into positive-expectancy proof.
RECENT_PERFORMANCE_WINDOW_DAYS = 7

# Report/action payloads are read by the trader prompt packet. Keep them short
# so the live cycle sees the repair priorities without drowning out the
# current broker/intent evidence; this is an engineering display cap, not a
# market threshold.
EXIT_REPAIR_ITEM_LIMIT = 4

# Pair/side/method repair rows are compact operator routing evidence, not a
# live-entry permission list. The cap keeps the prompt packet focused while the
# nested metrics above still retain the full realized bucket details.
SEGMENT_REPAIR_PRIORITY_LIMIT = 12
MARKET_CLOSE_LOSS_EXAMPLE_LIMIT = 5

TAKE_PROFIT_EXIT_REASON = "TAKE_PROFIT_ORDER"
MARKET_CLOSE_EXIT_REASON = "MARKET_ORDER_TRADE_CLOSE"
SYSTEM_ATTRIBUTION_SCOPE = "SYSTEM_GATEWAY_ATTRIBUTED_ONLY"

# OANDA cash fields are persisted to four decimal JPY precision. Match the
# target ledger's one-cent reconciliation allowance: smaller gaps are numeric
# aggregation noise; larger gaps make trade-level financing attribution unsafe.
FINANCING_COMPONENT_RECONCILIATION_TOLERANCE_JPY = 0.01

_MANUAL_OWNER_MARKERS = frozenset({"manual", "operator_manual", "unknown", "external"})
_UNSUPPORTED_CASH_FIELDS = frozenset(
    {
        "commission",
        "guaranteedExecutionFee",
        "guaranteedExecutionFeeHomeConversionCost",
        "quoteGuaranteedExecutionFee",
        "dividendAdjustment",
        "quoteDividendAdjustment",
    }
)
_NON_PNL_OANDA_TRANSACTION_TYPES = frozenset(
    {
        "TRANSFER_FUNDS",
        "TRADE_CLIENT_EXTENSIONS_MODIFY",
        "ORDER_CLIENT_EXTENSIONS_MODIFY",
    }
)
_OANDA_TRANSACTION_COVERAGE_START_KEY = "oanda_transaction_coverage_start_utc"

# Use the same statistical floor as the audit verdict for "scoped TP proof";
# RiskEngine and IntentGenerator mirror the live relaxation floor independently
# so manual/replayed receipts remain defended without importing this module.
SCOPED_TP_PROOF_MIN_EXIT_TRADES = MIN_SAMPLE_FOR_VERDICT

# Capital allocation may use the same exact-vehicle all-exit surface from the
# intent board and again immediately before a broker POST. Keep the arithmetic
# and statistical contract in this ledger-owning module so those consumers
# cannot drift into different definitions of "positive edge".
EXACT_VEHICLE_NET_EDGE_MIN_TRADES = MIN_SAMPLE_FOR_VERDICT
EXACT_VEHICLE_NET_ARITHMETIC_ABS_TOLERANCE_JPY = 0.05
EXACT_VEHICLE_NET_ARITHMETIC_REL_TOLERANCE = 1e-6
EXACT_VEHICLE_ALLOCATION_SURFACE_CONTRACT = (
    "QR_EXACT_VEHICLE_ALLOCATION_SURFACE_V2"
)
EXECUTION_COST_FLOOR_CONTRACT = "QR_NET_EXECUTION_COST_FLOOR_V1"
EXECUTION_COST_MIN_SAMPLES = MIN_SAMPLE_FOR_VERDICT
# Execution quality is a slower-moving broker/transport measurement than a
# quote.  Ninety days keeps a mature cohort usable through a temporary
# no-entry repair period while still refusing an indefinitely historical cost
# assumption.  The cost values themselves are always ledger-derived.
EXECUTION_COST_MAX_SAMPLE_AGE_SECONDS = 90 * 24 * 60 * 60


@dataclass(frozen=True)
class RealizedOutcome:
    ts_utc: str
    trade_id: str
    pair: str
    side: str
    lane_id: str
    method: str
    exit_reason: str
    realized_pl_jpy: float
    entry_vehicle: str = "UNKNOWN"
    entry_truth_consistent: bool = False
    exit_reasons: tuple[str, ...] = ()
    pure_take_profit_lifecycle: bool = False
    broker_close_ts_utc: str = ""
    broker_time_consistent: bool = False
    entry_units: float = 0.0
    audited_financing_jpy: float = 0.0
    adverse_financing_jpy: float = 0.0


@dataclass(frozen=True)
class UnresolvedRealizedOutcome:
    """Audited cash already realized on a system trade without a terminal close.

    This row deliberately does not count as a completed trade. It does make an
    exact-vehicle edge proof ineligible until the lifecycle terminates, so a
    partial reduction or open-trade financing adjustment cannot disappear
    behind an earlier completed winning sample.
    """

    ts_utc: str
    trade_id: str
    pair: str
    side: str
    lane_id: str
    method: str
    realized_pl_jpy: float
    entry_vehicle: str = "UNKNOWN"
    exit_reasons: tuple[str, ...] = ()
    reduction_count: int = 0


@dataclass(frozen=True)
class AttributedEntry:
    trade_id: str
    order_id: str
    pair: str
    side: str
    lane_id: str
    canonical_lane_id: str
    method: str
    entry_vehicle: str
    entry_ts_utc: str
    entry_units: float
    ledger_rowid: int
    broker_entry_ts_utc: str = ""
    broker_time_consistent: bool = False


def read_attributed_system_entries(ledger_path: Path) -> list[AttributedEntry] | None:
    """Return every system entry using the same gateway/fill attribution contract.

    Unlike ``read_attributed_net_outcomes``, this includes unresolved trades so
    forward cohorts cannot skip a slow open loss and substitute later closes.
    """

    if not ledger_path.exists():
        return None
    try:
        with closing(sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True)) as conn, conn:
            conn.row_factory = sqlite3.Row
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(execution_events)").fetchall()
            }
            required = {
                "ts_utc",
                "event_type",
                "lane_id",
                "order_id",
                "trade_id",
                "pair",
                "side",
                "units",
                "exit_reason",
                "raw_json",
            }
            if not required.issubset(columns):
                return None
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT
                        rowid AS ledger_rowid,
                        ts_utc,
                        event_type,
                        lane_id,
                        order_id,
                        trade_id,
                        pair,
                        side,
                        units,
                        exit_reason,
                        raw_json
                    FROM execution_events
                    WHERE event_type IN (
                        'GATEWAY_ORDER_SENT',
                        'ORDER_ACCEPTED',
                        'ORDER_FILLED'
                    )
                    ORDER BY rowid ASC
                    """
                ).fetchall()
            ]
    except (sqlite3.Error, TypeError, ValueError):
        return None

    for row in rows:
        raw_value = row.get("raw_json")
        raw = _optional_json_object(raw_value)
        if isinstance(raw_value, str) and raw_value.strip() and raw is None:
            return None
        if raw is not None and not _unsupported_cash_is_zero(raw):
            return None

    gateways_by_order: dict[str, list[dict[str, Any]]] = {}
    fills_by_trade: dict[str, list[dict[str, Any]]] = {}
    fill_rows: list[dict[str, Any]] = []
    for row in rows:
        event_type = str(row.get("event_type") or "")
        order_id = str(row.get("order_id") or "").strip()
        trade_id = str(row.get("trade_id") or "").strip()
        if event_type in {"GATEWAY_ORDER_SENT", "ORDER_ACCEPTED"} and order_id:
            gateways_by_order.setdefault(order_id, []).append(row)
        elif event_type == "ORDER_FILLED":
            fill_rows.append(row)
            if trade_id:
                fills_by_trade.setdefault(trade_id, []).append(row)

    if any(
        not str(fill.get("trade_id") or "").strip()
        and _fill_has_system_attribution_candidate(
            fill,
            gateways_by_order=gateways_by_order,
        )
        for fill in fill_rows
    ):
        # A fill following a QuantRabbit gateway receipt cannot disappear merely
        # because one normalized identity column was lost.  It may be one of the
        # first forward entries, so skipping it would create survivorship bias.
        return None

    entries: list[AttributedEntry] = []
    for trade_id, fills in fills_by_trade.items():
        entry = _resolve_system_entry(
            fills=fills,
            gateways_by_order=gateways_by_order,
        )
        if entry is None:
            if any(
                _fill_has_system_attribution_candidate(
                    fill,
                    gateways_by_order=gateways_by_order,
                )
                for fill in fills
            ):
                return None
            continue
        if not entry["truth_consistent"] or len(fills) != 1:
            return None
        fill = fills[0]
        order_id = str(fill.get("order_id") or "").strip()
        if (
            not order_id
            or not entry.get("broker_entry_ts_utc")
            or entry.get("broker_time_consistent") is not True
        ):
            return None
        lane_id = str(entry["lane_id"] or "").strip()
        vehicle = str(entry["vehicle"] or "UNKNOWN").upper()
        canonical_lane_id = (
            lane_id
            if len(lane_id.split(":")) >= 5
            else f"{lane_id}:{vehicle}"
        )
        entries.append(
            AttributedEntry(
                trade_id=trade_id,
                order_id=order_id,
                pair=str(entry["pair"] or "").upper(),
                side=str(entry["side"] or "").upper(),
                lane_id=lane_id,
                canonical_lane_id=canonical_lane_id,
                method=str(entry["method"] or "").upper(),
                entry_vehicle=vehicle,
                entry_ts_utc=str(entry["entry_ts_utc"] or ""),
                entry_units=float(entry["entry_units"]),
                ledger_rowid=int(fill.get("ledger_rowid") or 0),
                broker_entry_ts_utc=str(entry["broker_entry_ts_utc"]),
                broker_time_consistent=True,
            )
        )
    entries.sort(
        key=lambda item: (
            _rfc3339_utc_key(item.broker_entry_ts_utc)
            or (datetime.min.replace(tzinfo=timezone.utc).isoformat(), 0),
            item.ledger_rowid,
        )
    )
    return entries


def read_attributed_net_outcomes(
    ledger_path: Path,
    *,
    unresolved_realized_outcomes: list[UnresolvedRealizedOutcome] | None = None,
) -> list[RealizedOutcome] | None:
    """Return audited, resolved system-attributed outcomes after financing.

    ``None`` means the ledger/query could not be read safely; an empty list is
    a valid readable ledger with no resolved attributed outcomes. Entry
    attribution is deliberately narrower than historical lane scoring: an
    ``ORDER_FILLED`` row must carry its own lane or inherit one from a
    *preceding* gateway row with the exact same order id. Trade-id-only and
    post-fill gateway rows are never entry evidence.

    Before returning any outcome, every close/reduction row is reconciled to
    its exact raw OANDA ``ORDER_FILL`` trade component and every
    ``DAILY_FINANCING`` transaction is reconciled account-total-to-components.
    One unverifiable cash row makes the complete outcome surface unreadable;
    it is never silently converted into an empty/zero sample.
    """

    if not ledger_path.exists():
        return None
    try:
        with closing(sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True)) as conn, conn:
            conn.row_factory = sqlite3.Row
            table_names = {
                str(row["name"])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(execution_events)").fetchall()
            }
            required = {
                "ts_utc",
                "event_type",
                "lane_id",
                "order_id",
                "trade_id",
                "pair",
                "side",
                "units",
                "realized_pl_jpy",
                "financing_jpy",
                "exit_reason",
                "raw_json",
            }
            if not required.issubset(columns):
                return None
            oanda_transaction_id_expr = (
                "oanda_transaction_id"
                if "oanda_transaction_id" in columns
                else "NULL AS oanda_transaction_id"
            )
            coverage_start_raw: object = None
            if "sync_state" in table_names:
                coverage_row = conn.execute(
                    "SELECT value FROM sync_state WHERE key=?",
                    (_OANDA_TRANSACTION_COVERAGE_START_KEY,),
                ).fetchone()
                if coverage_row is not None:
                    coverage_start_raw = coverage_row["value"]
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT
                        rowid AS ledger_rowid,
                        ts_utc,
                        event_type,
                        lane_id,
                        order_id,
                        trade_id,
                        pair,
                        side,
                        units,
                        realized_pl_jpy,
                        financing_jpy,
                        exit_reason,
                        {oanda_transaction_id_expr},
                        raw_json
                    FROM execution_events
                    WHERE event_type IN (
                        'GATEWAY_ORDER_SENT',
                        'ORDER_ACCEPTED',
                        'ORDER_FILLED',
                        'TRADE_REDUCED',
                        'TRADE_CLOSED',
                        'OANDA_TRANSACTION'
                    )
                    ORDER BY rowid ASC
                    """.format(
                        oanda_transaction_id_expr=oanda_transaction_id_expr,
                    )
                ).fetchall()
            ]
            oanda_order_fill_transaction_rows: list[dict[str, Any]] = []
            oanda_daily_financing_transaction_rows: list[dict[str, Any]] = []
            oanda_transactions_authoritative = False
            if "oanda_transactions" in table_names:
                transaction_columns = {
                    str(row["name"])
                    for row in conn.execute(
                        "PRAGMA table_info(oanda_transactions)"
                    ).fetchall()
                }
                if not {"transaction_id", "type", "raw_json"}.issubset(
                    transaction_columns
                ):
                    return None
                oanda_transactions_authoritative = True
                oanda_order_fill_transaction_rows = [
                    dict(row)
                    for row in conn.execute(
                        """
                        SELECT transaction_id, type, raw_json
                        FROM oanda_transactions
                        WHERE type = 'ORDER_FILL'
                        ORDER BY CAST(transaction_id AS INTEGER), transaction_id
                        """
                    ).fetchall()
                ]
                oanda_daily_financing_transaction_rows = [
                    dict(row)
                    for row in conn.execute(
                        """
                        SELECT transaction_id, type, raw_json
                        FROM oanda_transactions
                        WHERE type = 'DAILY_FINANCING'
                        ORDER BY CAST(transaction_id AS INTEGER), transaction_id
                        """
                    ).fetchall()
                ]
    except (sqlite3.Error, TypeError, ValueError):
        return None

    coverage_start = _parse_utc_instant(coverage_start_raw)

    nonzero_financing_events_by_trade: dict[str, list[tuple[str, float]]] = {}
    financing_by_trade = _audited_financing_by_trade(
        rows,
        nonzero_events_by_trade=nonzero_financing_events_by_trade,
    )
    if financing_by_trade is None:
        return None
    if not _audit_daily_financing_transaction_completeness(
        rows,
        oanda_transaction_rows=oanda_daily_financing_transaction_rows,
        oanda_transactions_authoritative=oanda_transactions_authoritative,
    ):
        return None

    if not _audit_close_transaction_completeness(
        rows,
        oanda_transaction_rows=oanda_order_fill_transaction_rows,
        oanda_transactions_authoritative=oanda_transactions_authoritative,
    ):
        return None

    close_rows_by_trade: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        event_type = str(row.get("event_type") or "")
        if event_type not in {"TRADE_CLOSED", "TRADE_REDUCED"}:
            continue
        audited = _audit_close_row(row)
        if audited is None:
            return None
        realized, financing, reason, broker_close_ts_utc = audited
        row["audited_realized_pl_jpy"] = realized
        row["audited_financing_jpy"] = financing
        row["audited_exit_reason"] = reason
        row["audited_broker_close_ts_utc"] = broker_close_ts_utc
        trade_id = str(row.get("trade_id") or "").strip()
        close_rows_by_trade.setdefault(trade_id, []).append(row)

    # Unsupported cash on entry/gateway rows is not part of trade net and
    # cannot be allocated safely. Raw JSON may be NULL/blank on legacy rows,
    # but non-empty malformed JSON must never be treated as if it were absent.
    for row in rows:
        if str(row.get("event_type") or "") not in {
            "GATEWAY_ORDER_SENT",
            "ORDER_ACCEPTED",
            "ORDER_FILLED",
        }:
            continue
        raw_value = row.get("raw_json")
        raw = _optional_json_object(raw_value)
        if isinstance(raw_value, str) and raw_value.strip() and raw is None:
            return None
        if raw is not None and not _unsupported_cash_is_zero(raw):
            return None

    gateways_by_order: dict[str, list[dict[str, Any]]] = {}
    fills_by_trade: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        event_type = str(row.get("event_type") or "")
        order_id = str(row.get("order_id") or "").strip()
        trade_id = str(row.get("trade_id") or "").strip()
        if event_type in {"GATEWAY_ORDER_SENT", "ORDER_ACCEPTED"} and order_id:
            gateways_by_order.setdefault(order_id, []).append(row)
        elif event_type == "ORDER_FILLED":
            # Index by both normalized and raw broker trade identity. A broken
            # normalized id must not let a gateway-attributed financing/close
            # lifecycle disappear; _resolve_system_entry will then compare the
            # two identities and fail closed on any mismatch.
            trade_ids = {trade_id} if trade_id else set()
            raw = _optional_json_object(row.get("raw_json"))
            opened = raw.get("tradeOpened") if isinstance(raw, dict) else None
            if isinstance(opened, dict):
                raw_trade_id = _identifier_text(opened.get("tradeID"))
                if raw_trade_id:
                    trade_ids.add(raw_trade_id)
            for indexed_trade_id in trade_ids:
                fills_by_trade.setdefault(indexed_trade_id, []).append(row)

    outcomes: list[RealizedOutcome] = []
    unresolved: list[UnresolvedRealizedOutcome] = []
    for trade_id, lifecycle in close_rows_by_trade.items():
        entry = _resolve_system_entry(
            fills=fills_by_trade.get(trade_id, []),
            gateways_by_order=gateways_by_order,
        )
        if entry is None:
            if any(
                _fill_has_system_attribution_candidate(
                    fill,
                    gateways_by_order=gateways_by_order,
                )
                for fill in fills_by_trade.get(trade_id, [])
            ):
                return None
            continue
        if not entry["truth_consistent"]:
            # A system-looking lane whose broker entry truth contradicts its
            # pair/side/vehicle is corrupt attribution, not a zero sample.
            # Global CAP_AVG_WIN and reports consume this same reader, so they
            # must fail closed rather than merely hiding the row from exact TP.
            return None
        entry_ts = _parse_utc_instant(entry["entry_ts_utc"])
        if coverage_start is None or entry_ts is None or entry_ts < coverage_start:
            # The cold-baseline contract guarantees financing completeness only
            # from this marker forward. A pre-coverage entry may have carry
            # debits/credits absent from the local ledger, so neither global
            # avg-win nor exact TP proof may treat it as a zero-financing trade.
            return None
        lifecycle.sort(key=lambda row: int(row.get("ledger_rowid") or 0))
        terminal_rows = [
            row for row in lifecycle if str(row.get("event_type") or "") == "TRADE_CLOSED"
        ]
        exit_reasons = tuple(str(row["audited_exit_reason"]) for row in lifecycle)
        audited_financing_jpy = sum(
            float(row["audited_financing_jpy"])
            for row in lifecycle
        ) + financing_by_trade.get(trade_id, 0.0)
        adverse_financing_jpy = sum(
            max(0.0, -float(row["audited_financing_jpy"]))
            for row in lifecycle
        ) + sum(
            max(0.0, -float(amount))
            for _, amount in nonzero_financing_events_by_trade.get(
                trade_id, []
            )
        )
        net_jpy = sum(
            float(row["audited_realized_pl_jpy"])
            for row in lifecycle
        ) + audited_financing_jpy
        if not terminal_rows:
            # Cash from a partial reduction is already real even though the
            # trade is still open. Preserve it as unresolved evidence: it must
            # not increase the completed sample, but it must prevent a stale
            # positive sample from authorizing more risk.
            unresolved.append(
                UnresolvedRealizedOutcome(
                    ts_utc=str(lifecycle[-1].get("ts_utc") or ""),
                    trade_id=trade_id,
                    pair=entry["pair"],
                    side=entry["side"],
                    lane_id=entry["lane_id"],
                    method=entry["method"],
                    realized_pl_jpy=float(net_jpy),
                    entry_vehicle=entry["vehicle"],
                    exit_reasons=exit_reasons,
                    reduction_count=len(lifecycle),
                )
            )
            continue
        final_close = terminal_rows[-1]
        outcomes.append(
            RealizedOutcome(
                ts_utc=str(final_close.get("ts_utc") or ""),
                trade_id=trade_id,
                pair=entry["pair"],
                side=entry["side"],
                lane_id=entry["lane_id"],
                method=entry["method"],
                exit_reason=str(final_close["audited_exit_reason"]),
                realized_pl_jpy=float(net_jpy),
                entry_vehicle=entry["vehicle"],
                entry_truth_consistent=bool(entry["truth_consistent"]),
                exit_reasons=exit_reasons,
                pure_take_profit_lifecycle=bool(exit_reasons)
                and all(reason == TAKE_PROFIT_EXIT_REASON for reason in exit_reasons),
                broker_close_ts_utc=str(
                    final_close.get("audited_broker_close_ts_utc") or ""
                ),
                broker_time_consistent=True,
                entry_units=abs(float(entry["entry_units"])),
                audited_financing_jpy=float(audited_financing_jpy),
                adverse_financing_jpy=float(adverse_financing_jpy),
            )
        )
    for trade_id, financing_events in nonzero_financing_events_by_trade.items():
        if trade_id in close_rows_by_trade:
            # Terminal and partial-close lifecycles above already include the
            # complete audited financing total exactly once.
            continue
        entry = _resolve_system_entry(
            fills=fills_by_trade.get(trade_id, []),
            gateways_by_order=gateways_by_order,
        )
        if entry is None:
            if any(
                _fill_has_system_attribution_candidate(
                    fill,
                    gateways_by_order=gateways_by_order,
                )
                for fill in fills_by_trade.get(trade_id, [])
            ):
                return None
            # Manual/unattributed open-trade financing is account truth but is
            # outside the autonomous trader's exact-vehicle edge surface.
            continue
        if not entry["truth_consistent"]:
            return None
        entry_ts = _parse_utc_instant(entry["entry_ts_utc"])
        if coverage_start is None or entry_ts is None or entry_ts < coverage_start:
            return None
        latest_ts = financing_events[-1][0]
        unresolved.append(
            UnresolvedRealizedOutcome(
                ts_utc=latest_ts,
                trade_id=trade_id,
                pair=entry["pair"],
                side=entry["side"],
                lane_id=entry["lane_id"],
                method=entry["method"],
                realized_pl_jpy=float(financing_by_trade.get(trade_id, 0.0)),
                entry_vehicle=entry["vehicle"],
                exit_reasons=("DAILY_FINANCING",),
                reduction_count=0,
            )
        )
    if unresolved_realized_outcomes is not None:
        unresolved_realized_outcomes.extend(unresolved)
    return outcomes


def read_exact_vehicle_take_profit_metrics(
    ledger_path: Path,
) -> dict[tuple[str, str, str, str], dict[str, Any]] | None:
    """Aggregate pure broker-TP outcomes by audited entry vehicle.

    The same fail-closed reader backs capture economics, gateway pre-POST
    reconciliation, intent generation, and operator-board display. This keeps
    an unaudited/manual/mixed-exit row from becoming proof in one consumer
    after another consumer correctly rejected it.
    """

    outcomes = read_attributed_net_outcomes(ledger_path)
    if outcomes is None:
        return None
    return _aggregate_exact_vehicle_outcomes(outcomes, pure_take_profit_only=True)


def read_exact_vehicle_net_metrics(
    ledger_path: Path,
) -> dict[tuple[str, str, str, str], dict[str, Any]] | None:
    """Aggregate every audited net outcome by exact entry vehicle.

    Unlike :func:`read_exact_vehicle_take_profit_metrics`, this includes market
    closes, reductions, and every other terminal exit in the attributed trade
    lifecycle.  It is the realized all-exit edge surface used by GPT capital
    allocation so profitable TP rows cannot hide a larger stop/market-close
    leak for the same pair, side, method, and entry vehicle.
    """

    unresolved: list[UnresolvedRealizedOutcome] = []
    outcomes = read_attributed_net_outcomes(
        ledger_path,
        unresolved_realized_outcomes=unresolved,
    )
    if outcomes is None:
        return None
    metrics = _aggregate_exact_vehicle_outcomes(
        outcomes,
        pure_take_profit_only=False,
    )
    for row in metrics.values():
        row["unresolved_realized_trades"] = 0
        row["unresolved_realized_net_jpy"] = 0.0
    unresolved_ids: dict[tuple[str, str, str, str], list[str]] = {}
    for outcome in unresolved:
        pair = str(outcome.pair or "UNKNOWN").upper()
        side = str(outcome.side or "UNKNOWN").upper()
        method = str(outcome.method or "UNKNOWN").upper()
        vehicle = str(outcome.entry_vehicle or "UNKNOWN").upper()
        if (
            pair == "UNKNOWN"
            or side not in {"LONG", "SHORT"}
            or method == "UNKNOWN"
            or vehicle not in {"LIMIT", "MARKET", "STOP"}
        ):
            continue
        key = (pair, side, method, vehicle)
        row = metrics.setdefault(key, _empty_exact_vehicle_net_metrics())
        row["unresolved_realized_trades"] = int(
            row.get("unresolved_realized_trades") or 0
        ) + 1
        row["unresolved_realized_net_jpy"] = round(
            float(row.get("unresolved_realized_net_jpy") or 0.0)
            + float(outcome.realized_pl_jpy),
            4,
        )
        unresolved_ids.setdefault(key, []).append(outcome.trade_id)
    for key, trade_ids in unresolved_ids.items():
        metrics[key]["unresolved_trade_ids_sha256"] = hashlib.sha256(
            json.dumps(sorted(trade_ids), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    for row in metrics.values():
        row.setdefault("unresolved_trade_ids_sha256", canonical_empty_list_sha256())
    return metrics


def canonical_empty_list_sha256() -> str:
    """Stable empty identity used by semantic execution-ledger surfaces."""

    return hashlib.sha256(b"[]").hexdigest()


def _empty_exact_vehicle_net_metrics() -> dict[str, Any]:
    return {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "expectancy_jpy_per_trade": 0.0,
        "avg_win_jpy": 0.0,
        "avg_loss_jpy": 0.0,
        "net_jpy": 0.0,
        "source_scope": "PAIR_SIDE_METHOD_VEHICLE",
        "exit_scope": "ALL_AUDITED_EXITS",
        "unresolved_realized_trades": 0,
        "unresolved_realized_net_jpy": 0.0,
        "entry_units_total": 0.0,
        "financing_observation_trades": 0,
        "financing_adverse_trades": 0,
        "financing_adverse_total_jpy": 0.0,
        "financing_adverse_mean_jpy_per_unit": 0.0,
        "financing_adverse_occurrence_wilson95_upper": None,
        "financing_adverse_stress_jpy_per_unit": None,
        "latest_broker_close_ts_utc": None,
        "oldest_broker_close_ts_utc": None,
    }


def evaluate_exact_vehicle_net_edge(
    metrics: Mapping[str, Any] | None,
    *,
    min_trades: int = EXACT_VEHICLE_NET_EDGE_MIN_TRADES,
) -> dict[str, Any]:
    """Validate one exact-vehicle all-exit row and stress its realized edge.

    A thin, arithmetically consistent positive row does not itself prove edge,
    but it may coexist with the bounded exact-TP collection exception. Any
    known loss, contradiction, mature non-robust row, or unresolved realized
    lifecycle blocks that exception.
    """

    source = metrics if isinstance(metrics, Mapping) else {}

    def strict_int(key: str) -> int | None:
        value = source.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def strict_float(key: str) -> float | None:
        value = source.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None

    trades = strict_int("trades")
    wins = strict_int("wins")
    losses = strict_int("losses")
    net_jpy = strict_float("net_jpy")
    expectancy = strict_float("expectancy_jpy_per_trade")
    avg_win = strict_float("avg_win_jpy")
    avg_loss = strict_float("avg_loss_jpy")
    unresolved_trades = strict_int("unresolved_realized_trades")
    unresolved_net_jpy = strict_float("unresolved_realized_net_jpy")
    if unresolved_trades is None and "unresolved_realized_trades" not in source:
        unresolved_trades = 0
    if unresolved_net_jpy is None and "unresolved_realized_net_jpy" not in source:
        unresolved_net_jpy = 0.0

    counts_consistent = bool(
        trades is not None
        and wins is not None
        and losses is not None
        and trades >= 0
        and wins >= 0
        and losses >= 0
        and wins + losses <= trades
        and unresolved_trades is not None
        and unresolved_trades >= 0
        and unresolved_net_jpy is not None
    )
    magnitudes_consistent = bool(
        avg_win is not None
        and avg_loss is not None
        and avg_win >= 0.0
        and avg_loss >= 0.0
        and wins is not None
        and losses is not None
        and ((wins == 0 and avg_win == 0.0) or (wins > 0 and avg_win > 0.0))
        and ((losses == 0 and avg_loss == 0.0) or (losses > 0 and avg_loss > 0.0))
    )
    implied_net = (
        wins * avg_win - losses * avg_loss
        if counts_consistent
        and magnitudes_consistent
        and wins is not None
        and losses is not None
        and avg_win is not None
        and avg_loss is not None
        else None
    )
    net_identity_consistent = bool(
        implied_net is not None
        and net_jpy is not None
        and math.isclose(
            implied_net,
            net_jpy,
            rel_tol=EXACT_VEHICLE_NET_ARITHMETIC_REL_TOLERANCE,
            abs_tol=EXACT_VEHICLE_NET_ARITHMETIC_ABS_TOLERANCE_JPY,
        )
    )
    expectancy_identity_consistent = bool(
        trades is not None
        and trades >= 0
        and expectancy is not None
        and net_jpy is not None
        and math.isclose(
            expectancy * trades,
            net_jpy,
            rel_tol=EXACT_VEHICLE_NET_ARITHMETIC_REL_TOLERANCE,
            abs_tol=EXACT_VEHICLE_NET_ARITHMETIC_ABS_TOLERANCE_JPY,
        )
    )
    arithmetic_consistent = bool(
        counts_consistent
        and magnitudes_consistent
        and net_identity_consistent
        and expectancy_identity_consistent
    )
    evidence_present = bool(
        (trades is not None and trades > 0)
        or (unresolved_trades is not None and unresolved_trades > 0)
    )
    win_rate = (
        wins / trades
        if arithmetic_consistent
        and trades is not None
        and trades > 0
        and wins is not None
        else None
    )
    wilson_lower = hit_rate_wilson_lower(win_rate, trades or 0)
    loss_proxy = (
        avg_loss
        if losses is not None and losses > 0
        else avg_win
        if losses == 0
        else None
    )
    stressed_expectancy = (
        wilson_lower * avg_win - (1.0 - wilson_lower) * loss_proxy
        if wilson_lower is not None
        and avg_win is not None
        and avg_win > 0.0
        and loss_proxy is not None
        and loss_proxy > 0.0
        else None
    )
    no_unresolved_cash = bool(unresolved_trades == 0)
    positive_consistent = bool(
        arithmetic_consistent
        and trades is not None
        and trades > 0
        and net_jpy is not None
        and net_jpy > 0.0
        and expectancy is not None
        and expectancy > 0.0
        and no_unresolved_cash
    )
    proven = bool(
        positive_consistent
        and trades is not None
        and trades >= min_trades
        and stressed_expectancy is not None
        and stressed_expectancy > 0.0
    )
    thin_positive_consistent = bool(
        positive_consistent
        and trades is not None
        and trades < min_trades
        and losses == 0
    )
    blocks_tp_exception = bool(
        evidence_present
        and not thin_positive_consistent
        and not proven
    )
    return {
        "proven": proven,
        "evidence_present": evidence_present,
        "arithmetic_consistent": arithmetic_consistent,
        "counts_consistent": counts_consistent,
        "magnitudes_consistent": magnitudes_consistent,
        "net_identity_consistent": net_identity_consistent,
        "expectancy_identity_consistent": expectancy_identity_consistent,
        "positive_consistent": positive_consistent,
        "thin_positive_consistent": thin_positive_consistent,
        "blocks_tp_exception": blocks_tp_exception,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "net_jpy": net_jpy,
        "expectancy_jpy": expectancy,
        "avg_win_jpy": avg_win,
        "avg_loss_jpy": avg_loss,
        "unresolved_realized_trades": unresolved_trades,
        "unresolved_realized_net_jpy": unresolved_net_jpy,
        "implied_net_jpy": round(implied_net, 4) if implied_net is not None else None,
        "win_rate_wilson95_lower": (
            round(wilson_lower, 6) if wilson_lower is not None else None
        ),
        "wilson_stressed_expectancy_jpy": (
            round(stressed_expectancy, 4)
            if stressed_expectancy is not None
            else None
        ),
        "minimum_trades": min_trades,
    }


def _nearest_rank_percentile(values: list[float], percentile: float) -> float | None:
    if not values or not 0.0 < percentile <= 1.0:
        return None
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _execution_cost_financing_metrics(
    outcomes: list[RealizedOutcome],
) -> dict[str, Any]:
    timestamped = [
        (
            outcome,
            _parse_utc_instant(
                outcome.broker_close_ts_utc or outcome.ts_utc
            ),
        )
        for outcome in outcomes
    ]
    if any(timestamp is None for _, timestamp in timestamped):
        raise ValueError("financing observation has no broker close time")
    latest_instant = max(
        (timestamp for _, timestamp in timestamped if timestamp is not None),
        default=None,
    )
    if latest_instant is not None:
        window_start = latest_instant - timedelta(
            seconds=EXECUTION_COST_MAX_SAMPLE_AGE_SECONDS
        )
        outcomes = [
            outcome
            for outcome, timestamp in timestamped
            if timestamp is not None and timestamp >= window_start
        ]
    adverse_per_unit: list[float] = []
    latest: tuple[str, int] | None = None
    latest_text: str | None = None
    oldest: tuple[str, int] | None = None
    oldest_text: str | None = None
    entry_units_total = 0.0
    adverse_total_jpy = 0.0
    for outcome in outcomes:
        units = abs(float(outcome.entry_units))
        financing = float(outcome.audited_financing_jpy)
        adverse_financing = float(outcome.adverse_financing_jpy)
        if (
            not math.isfinite(units)
            or units <= 0.0
            or not math.isfinite(financing)
            or not math.isfinite(adverse_financing)
            or adverse_financing < 0.0
        ):
            raise ValueError("financing observation has invalid entry units or cash")
        entry_units_total += units
        if adverse_financing > 0.0:
            adverse = adverse_financing
            adverse_total_jpy += adverse
            adverse_per_unit.append(adverse / units)
        timestamp_text = str(
            outcome.broker_close_ts_utc or outcome.ts_utc or ""
        ).strip()
        timestamp_key = _rfc3339_utc_key(timestamp_text)
        if timestamp_key is None:
            raise ValueError("financing observation has no exact broker close time")
        if latest is None or timestamp_key > latest:
            latest = timestamp_key
            latest_text = timestamp_text
        if oldest is None or timestamp_key < oldest:
            oldest = timestamp_key
            oldest_text = timestamp_text
    observations = len(outcomes)
    adverse_count = len(adverse_per_unit)
    occurrence_upper = _wilson95_upper(
        adverse_count / observations if observations else 0.0,
        observations,
    )
    mean_adverse_per_unit = (
        sum(adverse_per_unit) / adverse_count if adverse_count else 0.0
    )
    stress_per_unit = (
        occurrence_upper * mean_adverse_per_unit
        if occurrence_upper is not None
        else None
    )
    return {
        "observation_trades": observations,
        "adverse_trades": adverse_count,
        "entry_units_total": round(entry_units_total, 4),
        "adverse_total_jpy": round(adverse_total_jpy, 8),
        "adverse_mean_jpy_per_unit": round(mean_adverse_per_unit, 12),
        "adverse_occurrence_wilson95_upper": (
            round(occurrence_upper, 12)
            if occurrence_upper is not None
            else None
        ),
        "adverse_stress_jpy_per_unit": (
            round(stress_per_unit, 12)
            if stress_per_unit is not None
            else None
        ),
        "latest_observation_utc": latest_text,
        "oldest_observation_utc": oldest_text,
    }


def _read_audited_execution_slippage(
    ledger_path: Path,
) -> dict[str, Any] | None:
    """Audit entry and broker-protection fill slippage from exact identities.

    MARKET entry reference is the gateway's executable RiskMetrics entry.
    Broker truth is ``tradeOpened.price``; deprecated/top-level price,
    ``fullVWAP``, normalized price, immediate response, order id, trade id,
    units, pair, and side must all agree. Protected exits bind the close fill's
    exact order id and trade id to the authoritative OANDA
    ``TAKE_PROFIT_ORDER``/``STOP_LOSS_ORDER`` transaction; a normalized
    ``PROTECTION_CREATED`` row is only a consistency copy and may not replace
    the authoritative join. Rows outside the system-attributed entry cohort
    never enter the sample.
    """

    entries = read_attributed_system_entries(ledger_path)
    if entries is None:
        return None
    try:
        with closing(
            sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True)
        ) as conn, conn:
            conn.row_factory = sqlite3.Row
            columns = {
                str(row["name"])
                for row in conn.execute(
                    "PRAGMA table_info(execution_events)"
                ).fetchall()
            }
            required = {
                "ts_utc",
                "event_type",
                "lane_id",
                "order_id",
                "trade_id",
                "pair",
                "side",
                "units",
                "exit_reason",
                "raw_json",
            }
            if not required.issubset(columns):
                return None
            tables = {
                str(row["name"])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            authoritative_transactions = "oanda_transactions" in tables
            if authoritative_transactions:
                transaction_columns = {
                    str(row["name"])
                    for row in conn.execute(
                        "PRAGMA table_info(oanda_transactions)"
                    ).fetchall()
                }
                if not {"transaction_id", "type", "raw_json"}.issubset(
                    transaction_columns
                ):
                    return None
            price_expr = "price" if "price" in columns else "NULL AS price"
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT rowid AS ledger_rowid, ts_utc, event_type, lane_id,
                           order_id, trade_id, pair, side, units, {price_expr},
                           exit_reason, raw_json
                    FROM execution_events
                    WHERE event_type IN (
                        'GATEWAY_ORDER_SENT', 'ORDER_FILLED',
                        'PROTECTION_CREATED', 'TRADE_CLOSED'
                    )
                    ORDER BY rowid ASC
                    """.format(price_expr=price_expr)
                ).fetchall()
            ]
            protection_transactions = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT transaction_id, type, raw_json
                    FROM oanda_transactions
                    WHERE type IN ('TAKE_PROFIT_ORDER', 'STOP_LOSS_ORDER')
                    ORDER BY CAST(transaction_id AS INTEGER), transaction_id
                    """
                ).fetchall()
            ] if authoritative_transactions else []
    except (sqlite3.Error, TypeError, ValueError):
        return None

    gateways: dict[str, list[dict[str, Any]]] = {}
    fills: dict[tuple[str, str], list[dict[str, Any]]] = {}
    protections: dict[tuple[str, str], list[dict[str, Any]]] = {}
    closes: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        event_type = str(row.get("event_type") or "")
        order_id = _identifier_text(row.get("order_id"))
        trade_id = _identifier_text(row.get("trade_id"))
        if event_type == "GATEWAY_ORDER_SENT" and order_id:
            gateways.setdefault(order_id, []).append(row)
        elif event_type == "ORDER_FILLED" and order_id and trade_id:
            fills.setdefault((order_id, trade_id), []).append(row)
        elif event_type == "PROTECTION_CREATED" and order_id and trade_id:
            protections.setdefault((order_id, trade_id), []).append(row)
        elif event_type == "TRADE_CLOSED" and trade_id:
            closes.setdefault(trade_id, []).append(row)
    authoritative_protections: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = {}
    for transaction in protection_transactions:
        raw = _optional_json_object(transaction.get("raw_json"))
        if (
            raw is None
            or str(raw.get("type") or "").upper()
            not in {"TAKE_PROFIT_ORDER", "STOP_LOSS_ORDER"}
            or _identifier_text(raw.get("id"))
            != _identifier_text(transaction.get("transaction_id"))
            or str(transaction.get("type") or "").upper()
            != str(raw.get("type") or "").upper()
        ):
            return None
        order_id = _identifier_text(raw.get("id"))
        trade_id = _identifier_text(raw.get("tradeID"))
        if not order_id or not trade_id:
            return None
        authoritative_protections.setdefault(
            (order_id, trade_id), []
        ).append(raw)

    entry_rows: list[dict[str, Any]] = []
    for entry in entries:
        if entry.entry_vehicle != "MARKET":
            continue
        matching_fill = fills.get((entry.order_id, entry.trade_id), [])
        matching_gateway = gateways.get(entry.order_id, [])
        # Legacy attributed fills predate durable gateway response receipts.
        # They remain in realized economics but are outside this transport
        # calibration cohort. Once a gateway response exists, its exact join
        # is mandatory and any duplicate/malformed proof invalidates the
        # surface instead of being skipped.
        if not matching_gateway:
            continue
        if len(matching_fill) != 1 or len(matching_gateway) != 1:
            return None
        fill = matching_fill[0]
        gateway = matching_gateway[0]
        if int(gateway.get("ledger_rowid") or 0) >= int(
            fill.get("ledger_rowid") or 0
        ):
            return None
        fill_raw = _optional_json_object(fill.get("raw_json"))
        gateway_raw = _optional_json_object(gateway.get("raw_json"))
        opened = (
            fill_raw.get("tradeOpened")
            if isinstance(fill_raw, dict)
            else None
        )
        order_request = (
            gateway_raw.get("order_request")
            if isinstance(gateway_raw, dict)
            and isinstance(gateway_raw.get("order_request"), dict)
            else None
        )
        risk_metrics = (
            gateway_raw.get("risk_metrics")
            if isinstance(gateway_raw, dict)
            and isinstance(gateway_raw.get("risk_metrics"), dict)
            else None
        )
        response = (
            gateway_raw.get("response")
            if isinstance(gateway_raw, dict)
            and isinstance(gateway_raw.get("response"), dict)
            else None
        )
        create = (
            response.get("orderCreateTransaction")
            if isinstance(response, dict)
            and isinstance(response.get("orderCreateTransaction"), dict)
            else None
        )
        immediate_fill = (
            response.get("orderFillTransaction")
            if isinstance(response, dict)
            and isinstance(response.get("orderFillTransaction"), dict)
            else None
        )
        immediate_opened = (
            immediate_fill.get("tradeOpened")
            if isinstance(immediate_fill, dict)
            and isinstance(immediate_fill.get("tradeOpened"), dict)
            else None
        )
        reference = (
            _finite_float(risk_metrics.get("entry_price"))
            if isinstance(risk_metrics, dict)
            else None
        )
        broker_price = (
            _finite_float(opened.get("price"))
            if isinstance(opened, dict)
            else None
        )
        comparable_prices = (
            _finite_float(fill_raw.get("price"))
            if isinstance(fill_raw, dict)
            else None,
            _finite_float(fill_raw.get("fullVWAP"))
            if isinstance(fill_raw, dict)
            else None,
            _finite_float(immediate_fill.get("price"))
            if isinstance(immediate_fill, dict)
            else None,
            _finite_float(immediate_fill.get("fullVWAP"))
            if isinstance(immediate_fill, dict)
            else None,
            _finite_float(immediate_opened.get("price"))
            if isinstance(immediate_opened, dict)
            else None,
        )
        if authoritative_transactions:
            comparable_prices = (
                _finite_float(fill.get("price")),
                *comparable_prices,
            )
        signed_units = _finite_float(
            immediate_opened.get("units")
            if isinstance(immediate_opened, dict)
            else None
        )
        request_units = _finite_float(
            order_request.get("units")
            if isinstance(order_request, dict)
            else None
        )
        create_units = _finite_float(
            create.get("units") if isinstance(create, dict) else None
        )
        if (
            not isinstance(order_request, dict)
            or str(order_request.get("type") or "").upper() != "MARKET"
            or str(order_request.get("instrument") or "").upper()
            != entry.pair
            or str(order_request.get("timeInForce") or "").upper() != "FOK"
            or str(order_request.get("positionFill") or "").upper()
            not in {"DEFAULT", "OPEN_ONLY"}
            or request_units is None
            or abs(request_units) != abs(entry.entry_units)
            or (request_units > 0.0) != (entry.side == "LONG")
            or reference is None
            or reference <= 0.0
            or broker_price is None
            or broker_price <= 0.0
            or any(
                price is None
                or not math.isclose(
                    float(price), broker_price, rel_tol=0.0, abs_tol=1e-12
                )
                for price in comparable_prices
            )
            or _identifier_text(create.get("id") if isinstance(create, dict) else None)
            != entry.order_id
            or str(create.get("instrument") or "").upper() != entry.pair
            or create_units is None
            or abs(create_units) != abs(entry.entry_units)
            or (create_units > 0.0) != (entry.side == "LONG")
            or _identifier_text(immediate_fill.get("orderID") if isinstance(immediate_fill, dict) else None)
            != entry.order_id
            or _identifier_text(immediate_opened.get("tradeID") if isinstance(immediate_opened, dict) else None)
            != entry.trade_id
            or signed_units is None
            or abs(signed_units) != abs(entry.entry_units)
            or (signed_units > 0.0) != (entry.side == "LONG")
            or str(immediate_fill.get("instrument") or "").upper()
            != entry.pair
        ):
            return None
        pip_factor = instrument_pip_factor(entry.pair)
        signed_slippage_pips = (
            (broker_price - reference) * pip_factor
            if entry.side == "LONG"
            else (reference - broker_price) * pip_factor
        )
        entry_rows.append(
            {
                "order_id": entry.order_id,
                "trade_id": entry.trade_id,
                "pair": entry.pair,
                "side": entry.side,
                "units": abs(entry.entry_units),
                "reference_price": reference,
                "broker_fill_price": broker_price,
                "adverse_slippage_pips": round(
                    max(0.0, signed_slippage_pips), 12
                ),
                "signed_slippage_pips": round(signed_slippage_pips, 12),
                "fill_time_utc": entry.broker_entry_ts_utc,
            }
        )

    system_entries_by_trade = {entry.trade_id: entry for entry in entries}
    exit_rows: dict[str, list[dict[str, Any]]] = {
        "TAKE_PROFIT_ORDER": [],
        "STOP_LOSS_ORDER": [],
    }
    for trade_id, entry in system_entries_by_trade.items():
        for close in closes.get(trade_id, []):
            reason = str(close.get("exit_reason") or "").upper()
            if reason not in exit_rows:
                continue
            order_id = _identifier_text(close.get("order_id"))
            authoritative = authoritative_protections.get(
                (order_id, trade_id), []
            )
            normalized_protection = protections.get(
                (order_id, trade_id), []
            )
            if not authoritative_transactions:
                authoritative = [
                    raw
                    for row in normalized_protection
                    if (
                        raw := _optional_json_object(row.get("raw_json"))
                    )
                    is not None
                ]
            if not order_id or len(authoritative) != 1:
                return None
            protection_raw = authoritative[0]
            if len(normalized_protection) > 1:
                return None
            if normalized_protection:
                normalized = normalized_protection[0]
                if (
                    int(normalized.get("ledger_rowid") or 0)
                    >= int(close.get("ledger_rowid") or 0)
                    or (
                        authoritative_transactions
                        and _optional_json_object(
                            normalized.get("raw_json")
                        )
                        != protection_raw
                    )
                ):
                    return None
            close_raw = _optional_json_object(close.get("raw_json"))
            components = (
                close_raw.get("tradesClosed")
                if isinstance(close_raw, dict)
                else None
            )
            matching_components = [
                component
                for component in components or []
                if isinstance(component, dict)
                and _identifier_text(component.get("tradeID")) == trade_id
            ]
            component = (
                matching_components[0]
                if len(matching_components) == 1
                else None
            )
            trigger = _finite_float(protection_raw.get("price"))
            broker_price = (
                _finite_float(component.get("price"))
                if isinstance(component, dict)
                else None
            )
            comparable_prices = (
                _finite_float(close_raw.get("price"))
                if isinstance(close_raw, dict)
                else None,
                _finite_float(close_raw.get("fullVWAP"))
                if isinstance(close_raw, dict)
                else None,
            )
            if authoritative_transactions:
                comparable_prices = (
                    _finite_float(close.get("price")),
                    *comparable_prices,
                )
            expected_type = (
                "TAKE_PROFIT_ORDER"
                if reason == "TAKE_PROFIT_ORDER"
                else "STOP_LOSS_ORDER"
            )
            if (
                trigger is None
                or trigger <= 0.0
                or broker_price is None
                or broker_price <= 0.0
                or any(
                    price is None
                    or not math.isclose(
                        float(price), broker_price, rel_tol=0.0, abs_tol=1e-12
                    )
                    for price in comparable_prices
                )
                or not isinstance(protection_raw, dict)
                or str(protection_raw.get("type") or "").upper()
                != expected_type
                or _identifier_text(protection_raw.get("id")) != order_id
                or _identifier_text(protection_raw.get("tradeID")) != trade_id
                or not math.isclose(
                    float(_finite_float(protection_raw.get("price")) or -1.0),
                    trigger,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or _identifier_text(close_raw.get("orderID") if isinstance(close_raw, dict) else None)
                != order_id
                or str(close_raw.get("reason") or "").upper() != reason
            ):
                return None
            pip_factor = instrument_pip_factor(entry.pair)
            signed_slippage_pips = (
                (trigger - broker_price) * pip_factor
                if entry.side == "LONG"
                else (broker_price - trigger) * pip_factor
            )
            exit_rows[reason].append(
                {
                    "order_id": order_id,
                    "trade_id": trade_id,
                    "pair": entry.pair,
                    "entry_side": entry.side,
                    "reason": reason,
                    "trigger_price": trigger,
                    "broker_fill_price": broker_price,
                    "adverse_slippage_pips": round(
                        max(0.0, signed_slippage_pips), 12
                    ),
                    "signed_slippage_pips": round(
                        signed_slippage_pips, 12
                    ),
                    "fill_time_utc": str(close_raw.get("time") or ""),
                }
            )

    def summarized(rows: list[dict[str, Any]]) -> dict[str, Any]:
        parsed_rows = [
            (row, _parse_utc_instant(row.get("fill_time_utc")))
            for row in rows
        ]
        if any(timestamp is None for _, timestamp in parsed_rows):
            raise ValueError("slippage calibration row has invalid fill time")
        latest_instant = max(
            (
                timestamp
                for _, timestamp in parsed_rows
                if timestamp is not None
            ),
            default=None,
        )
        if latest_instant is not None:
            window_start = latest_instant - timedelta(
                seconds=EXECUTION_COST_MAX_SAMPLE_AGE_SECONDS
            )
            rows = [
                row
                for row, timestamp in parsed_rows
                if timestamp is not None and timestamp >= window_start
            ]
        adverse = [float(row["adverse_slippage_pips"]) for row in rows]
        latest_row = max(
            rows,
            key=lambda row: _rfc3339_utc_key(row["fill_time_utc"])
            or ("", 0),
        ) if rows else None
        oldest_row = min(
            rows,
            key=lambda row: _rfc3339_utc_key(row["fill_time_utc"])
            or ("", 0),
        ) if rows else None
        p95 = _nearest_rank_percentile(adverse, 0.95)
        return {
            "samples": len(rows),
            "adverse_p95_pips": round(p95, 12) if p95 is not None else None,
            "adverse_max_pips": round(max(adverse), 12) if adverse else None,
            "latest_fill_utc": (
                str(latest_row["fill_time_utc"]) if latest_row else None
            ),
            "oldest_fill_utc": (
                str(oldest_row["fill_time_utc"]) if oldest_row else None
            ),
            "rows_sha256": _canonical_json_sha256(rows),
        }

    return {
        "market_entry": summarized(entry_rows),
        "take_profit_exit": summarized(exit_rows["TAKE_PROFIT_ORDER"]),
        "stop_loss_exit": summarized(exit_rows["STOP_LOSS_ORDER"]),
    }


def read_execution_cost_surface(ledger_path: Path) -> dict[str, Any]:
    """Return one content-addressed execution-cost calibration surface."""

    try:
        outcomes = read_attributed_net_outcomes(ledger_path)
        slippage = _read_audited_execution_slippage(ledger_path)
        if outcomes is None or slippage is None:
            raise ValueError("execution-cost source is unreadable")
        financing = _execution_cost_financing_metrics(outcomes)
        material = {
            "contract": EXECUTION_COST_FLOOR_CONTRACT,
            "parse_status": "VALID",
            "scope": "SYSTEM_GATEWAY_ATTRIBUTED_ALL_PAIRS_SIDES_METHODS",
            "minimum_samples": EXECUTION_COST_MIN_SAMPLES,
            "maximum_sample_age_seconds": (
                EXECUTION_COST_MAX_SAMPLE_AGE_SECONDS
            ),
            **slippage,
            "global_financing": financing,
        }
    except (OSError, sqlite3.Error, TypeError, ValueError):
        material = {
            "contract": EXECUTION_COST_FLOOR_CONTRACT,
            "parse_status": "INVALID",
            "scope": "SYSTEM_GATEWAY_ATTRIBUTED_ALL_PAIRS_SIDES_METHODS",
            "minimum_samples": EXECUTION_COST_MIN_SAMPLES,
            "maximum_sample_age_seconds": (
                EXECUTION_COST_MAX_SAMPLE_AGE_SECONDS
            ),
            "market_entry": {},
            "take_profit_exit": {},
            "stop_loss_exit": {},
            "global_financing": {},
        }
    return {
        **material,
        "execution_cost_surface_sha256": _canonical_json_sha256(material),
    }


def execution_cost_floor_from_surface(
    surface: Mapping[str, Any] | None,
    *,
    exact_key: tuple[str, str, str, str],
    as_of: datetime,
) -> dict[str, Any]:
    """Bind global transport tails and exact financing to one ordinary lane."""

    source = surface if isinstance(surface, Mapping) else {}
    cost = (
        source.get("execution_cost")
        if isinstance(source.get("execution_cost"), Mapping)
        else {}
    )
    net_metrics = exact_vehicle_metrics_from_surface(
        source,
        field="exact_vehicle_net",
    )
    exact = dict(net_metrics.get(exact_key) or {}) if net_metrics else {}
    as_of_utc = as_of.astimezone(timezone.utc)

    def strict_nonnegative_number(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) and parsed >= 0.0 else None

    def strict_count(value: Any) -> int | None:
        return (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            else None
        )

    minimum = strict_count(cost.get("minimum_samples"))
    max_age = strict_count(cost.get("maximum_sample_age_seconds"))
    failed: list[str] = []
    if (
        source.get("parse_status") != "VALID"
        or cost.get("parse_status") != "VALID"
        or cost.get("contract") != EXECUTION_COST_FLOOR_CONTRACT
        or minimum != EXECUTION_COST_MIN_SAMPLES
        or max_age != EXECUTION_COST_MAX_SAMPLE_AGE_SECONDS
    ):
        failed.append("EXECUTION_COST_SURFACE_INVALID")

    sections: dict[str, Mapping[str, Any]] = {}
    for name in (
        "market_entry",
        "take_profit_exit",
        "stop_loss_exit",
        "global_financing",
    ):
        section = cost.get(name)
        if not isinstance(section, Mapping):
            failed.append(f"{name.upper()}_MISSING")
            section = {}
        sections[name] = section

    for name in ("market_entry", "take_profit_exit", "stop_loss_exit"):
        samples = strict_count(sections[name].get("samples"))
        p95 = strict_nonnegative_number(
            sections[name].get("adverse_p95_pips")
        )
        latest = _parse_utc_instant(sections[name].get("latest_fill_utc"))
        oldest = _parse_utc_instant(sections[name].get("oldest_fill_utc"))
        rows_sha = str(sections[name].get("rows_sha256") or "")
        age = (
            (as_of_utc - latest).total_seconds()
            if latest is not None
            else math.inf
        )
        if samples is None or minimum is None or samples < minimum:
            failed.append(f"{name.upper()}_SAMPLE_FLOOR_NOT_MET")
        if p95 is None:
            failed.append(f"{name.upper()}_P95_INVALID")
        if (
            latest is None
            or max_age is None
            or age < -60.0
            or age > max_age
        ):
            failed.append(f"{name.upper()}_STALE")
        if (
            latest is None
            or oldest is None
            or max_age is None
            or (latest - oldest).total_seconds() > max_age
        ):
            failed.append(f"{name.upper()}_COHORT_EXCEEDS_AGE_WINDOW")
        if not re.fullmatch(r"[0-9a-f]{64}", rows_sha):
            failed.append(f"{name.upper()}_DIGEST_INVALID")

    global_financing = sections["global_financing"]
    global_observations = strict_count(
        global_financing.get("observation_trades")
    )
    global_adverse_trades = strict_count(
        global_financing.get("adverse_trades")
    )
    global_adverse_mean = strict_nonnegative_number(
        global_financing.get("adverse_mean_jpy_per_unit")
    )
    global_financing_stress = strict_nonnegative_number(
        global_financing.get("adverse_stress_jpy_per_unit")
    )
    global_latest = _parse_utc_instant(
        global_financing.get("latest_observation_utc")
    )
    global_oldest = _parse_utc_instant(
        global_financing.get("oldest_observation_utc")
    )
    global_age = (
        (as_of_utc - global_latest).total_seconds()
        if global_latest is not None
        else math.inf
    )
    if (
        global_observations is None
        or minimum is None
        or global_observations < minimum
    ):
        failed.append("GLOBAL_FINANCING_SAMPLE_FLOOR_NOT_MET")
    if global_financing_stress is None:
        failed.append("GLOBAL_FINANCING_STRESS_INVALID")
    if (
        global_adverse_trades is None
        or global_adverse_trades < 1
        or global_adverse_mean is None
        or global_adverse_mean <= 0.0
        or global_financing_stress is None
        or global_financing_stress <= 0.0
    ):
        failed.append("GLOBAL_ADVERSE_FINANCING_OBSERVATION_MISSING")
    if (
        global_latest is None
        or max_age is None
        or global_age < -60.0
        or global_age > max_age
    ):
        failed.append("GLOBAL_FINANCING_STALE")
    if (
        global_latest is None
        or global_oldest is None
        or max_age is None
        or (global_latest - global_oldest).total_seconds() > max_age
    ):
        failed.append("GLOBAL_FINANCING_COHORT_EXCEEDS_AGE_WINDOW")

    exact_observations = strict_count(exact.get("financing_observation_trades"))
    exact_trades = strict_count(exact.get("trades"))
    exact_financing_stress = strict_nonnegative_number(
        exact.get("financing_adverse_stress_jpy_per_unit")
    )
    exact_latest = _parse_utc_instant(
        exact.get("financing_latest_observation_utc")
    )
    exact_oldest = _parse_utc_instant(
        exact.get("financing_oldest_observation_utc")
    )
    exact_age = (
        (as_of_utc - exact_latest).total_seconds()
        if exact_latest is not None
        else math.inf
    )
    if (
        exact_observations is None
        or exact_trades is None
        or exact_observations <= 0
        or exact_observations > exact_trades
        or exact_financing_stress is None
    ):
        failed.append("EXACT_FINANCING_STRESS_INVALID")
    if (
        exact_latest is None
        or max_age is None
        or exact_age < -60.0
        or exact_age > max_age
    ):
        failed.append("EXACT_FINANCING_STALE")
    if (
        exact_latest is None
        or exact_oldest is None
        or max_age is None
        or (exact_latest - exact_oldest).total_seconds() > max_age
    ):
        failed.append("EXACT_FINANCING_COHORT_EXCEEDS_AGE_WINDOW")

    entry_p95 = strict_nonnegative_number(
        sections["market_entry"].get("adverse_p95_pips")
    )
    tp_exit_p95 = strict_nonnegative_number(
        sections["take_profit_exit"].get("adverse_p95_pips")
    )
    sl_exit_p95 = strict_nonnegative_number(
        sections["stop_loss_exit"].get("adverse_p95_pips")
    )
    audited_exit_p95 = (
        max(tp_exit_p95, sl_exit_p95)
        if tp_exit_p95 is not None and sl_exit_p95 is not None
        else None
    )
    financing_stress = (
        max(global_financing_stress, exact_financing_stress)
        if global_financing_stress is not None
        and exact_financing_stress is not None
        else None
    )
    surface_sha = str(cost.get("execution_cost_surface_sha256") or "")
    cost_material = dict(cost)
    cost_material.pop("execution_cost_surface_sha256", None)
    if (
        not re.fullmatch(r"[0-9a-f]{64}", surface_sha)
        or surface_sha != _canonical_json_sha256(cost_material)
    ):
        failed.append("EXECUTION_COST_SURFACE_DIGEST_MISMATCH")
    material = {
        "contract": EXECUTION_COST_FLOOR_CONTRACT,
        "status": "PASSED" if not failed else "BLOCKED",
        "reason": (
            "DYNAMIC_LEDGER_EXECUTION_COST_FLOOR_PROVEN"
            if not failed
            else failed[0]
        ),
        "failed_checks": failed,
        "scope_key": "|".join(exact_key),
        "execution_cost_surface_sha256": surface_sha or None,
        "minimum_samples": minimum,
        "maximum_sample_age_seconds": max_age,
        "market_entry_samples": strict_count(
            sections["market_entry"].get("samples")
        ),
        "market_entry_adverse_p95_pips": entry_p95,
        "market_entry_latest_fill_utc": sections["market_entry"].get(
            "latest_fill_utc"
        ),
        "market_entry_oldest_fill_utc": sections["market_entry"].get(
            "oldest_fill_utc"
        ),
        "take_profit_exit_samples": strict_count(
            sections["take_profit_exit"].get("samples")
        ),
        "take_profit_exit_adverse_p95_pips": tp_exit_p95,
        "take_profit_exit_latest_fill_utc": sections[
            "take_profit_exit"
        ].get("latest_fill_utc"),
        "take_profit_exit_oldest_fill_utc": sections[
            "take_profit_exit"
        ].get("oldest_fill_utc"),
        "stop_loss_exit_samples": strict_count(
            sections["stop_loss_exit"].get("samples")
        ),
        "stop_loss_exit_adverse_p95_pips": sl_exit_p95,
        "stop_loss_exit_latest_fill_utc": sections[
            "stop_loss_exit"
        ].get("latest_fill_utc"),
        "stop_loss_exit_oldest_fill_utc": sections[
            "stop_loss_exit"
        ].get("oldest_fill_utc"),
        "audited_protected_exit_adverse_p95_pips": audited_exit_p95,
        "global_financing_observation_trades": global_observations,
        "global_financing_adverse_trades": global_adverse_trades,
        "global_financing_adverse_mean_jpy_per_unit": global_adverse_mean,
        "global_financing_adverse_stress_jpy_per_unit": (
            global_financing_stress
        ),
        "global_financing_latest_observation_utc": global_financing.get(
            "latest_observation_utc"
        ),
        "global_financing_oldest_observation_utc": global_financing.get(
            "oldest_observation_utc"
        ),
        "exact_financing_observation_trades": exact_observations,
        "exact_financing_adverse_stress_jpy_per_unit": (
            exact_financing_stress
        ),
        "exact_financing_latest_observation_utc": exact.get(
            "financing_latest_observation_utc"
        ),
        "exact_financing_oldest_observation_utc": exact.get(
            "financing_oldest_observation_utc"
        ),
        "financing_floor_basis": "MAX_GLOBAL_AND_EXACT_WILSON95_UPPER_STRESS",
        "financing_adverse_stress_jpy_per_unit": financing_stress,
        "spread_double_count_forbidden": True,
    }
    return {
        **material,
        "proof_sha256": _canonical_json_sha256(material),
    }


def read_exact_vehicle_allocation_surface(ledger_path: Path) -> dict[str, Any]:
    """Return a WAL-safe semantic snapshot of allocation-relevant ledger truth.

    SQLite's main file hash does not include uncheckpointed WAL rows. The
    online backup API copies one coherent database snapshot including WAL, and
    both exact-vehicle readers then evaluate that immutable copy. The returned
    digest therefore changes for a new audited close/reduction even when the
    main ``.db`` bytes have not changed yet.
    """

    path = Path(ledger_path).expanduser().resolve(strict=False)
    if not path.exists() or not path.is_file():
        missing_execution_cost = read_execution_cost_surface(path)
        material = {
            "contract": EXACT_VEHICLE_ALLOCATION_SURFACE_CONTRACT,
            "parse_status": "MISSING",
            "coverage_start_utc": None,
            "latest_realized_event": None,
            "last_oanda_transaction_id": None,
            "exact_vehicle_net": [],
            "exact_vehicle_take_profit": [],
            "execution_cost": missing_execution_cost,
        }
        return {
            **material,
            "allocation_surface_sha256": _canonical_json_sha256(material),
        }
    try:
        with tempfile.TemporaryDirectory(prefix="qr-ledger-allocation-surface-") as tmp:
            snapshot = Path(tmp) / "execution_ledger_snapshot.db"
            with closing(
                sqlite3.connect(
                    f"file:{path}?mode=ro",
                    uri=True,
                    timeout=30.0,
                )
            ) as source, closing(
                sqlite3.connect(snapshot, timeout=30.0)
            ) as destination:
                # The Guardian and hourly trader may briefly commit to the WAL
                # while the allocation packet takes its semantic snapshot. A
                # five-second default timeout turned that ordinary overlap into
                # parse_status=INVALID, then made the same untouched ledger
                # VALID at apply time and invalidated the AI handoff. Wait for
                # the bounded writer window; persistent lock/corruption still
                # falls through to the fail-closed INVALID surface below.
                source.execute("PRAGMA busy_timeout=30000")
                destination.execute("PRAGMA busy_timeout=30000")
                source.backup(destination, sleep=0.05)
            # `sqlite3_backup` preserves the source's WAL journal mode.  The
            # immutable temp snapshot has no pre-existing `-shm` file, so the
            # strict read-only readers below otherwise fail with "unable to
            # open database file" before they can inspect valid rows. Convert
            # only this isolated copy to DELETE mode; the live ledger and its
            # WAL remain untouched.
            with closing(sqlite3.connect(snapshot)) as normalized:
                normalized.execute("PRAGMA journal_mode=DELETE").fetchone()
            net_metrics = read_exact_vehicle_net_metrics(snapshot)
            tp_metrics = read_exact_vehicle_take_profit_metrics(snapshot)
            execution_cost = read_execution_cost_surface(snapshot)
            if (
                net_metrics is None
                or tp_metrics is None
            ):
                raise ValueError("exact-vehicle ledger surface is unreadable")
            coverage_start, latest_event, last_transaction_id = (
                _allocation_surface_ledger_identity(snapshot)
            )
    except (OSError, sqlite3.Error, TypeError, ValueError):
        material = {
            "contract": EXACT_VEHICLE_ALLOCATION_SURFACE_CONTRACT,
            "parse_status": "INVALID",
            "coverage_start_utc": None,
            "latest_realized_event": None,
            "last_oanda_transaction_id": None,
            "exact_vehicle_net": [],
            "exact_vehicle_take_profit": [],
            "execution_cost": read_execution_cost_surface(path),
        }
        return {
            **material,
            "allocation_surface_sha256": _canonical_json_sha256(material),
        }

    material = {
        "contract": EXACT_VEHICLE_ALLOCATION_SURFACE_CONTRACT,
        "parse_status": "VALID",
        "coverage_start_utc": coverage_start,
        "latest_realized_event": latest_event,
        "last_oanda_transaction_id": last_transaction_id,
        "exact_vehicle_net": _serialize_exact_vehicle_metrics(net_metrics),
        "exact_vehicle_take_profit": _serialize_exact_vehicle_metrics(tp_metrics),
        "execution_cost": execution_cost,
    }
    return {
        **material,
        "allocation_surface_sha256": _canonical_json_sha256(material),
    }


def exact_vehicle_metrics_from_surface(
    surface: Mapping[str, Any] | None,
    *,
    field: str,
) -> dict[tuple[str, str, str, str], dict[str, Any]] | None:
    """Rehydrate one canonical metric map from an allocation surface."""

    if not isinstance(surface, Mapping) or surface.get("parse_status") != "VALID":
        return None
    raw_rows = surface.get(field)
    if not isinstance(raw_rows, list):
        return None
    rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            return None
        key_parts = tuple(
            str(raw.get(name) or "").strip().upper()
            for name in ("pair", "side", "method", "vehicle")
        )
        if (
            len(key_parts) != 4
            or not all(key_parts)
            or key_parts[1] not in {"LONG", "SHORT"}
            or key_parts[3] not in {"LIMIT", "MARKET", "STOP"}
            or key_parts in rows
        ):
            return None
        rows[key_parts] = {
            str(name): value
            for name, value in raw.items()
            if name not in {"pair", "side", "method", "vehicle"}
        }
    return rows


def _serialize_exact_vehicle_metrics(
    metrics: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (pair, side, method, vehicle), values in sorted(metrics.items()):
        rows.append(
            {
                "pair": str(pair).upper(),
                "side": str(side).upper(),
                "method": str(method).upper(),
                "vehicle": str(vehicle).upper(),
                **{str(name): value for name, value in sorted(values.items())},
            }
        )
    return rows


def _allocation_surface_ledger_identity(
    ledger_path: Path,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    with closing(sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True)) as conn, conn:
        conn.row_factory = sqlite3.Row
        tables = {
            str(row["name"])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        coverage_start: str | None = None
        if "sync_state" in tables:
            row = conn.execute(
                "SELECT value FROM sync_state WHERE key=?",
                (_OANDA_TRANSACTION_COVERAGE_START_KEY,),
            ).fetchone()
            coverage_start = str(row["value"]) if row is not None else None
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(execution_events)").fetchall()
        }
        required = {"ts_utc", "event_type", "trade_id", "raw_json"}
        if not required.issubset(columns):
            raise ValueError("execution_events identity columns are missing")
        optional = [
            name
            for name in ("event_uid", "order_id", "oanda_transaction_id")
            if name in columns
        ]
        select_columns = ["rowid", "ts_utc", "event_type", "trade_id", "raw_json", *optional]
        latest = conn.execute(
            f"""
            SELECT {', '.join(select_columns)}
            FROM execution_events
            WHERE event_type IN ('TRADE_REDUCED', 'TRADE_CLOSED')
            ORDER BY rowid DESC
            LIMIT 1
            """
        ).fetchone()
        latest_event = None
        if latest is not None:
            raw_json = latest["raw_json"]
            raw_bytes = (
                raw_json.encode("utf-8")
                if isinstance(raw_json, str)
                else b""
            )
            latest_event = {
                "rowid": int(latest["rowid"]),
                "ts_utc": latest["ts_utc"],
                "event_type": latest["event_type"],
                "trade_id": latest["trade_id"],
                "raw_json_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                **{name: latest[name] for name in optional},
            }
        last_transaction_id: str | None = None
        if "oanda_transactions" in tables:
            row = conn.execute(
                """
                SELECT transaction_id
                FROM oanda_transactions
                ORDER BY CAST(transaction_id AS INTEGER) DESC, transaction_id DESC
                LIMIT 1
                """
            ).fetchone()
            last_transaction_id = (
                str(row["transaction_id"]) if row is not None else None
            )
    return coverage_start, latest_event, last_transaction_id


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _aggregate_exact_vehicle_outcomes(
    outcomes: list[RealizedOutcome],
    *,
    pure_take_profit_only: bool,
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    accum: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    financing_outcomes: dict[
        tuple[str, str, str, str], list[RealizedOutcome]
    ] = {}
    for outcome in outcomes:
        if not outcome.entry_truth_consistent:
            continue
        if pure_take_profit_only and not outcome.pure_take_profit_lifecycle:
            continue
        pair = str(outcome.pair or "UNKNOWN").upper()
        side = str(outcome.side or "UNKNOWN").upper()
        method = str(outcome.method or "UNKNOWN").upper()
        vehicle = str(outcome.entry_vehicle or "UNKNOWN").upper()
        if pair == "UNKNOWN" or side not in {"LONG", "SHORT"}:
            continue
        if method == "UNKNOWN" or vehicle not in {"LIMIT", "MARKET", "STOP"}:
            continue
        net_jpy = float(outcome.realized_pl_jpy)
        key = (pair, side, method, vehicle)
        financing_outcomes.setdefault(key, []).append(outcome)
        slot = accum.setdefault(
            key,
            {
                "trades": 0.0,
                "wins": 0.0,
                "losses": 0.0,
                "net_jpy": 0.0,
                "win_jpy": 0.0,
                "loss_jpy": 0.0,
                "entry_units_total": 0.0,
                "financing_adverse_trades": 0.0,
                "financing_adverse_total_jpy": 0.0,
                "financing_adverse_per_unit_sum": 0.0,
                "latest_broker_close_ts_utc": "",
                "oldest_broker_close_ts_utc": "",
            },
        )
        slot["trades"] += 1
        slot["wins"] += 1 if net_jpy > 0.0 else 0
        slot["losses"] += 1 if net_jpy < 0.0 else 0
        slot["net_jpy"] += net_jpy
        slot["win_jpy"] += net_jpy if net_jpy > 0.0 else 0.0
        slot["loss_jpy"] += net_jpy if net_jpy < 0.0 else 0.0
        entry_units = abs(float(outcome.entry_units))
        financing = float(outcome.audited_financing_jpy)
        adverse_financing = float(outcome.adverse_financing_jpy)
        if (
            not math.isfinite(entry_units)
            or entry_units <= 0.0
            or not math.isfinite(financing)
            or not math.isfinite(adverse_financing)
            or adverse_financing < 0.0
        ):
            # The strict ledger reader should make this unreachable. Keeping
            # the aggregate fail-closed prevents a future alternate caller
            # from manufacturing a zero financing rate with missing units.
            raise ValueError(
                "exact-vehicle financing observation has invalid entry units"
            )
        slot["entry_units_total"] += entry_units
        close_key = _rfc3339_utc_key(outcome.broker_close_ts_utc)
        current_latest_key = _rfc3339_utc_key(
            slot["latest_broker_close_ts_utc"]
        )
        current_oldest_key = _rfc3339_utc_key(
            slot["oldest_broker_close_ts_utc"]
        )
        if close_key is None:
            raise ValueError(
                "exact-vehicle financing observation has no broker close time"
            )
        if current_latest_key is None or close_key > current_latest_key:
            slot["latest_broker_close_ts_utc"] = (
                outcome.broker_close_ts_utc
            )
        if current_oldest_key is None or close_key < current_oldest_key:
            slot["oldest_broker_close_ts_utc"] = (
                outcome.broker_close_ts_utc
            )
        if adverse_financing > 0.0:
            adverse = adverse_financing
            slot["financing_adverse_trades"] += 1
            slot["financing_adverse_total_jpy"] += adverse
            slot["financing_adverse_per_unit_sum"] += adverse / entry_units

    metrics: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for key, slot in accum.items():
        # Cost calibration is intentionally a rolling cohort even though the
        # exact-vehicle realized edge remains a lifetime ledger aggregate.
        # Otherwise an arbitrary number of old zero/credit observations can
        # dilute a recent adverse financing event.  The helper uses the exact
        # key's own latest observation as the 90-day anchor; the consumer also
        # checks that anchor against the current quote time.
        financing_metrics = _execution_cost_financing_metrics(
            financing_outcomes[key]
        )
        trades = int(slot["trades"])
        wins = int(slot["wins"])
        losses = int(slot["losses"])
        net_jpy = float(slot["net_jpy"])
        adverse_financing_trades = int(
            financing_metrics["adverse_trades"]
        )
        adverse_mean_per_unit = float(
            financing_metrics["adverse_mean_jpy_per_unit"]
        )
        adverse_occurrence_upper = financing_metrics[
            "adverse_occurrence_wilson95_upper"
        ]
        adverse_stress_per_unit = financing_metrics[
            "adverse_stress_jpy_per_unit"
        ]
        metrics[key] = {
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "expectancy_jpy_per_trade": round(net_jpy / trades, 4) if trades else 0.0,
            "avg_win_jpy": round(float(slot["win_jpy"]) / wins, 4) if wins else 0.0,
            "avg_loss_jpy": (
                round(abs(float(slot["loss_jpy"])) / losses, 4) if losses else 0.0
            ),
            "net_jpy": round(net_jpy, 4),
            "entry_units_total": round(float(slot["entry_units_total"]), 4),
            "financing_observation_trades": int(
                financing_metrics["observation_trades"]
            ),
            "financing_adverse_trades": adverse_financing_trades,
            "financing_adverse_total_jpy": financing_metrics[
                "adverse_total_jpy"
            ],
            "financing_adverse_mean_jpy_per_unit": round(
                adverse_mean_per_unit,
                12,
            ),
            "financing_adverse_occurrence_wilson95_upper": (
                round(adverse_occurrence_upper, 12)
                if adverse_occurrence_upper is not None
                else None
            ),
            "financing_adverse_stress_jpy_per_unit": (
                round(adverse_stress_per_unit, 12)
                if adverse_stress_per_unit is not None
                else None
            ),
            "financing_latest_observation_utc": financing_metrics[
                "latest_observation_utc"
            ],
            "financing_oldest_observation_utc": financing_metrics[
                "oldest_observation_utc"
            ],
            "latest_broker_close_ts_utc": str(
                slot["latest_broker_close_ts_utc"]
            ),
            "oldest_broker_close_ts_utc": str(
                slot["oldest_broker_close_ts_utc"]
            ),
            "source_scope": "PAIR_SIDE_METHOD_VEHICLE",
            "exit_scope": (
                "PURE_TAKE_PROFIT_LIFECYCLE"
                if pure_take_profit_only
                else "ALL_AUDITED_EXITS"
            ),
        }
    return metrics


def _wilson95_upper(rate: float, samples: int) -> float | None:
    """Wilson 95% upper bound, mirrored from the shared lower bound."""

    if samples <= 0 or not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        return None
    lower_for_complement = hit_rate_wilson_lower(1.0 - rate, samples)
    return (
        1.0 - lower_for_complement
        if lower_for_complement is not None
        else None
    )


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _parse_utc_instant(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    # OANDA emits RFC3339 timestamps with nanosecond precision while Python
    # 3.9's ``datetime.fromisoformat`` accepts at most six fractional digits.
    # Normalize only the fractional component; do not round into the next
    # second because ledger ordering and coverage comparisons are conservative.
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})",
        text,
    )
    if match is not None:
        prefix, fraction, offset = match.groups()
        micros = f".{fraction[:6].ljust(6, '0')}" if fraction else ""
        text = f"{prefix}{micros}{'+00:00' if offset == 'Z' else offset}"
    else:
        text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _rfc3339_utc_key(value: object) -> tuple[str, int] | None:
    """Return an exact UTC-second/nanosecond key without Python 3.9 truncation.

    OANDA timestamps carry up to nanosecond precision.  ``datetime`` on the
    production Python 3.9 runtime cannot retain the final three digits, so a
    plain datetime comparison could accept a one-nanosecond ledger tamper.
    """

    if not isinstance(value, str) or not value.strip():
        return None
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})",
        value.strip(),
    )
    if match is None:
        return None
    prefix, fraction, offset = match.groups()
    try:
        base = datetime.fromisoformat(
            f"{prefix}{'+00:00' if offset == 'Z' else offset}"
        )
    except ValueError:
        return None
    if base.tzinfo is None:
        return None
    utc_second = base.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    nanosecond = int((fraction or "0").ljust(9, "0"))
    return utc_second, nanosecond


def _broker_timestamp_matches(*, normalized: object, broker: object) -> bool:
    normalized_key = _rfc3339_utc_key(normalized)
    return normalized_key is not None and normalized_key == _rfc3339_utc_key(broker)


def _optional_json_object(value: object) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _unsupported_cash_is_zero(raw: object) -> bool:
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key in _UNSUPPORTED_CASH_FIELDS and value is not None:
                amount = _finite_float(value)
                if amount is None or abs(amount) > 0.0:
                    return False
            if not _unsupported_cash_is_zero(value):
                return False
    elif isinstance(raw, list):
        return all(_unsupported_cash_is_zero(value) for value in raw)
    return True


def _raw_has_nonzero_cash(raw: dict[str, Any]) -> bool:
    cash_keys = {
        "amount",
        "pl",
        "realizedPL",
        "financing",
        *_UNSUPPORTED_CASH_FIELDS,
    }

    def walk(value: object) -> bool:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in cash_keys and child is not None:
                    amount = _finite_float(child)
                    if amount is None or abs(amount) > 0.0:
                        return True
                if walk(child):
                    return True
        elif isinstance(value, list):
            return any(walk(child) for child in value)
        return False

    return walk(raw)


def _audited_financing_by_trade(
    rows: list[dict[str, Any]],
    *,
    nonzero_events_by_trade: dict[str, list[tuple[str, float]]] | None = None,
) -> dict[str, float] | None:
    allocated: dict[str, float] = {}
    tolerance = FINANCING_COMPONENT_RECONCILIATION_TOLERANCE_JPY
    for row in rows:
        if str(row.get("event_type") or "") != "OANDA_TRANSACTION":
            continue
        raw = _optional_json_object(row.get("raw_json"))
        if raw is None or not isinstance(raw.get("type"), str) or not raw["type"].strip():
            return None
        if not _unsupported_cash_is_zero(raw):
            return None
        transaction_type = str(raw["type"]).upper()
        ledger_financing = _finite_float(row.get("financing_jpy"))
        if transaction_type == "DAILY_FINANCING":
            raw_total = _finite_float(raw.get("financing"))
            position_financings = raw.get("positionFinancings")
            if raw_total is None or ledger_financing is None or not isinstance(position_financings, list):
                return None
            transaction_components: list[tuple[str, float]] = []
            for position in position_financings:
                if not isinstance(position, dict):
                    return None
                open_trades = position.get("openTradeFinancings")
                if not isinstance(open_trades, list):
                    return None
                for component in open_trades:
                    if not isinstance(component, dict):
                        return None
                    raw_trade_id = component.get("tradeID")
                    trade_id = (
                        str(raw_trade_id).strip()
                        if isinstance(raw_trade_id, (str, int)) and not isinstance(raw_trade_id, bool)
                        else ""
                    )
                    financing = _finite_float(component.get("financing"))
                    if not trade_id or financing is None:
                        return None
                    transaction_components.append((trade_id, financing))
            component_total = sum(amount for _, amount in transaction_components)
            if abs(component_total - raw_total) > tolerance or abs(component_total - ledger_financing) > tolerance:
                return None
            # Allocate even when the top-level total is zero: system and
            # manual components may offset exactly at account level.
            for trade_id, amount in transaction_components:
                allocated[trade_id] = allocated.get(trade_id, 0.0) + amount
                if nonzero_events_by_trade is not None and amount != 0.0:
                    nonzero_events_by_trade.setdefault(trade_id, []).append(
                        (str(row.get("ts_utc") or ""), amount)
                    )
            continue

        # Known non-P/L transactions are intentionally outside realized trade
        # economics. Unknown transaction types are harmless only when they
        # prove they contain no cash adjustment at all.
        if transaction_type in _NON_PNL_OANDA_TRANSACTION_TYPES:
            if ledger_financing not in {None, 0.0}:
                return None
            continue
        if ledger_financing not in {None, 0.0} or _raw_has_nonzero_cash(raw):
            return None
        if raw.get("accountBalance") is not None:
            return None
    return allocated


def _audit_daily_financing_transaction_completeness(
    rows: list[dict[str, Any]],
    *,
    oanda_transaction_rows: list[dict[str, Any]],
    oanda_transactions_authoritative: bool,
) -> bool:
    """Require every authoritative DAILY_FINANCING row to remain normalized.

    Component arithmetic in :func:`_audited_financing_by_trade` proves each
    execution event that is present, but cannot detect an event that disappeared
    completely. Current ledgers retain the raw broker transaction table, so bind
    each broker-identified financing event one-to-one to that authoritative row.
    Legacy/synthetic events without a broker id remain internally audited when
    no authoritative counterpart exists.
    """

    if not oanda_transactions_authoritative:
        return True

    actual_broker_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("event_type") or "") != "OANDA_TRANSACTION":
            continue
        raw = _optional_json_object(row.get("raw_json"))
        if raw is None:
            return False
        if str(raw.get("type") or "").upper() != "DAILY_FINANCING":
            continue
        identity = _close_transaction_identity(
            oanda_transaction_id=row.get("oanda_transaction_id"),
            raw=raw,
            legacy_order_id=None,
        )
        if identity is None:
            # Once the authoritative table exists, every normalized financing
            # row must be broker-identifiable even when that table is empty.
            # Otherwise an id-less synthetic cash row can manufacture profit or
            # cancel broker carry without any authoritative counterpart.
            if _identifier_text(row.get("oanda_transaction_id")) or _identifier_text(
                raw.get("id")
            ):
                return False
            return False
        if identity in actual_broker_rows:
            return False
        actual_broker_rows[identity] = raw

    authoritative_rows: dict[str, dict[str, Any]] = {}
    for transaction in oanda_transaction_rows:
        raw = _optional_json_object(transaction.get("raw_json"))
        if raw is None or str(raw.get("type") or "").upper() != "DAILY_FINANCING":
            return False
        identity = _close_transaction_identity(
            oanda_transaction_id=transaction.get("transaction_id"),
            raw=raw,
            legacy_order_id=None,
        )
        if identity is None or identity in authoritative_rows:
            return False
        authoritative_rows[identity] = raw

    if set(authoritative_rows) != set(actual_broker_rows):
        return False
    return all(
        actual_broker_rows[identity] == raw
        for identity, raw in authoritative_rows.items()
    )


def _audit_close_transaction_completeness(
    rows: list[dict[str, Any]],
    *,
    oanda_transaction_rows: list[dict[str, Any]],
    oanda_transactions_authoritative: bool,
) -> bool:
    """Require one normalized close event per raw broker close component.

    One OANDA ``ORDER_FILL`` can reduce one trade and/or close several trades.
    Every normalized event stores the full raw transaction, so validating only
    the component named by that event would miss a parser/legacy omission of a
    sibling loss.  Group by the broker transaction id (falling back to the raw
    id or legacy order id), then compare the complete raw component map with
    the normalized event map exactly.  Duplicate and extra normalized rows are
    rejected as well, because either would double-count realized economics.

    Current ledgers also retain the authoritative ``oanda_transactions`` table.
    When present, it is compared against the normalized groups so a transaction
    whose *entire* close surface was omitted cannot disappear merely because no
    execution-event row survived.  Older fixture/legacy ledgers without that
    table still receive the strongest proof available from each event's raw
    transaction and otherwise fail closed.
    """

    actual_groups: dict[
        str,
        dict[str, Any],
    ] = {}
    for row in rows:
        event_type = str(row.get("event_type") or "")
        if event_type not in {"TRADE_CLOSED", "TRADE_REDUCED"}:
            continue
        raw = _optional_json_object(row.get("raw_json"))
        if raw is None or str(raw.get("type") or "").upper() != "ORDER_FILL":
            return False
        identity = _close_transaction_identity(
            oanda_transaction_id=row.get("oanda_transaction_id"),
            raw=raw,
            legacy_order_id=row.get("order_id"),
        )
        expected = _raw_close_component_map(raw)
        if identity is None or expected is None or not expected:
            return False
        group = actual_groups.setdefault(
            identity,
            {
                "raw_components": expected,
                "events": {},
            },
        )
        if group["raw_components"] != expected:
            return False
        trade_id = str(row.get("trade_id") or "").strip()
        actual_key = (event_type, trade_id)
        if not trade_id or actual_key in group["events"]:
            return False
        realized = _finite_float(row.get("realized_pl_jpy"))
        financing = _finite_float(row.get("financing_jpy"))
        if realized is None or financing is None:
            return False
        group["events"][actual_key] = (realized, financing)

    for group in actual_groups.values():
        expected = group["raw_components"]
        actual = group["events"]
        if set(actual) != set(expected):
            return False
        for key, (expected_realized, expected_financing) in expected.items():
            actual_realized, actual_financing = actual[key]
            if (
                abs(actual_realized - expected_realized)
                > FINANCING_COMPONENT_RECONCILIATION_TOLERANCE_JPY
                or abs(actual_financing - expected_financing)
                > FINANCING_COMPONENT_RECONCILIATION_TOLERANCE_JPY
            ):
                return False

    if not oanda_transactions_authoritative:
        return True

    authoritative_groups: dict[str, dict[tuple[str, str], tuple[float, float]]] = {}
    for transaction in oanda_transaction_rows:
        raw = _optional_json_object(transaction.get("raw_json"))
        if raw is None or str(raw.get("type") or "").upper() != "ORDER_FILL":
            return False
        expected = _raw_close_component_map(raw)
        if expected is None:
            return False
        if not expected:
            continue
        identity = _close_transaction_identity(
            oanda_transaction_id=transaction.get("transaction_id"),
            raw=raw,
            legacy_order_id=None,
        )
        if identity is None or identity in authoritative_groups:
            return False
        authoritative_groups[identity] = expected

    # Synthetic/legacy execution rows may coexist with an initialized but
    # historically empty transaction table (notably dry-run/test ledgers).
    # Their ``legacy-order:`` groups are still proved one-to-one from their own
    # raw ORDER_FILL above, but cannot be required to appear in broker history.
    # Conversely every broker-identified group must match the authoritative
    # table, and every authoritative close transaction must have normalized
    # events, so a real omitted transaction remains fail-closed.
    actual_broker_groups = {
        identity: group
        for identity, group in actual_groups.items()
        if identity.startswith("broker:")
    }
    if set(authoritative_groups) != set(actual_broker_groups):
        return False
    return all(
        actual_broker_groups[identity]["raw_components"] == expected
        for identity, expected in authoritative_groups.items()
    )


def _close_transaction_identity(
    *,
    oanda_transaction_id: object,
    raw: dict[str, Any],
    legacy_order_id: object,
) -> str | None:
    normalized_id = _identifier_text(oanda_transaction_id)
    raw_id = _identifier_text(raw.get("id"))
    if normalized_id and raw_id and normalized_id != raw_id:
        return None
    broker_id = normalized_id or raw_id
    if broker_id:
        return f"broker:{broker_id}"
    order_id = _identifier_text(legacy_order_id)
    if not order_id:
        return None
    # Old/synthetic ledgers may reuse human-readable order ids across fixtures
    # or imported sources.  The full raw transaction fingerprint separates
    # those independent rows while still grouping the identical raw payload
    # copied onto every sibling component of one real multi-close.
    raw_digest = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"legacy-order:{order_id}:{raw_digest}"


def _identifier_text(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if not isinstance(value, (str, int)):
        return ""
    return str(value).strip()


def _raw_close_component_map(
    raw: dict[str, Any],
) -> dict[tuple[str, str], tuple[float, float]] | None:
    """Return exact normalized event keys/amounts declared by one ORDER_FILL."""

    components: dict[tuple[str, str], tuple[float, float]] = {}

    if "tradeReduced" in raw:
        reduced = raw.get("tradeReduced")
        if not isinstance(reduced, dict):
            return None
        if not _append_raw_close_component(
            components,
            event_type="TRADE_REDUCED",
            component=reduced,
        ):
            return None

    if "tradesClosed" in raw:
        closed_items = raw.get("tradesClosed")
        if not isinstance(closed_items, list):
            return None
        for component in closed_items:
            if not isinstance(component, dict) or not _append_raw_close_component(
                components,
                event_type="TRADE_CLOSED",
                component=component,
            ):
                return None

    return components


def _append_raw_close_component(
    components: dict[tuple[str, str], tuple[float, float]],
    *,
    event_type: str,
    component: dict[str, Any],
) -> bool:
    trade_id = _identifier_text(component.get("tradeID"))
    realized = _finite_float(component.get("realizedPL"))
    financing = _finite_float(component.get("financing"))
    key = (event_type, trade_id)
    if (
        not trade_id
        or realized is None
        or financing is None
        or key in components
    ):
        return False
    components[key] = (realized, financing)
    return True


def _audit_close_row(row: dict[str, Any]) -> tuple[float, float, str, str] | None:
    trade_id = str(row.get("trade_id") or "").strip()
    raw = _optional_json_object(row.get("raw_json"))
    if not trade_id or raw is None or str(raw.get("type") or "").upper() != "ORDER_FILL":
        return None
    if not _unsupported_cash_is_zero(raw):
        return None
    broker_close_ts_utc = str(raw.get("time") or "").strip()
    if not _broker_timestamp_matches(
        normalized=row.get("ts_utc"),
        broker=broker_close_ts_utc,
    ):
        return None
    event_type = str(row.get("event_type") or "")
    components: list[dict[str, Any]] = []
    if event_type == "TRADE_CLOSED":
        raw_components = raw.get("tradesClosed")
        if not isinstance(raw_components, list):
            return None
        components = [
            component
            for component in raw_components
            if isinstance(component, dict)
            and str(component.get("tradeID") or "").strip() == trade_id
        ]
    elif event_type == "TRADE_REDUCED":
        component = raw.get("tradeReduced")
        if (
            isinstance(component, dict)
            and str(component.get("tradeID") or "").strip() == trade_id
        ):
            components = [component]
    if len(components) != 1:
        return None
    component = components[0]
    normalized_realized = _finite_float(row.get("realized_pl_jpy"))
    normalized_financing = _finite_float(row.get("financing_jpy"))
    raw_realized = _finite_float(component.get("realizedPL"))
    raw_financing = _finite_float(component.get("financing"))
    if None in {normalized_realized, normalized_financing, raw_realized, raw_financing}:
        return None
    assert normalized_realized is not None
    assert normalized_financing is not None
    assert raw_realized is not None
    assert raw_financing is not None
    tolerance = FINANCING_COMPONENT_RECONCILIATION_TOLERANCE_JPY
    if (
        abs(normalized_realized - raw_realized) > tolerance
        or abs(normalized_financing - raw_financing) > tolerance
    ):
        return None
    normalized_reason = str(row.get("exit_reason") or "").strip().upper()
    raw_reason = str(raw.get("reason") or "").strip().upper()
    if not normalized_reason or normalized_reason != raw_reason:
        return None
    return normalized_realized, normalized_financing, raw_reason, broker_close_ts_utc


def _manual_owner_marked(raw: object) -> bool:
    if not isinstance(raw, dict):
        return False

    def walk(value: object) -> bool:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in {"tag", "owner", "owner_tag"}:
                    if str(child or "").strip().lower() in _MANUAL_OWNER_MARKERS:
                        return True
                if walk(child):
                    return True
        elif isinstance(value, list):
            return any(walk(child) for child in value)
        return False

    return walk(raw)


def _manual_lane(lane_id: str) -> bool:
    lane = str(lane_id or "").strip().lower()
    if not lane:
        return False
    return lane.split(":", 1)[0] in _MANUAL_OWNER_MARKERS


def _entry_vehicle_from_event(row: dict[str, Any]) -> str:
    candidates = [row.get("exit_reason")]
    raw = _optional_json_object(row.get("raw_json"))
    if raw is not None:
        candidates.extend((raw.get("reason"), raw.get("type")))
    for candidate in candidates:
        value = str(candidate or "").strip().upper()
        if value == "LIMIT_ORDER":
            return "LIMIT"
        if value in {"STOP_ORDER", "MARKET_IF_TOUCHED_ORDER"}:
            return "STOP"
        if value == "MARKET_ORDER":
            return "MARKET"
    return "UNKNOWN"


def _lane_parts(lane_id: str) -> tuple[str, str, str, str]:
    parts = str(lane_id or "").split(":")
    if len(parts) < 4:
        return "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"
    vehicle = str(parts[4] or "UNKNOWN").upper() if len(parts) >= 5 else "UNKNOWN"
    return (
        str(parts[1] or "UNKNOWN").upper(),
        str(parts[2] or "UNKNOWN").upper(),
        str(parts[3] or "UNKNOWN").upper(),
        vehicle,
    )


def _audit_entry_fill_truth(fill: dict[str, Any]) -> dict[str, Any] | None:
    """Reconcile one normalized ORDER_FILLED entry to raw broker truth."""

    raw = _optional_json_object(fill.get("raw_json"))
    opened = raw.get("tradeOpened") if isinstance(raw, dict) else None
    if (
        raw is None
        or str(raw.get("type") or "").upper() != "ORDER_FILL"
        or not isinstance(opened, dict)
    ):
        return None
    broker_entry_ts_utc = str(raw.get("time") or "").strip()
    broker_pair = str(raw.get("instrument") or "").strip().upper()
    broker_order_id = _identifier_text(raw.get("orderID"))
    broker_trade_id = _identifier_text(opened.get("tradeID"))
    broker_order_units = _finite_float(raw.get("units"))
    broker_open_units = _finite_float(opened.get("units"))
    normalized_units = _finite_float(fill.get("units"))
    normalized_pair = str(fill.get("pair") or "").strip().upper()
    normalized_side = str(fill.get("side") or "").strip().upper()
    normalized_order_id = _identifier_text(fill.get("order_id"))
    normalized_trade_id = _identifier_text(fill.get("trade_id"))
    raw_reason = str(raw.get("reason") or "").strip().upper()
    normalized_reason = str(fill.get("exit_reason") or "").strip().upper()
    vehicle = _entry_vehicle_from_event(
        {
            "exit_reason": raw_reason,
            "raw_json": fill.get("raw_json"),
        }
    )
    if (
        not _broker_timestamp_matches(
            normalized=fill.get("ts_utc"),
            broker=broker_entry_ts_utc,
        )
        or not broker_pair
        or broker_pair != normalized_pair
        or not broker_order_id
        or broker_order_id != normalized_order_id
        or not broker_trade_id
        or broker_trade_id != normalized_trade_id
        or broker_order_units is None
        or broker_open_units is None
        or normalized_units is None
        or broker_order_units == 0.0
        or broker_open_units == 0.0
        or normalized_units == 0.0
        or (broker_order_units > 0.0) != (broker_open_units > 0.0)
        or abs(broker_order_units) < abs(broker_open_units)
        or abs(normalized_units) != abs(broker_open_units)
        or normalized_side != ("LONG" if broker_open_units > 0.0 else "SHORT")
        or not raw_reason
        or raw_reason != normalized_reason
        or vehicle not in {"LIMIT", "MARKET", "STOP"}
    ):
        return None
    return {
        "broker_entry_ts_utc": broker_entry_ts_utc,
        "broker_time_consistent": True,
        "entry_units": broker_open_units,
        "vehicle": vehicle,
    }


def _fill_has_system_attribution_candidate(
    fill: dict[str, Any],
    *,
    gateways_by_order: dict[str, list[dict[str, Any]]],
) -> bool:
    """Detect a broken fill that could otherwise vanish from a system cohort."""

    fill_raw = _optional_json_object(fill.get("raw_json"))
    own_lane = str(fill.get("lane_id") or "").strip()
    if own_lane and not _manual_lane(own_lane):
        return True
    fill_rowid = int(fill.get("ledger_rowid") or 0)
    order_ids = {
        order_id
        for order_id in (
            _identifier_text(fill.get("order_id")),
            _identifier_text(fill_raw.get("orderID"))
            if isinstance(fill_raw, dict)
            else "",
        )
        if order_id
    }
    for order_id in order_ids:
        for gateway in gateways_by_order.get(order_id, []):
            if int(gateway.get("ledger_rowid") or 0) >= fill_rowid:
                continue
            lane_id = str(gateway.get("lane_id") or "").strip()
            if (
                lane_id
                and not _manual_lane(lane_id)
                and not _manual_owner_marked(
                    _optional_json_object(gateway.get("raw_json"))
                )
            ):
                return True
    return False


def _resolve_system_entry(
    *,
    fills: list[dict[str, Any]],
    gateways_by_order: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    if not fills:
        return None
    attributed_lanes: set[str] = set()
    matched_gateways: dict[int, dict[str, Any]] = {}
    audited_fills: list[dict[str, Any] | None] = []
    for fill in fills:
        fill_rowid = int(fill.get("ledger_rowid") or 0)
        order_id = str(fill.get("order_id") or "").strip()
        all_prior_gateways = [
            gateway
            for gateway in gateways_by_order.get(order_id, [])
            if int(gateway.get("ledger_rowid") or 0) < fill_rowid
        ]
        prior_gateways = [
            gateway
            for gateway in all_prior_gateways
            if str(gateway.get("lane_id") or "").strip()
        ]
        own_lane = str(fill.get("lane_id") or "").strip()
        if own_lane:
            lane = own_lane
        else:
            gateway_lanes = {
                str(gateway.get("lane_id") or "").strip()
                for gateway in prior_gateways
            }
            if len(gateway_lanes) != 1:
                return None
            lane = next(iter(gateway_lanes))
        if _manual_lane(lane):
            return None
        attributed_lanes.add(lane)
        for gateway in all_prior_gateways:
            matched_gateways[int(gateway.get("ledger_rowid") or 0)] = gateway
        fill_raw = _optional_json_object(fill.get("raw_json"))
        if _manual_owner_marked(fill_raw):
            return None
        audited_fills.append(_audit_entry_fill_truth(fill))
    if len(attributed_lanes) != 1:
        return None
    lane_id = next(iter(attributed_lanes))
    gateway_rows = list(matched_gateways.values())
    for gateway in gateway_rows:
        if _manual_lane(str(gateway.get("lane_id") or "")):
            return None
        if _manual_owner_marked(_optional_json_object(gateway.get("raw_json"))):
            return None

    pair_truth = {
        str(row.get("pair") or "").strip().upper()
        for row in (*fills, *gateway_rows)
        if str(row.get("pair") or "").strip()
    }
    side_truth = {
        str(row.get("side") or "").strip().upper()
        for row in (*fills, *gateway_rows)
        if str(row.get("side") or "").strip()
    }
    fill_vehicles = {
        vehicle
        for vehicle in (_entry_vehicle_from_event(row) for row in fills)
        if vehicle != "UNKNOWN"
    }
    gateway_vehicles = {
        vehicle
        for vehicle in (_entry_vehicle_from_event(row) for row in gateway_rows)
        if vehicle != "UNKNOWN"
    }
    pair = next(iter(pair_truth)) if len(pair_truth) == 1 else "UNKNOWN"
    side = next(iter(side_truth)) if len(side_truth) == 1 else "UNKNOWN"
    vehicle = next(iter(fill_vehicles)) if len(fill_vehicles) == 1 else "UNKNOWN"
    lane_pair, lane_side, method, lane_vehicle = _lane_parts(lane_id)
    entry_times = [_parse_utc_instant(fill.get("ts_utc")) for fill in fills]
    broker_entry_times = [
        str(audit.get("broker_entry_ts_utc") or "")
        for audit in audited_fills
        if isinstance(audit, dict)
    ]
    gateway_lane_consistent = all(
        not str(row.get("lane_id") or "").strip()
        or _lane_ids_same_entry_shape(
            lane_id,
            str(row.get("lane_id") or "").strip(),
            resolved_vehicle=vehicle,
        )
        for row in gateway_rows
    )
    truth_consistent = (
        pair != "UNKNOWN"
        and side in {"LONG", "SHORT"}
        and method != "UNKNOWN"
        and vehicle != "UNKNOWN"
        and lane_pair == pair
        and lane_side == side
        and lane_vehicle in {"UNKNOWN", vehicle}
        and (not gateway_vehicles or gateway_vehicles == {vehicle})
        and gateway_lane_consistent
        and all(entry_time is not None for entry_time in entry_times)
        and len(audited_fills) == len(fills)
        and all(isinstance(audit, dict) for audit in audited_fills)
    )
    broker_entry_ts_utc = (
        min(
            broker_entry_times,
            key=lambda value: _rfc3339_utc_key(value)
            or (datetime.max.replace(tzinfo=timezone.utc).isoformat(), 0),
        )
        if broker_entry_times
        else ""
    )
    entry_units = (
        float(audited_fills[0]["entry_units"])
        if len(audited_fills) == 1 and isinstance(audited_fills[0], dict)
        else 0.0
    )
    return {
        "pair": pair if pair != "UNKNOWN" else lane_pair,
        "side": side if side != "UNKNOWN" else lane_side,
        "lane_id": lane_id,
        "method": method if method != "UNKNOWN" else _lane_method(lane_id),
        "vehicle": vehicle,
        "truth_consistent": truth_consistent,
        "entry_units": entry_units,
        "entry_ts_utc": (
            min(entry_time for entry_time in entry_times if entry_time is not None).isoformat()
            if any(entry_time is not None for entry_time in entry_times)
            else ""
        ),
        "broker_entry_ts_utc": broker_entry_ts_utc,
        "broker_time_consistent": bool(truth_consistent and broker_entry_ts_utc),
    }


def _lane_ids_same_entry_shape(
    fill_lane_id: str,
    gateway_lane_id: str,
    *,
    resolved_vehicle: str,
) -> bool:
    """Match parent/vehicle lane spellings without weakening lane identity.

    The gateway records the selected vehicle (for example ``:MARKET``), while
    OANDA's accepted/fill rows may retain the four-part parent lane.  Pair,
    side, and method must still match exactly, and any vehicle explicitly
    present on either side must match the broker-resolved entry vehicle.
    """

    fill_pair, fill_side, fill_method, fill_vehicle = _lane_parts(fill_lane_id)
    gateway_pair, gateway_side, gateway_method, gateway_vehicle = _lane_parts(
        gateway_lane_id
    )
    if (
        "UNKNOWN" in {fill_pair, fill_side, fill_method}
        or "UNKNOWN" in {gateway_pair, gateway_side, gateway_method}
        or (fill_pair, fill_side, fill_method)
        != (gateway_pair, gateway_side, gateway_method)
    ):
        return False
    vehicle = str(resolved_vehicle or "UNKNOWN").upper()
    return vehicle != "UNKNOWN" and all(
        lane_vehicle in {"UNKNOWN", vehicle}
        for lane_vehicle in (fill_vehicle, gateway_vehicle)
    )


@dataclass(frozen=True)
class CaptureEconomicsSummary:
    output_path: Path
    report_path: Path
    status: str
    trades: int
    win_rate: float | None
    payoff_ratio: float | None
    breakeven_payoff: float | None
    expectancy_jpy: float | None


def _bucket_metrics(rows: list[RealizedOutcome]) -> dict[str, Any]:
    wins = [r.realized_pl_jpy for r in rows if r.realized_pl_jpy > 0]
    losses = [r.realized_pl_jpy for r in rows if r.realized_pl_jpy < 0]
    n = len(wins) + len(losses)
    if n == 0:
        return {"trades": 0}
    p = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else None
    breakeven = ((1.0 - p) / p) if p > 0 else None
    expectancy = (sum(wins) + sum(losses)) / n
    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(p, 4),
        "avg_win_jpy": round(avg_win, 1),
        "avg_loss_jpy": round(avg_loss, 1),
        "payoff_ratio": round(payoff, 3) if payoff is not None else None,
        "breakeven_payoff_at_win_rate": round(breakeven, 3) if breakeven is not None else None,
        "expectancy_jpy_per_trade": round(expectancy, 1),
        "net_jpy": round(sum(wins) + sum(losses), 1),
    }


def _nested_segment_metrics(
    rows: list[RealizedOutcome],
    dimensions: tuple[str, ...],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups: dict[tuple[str, ...], list[RealizedOutcome]] = {}
    for row in rows:
        key = tuple(str(getattr(row, dimension) or "UNKNOWN") for dimension in dimensions)
        groups.setdefault(key, []).append(row)
    for key, bucket in sorted(groups.items()):
        cursor = out
        for part in key[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[key[-1]] = _bucket_metrics(bucket)
    return out


def _lane_method(lane_id: str) -> str:
    parts = [part for part in str(lane_id or "").split(":") if part]
    if len(parts) >= 4:
        return parts[3]
    return "UNKNOWN"


def _iso_week(ts_utc: str) -> str:
    parsed = _parse_utc_instant(ts_utc)
    if parsed is None:
        return "unknown"
    year, week, _ = parsed.date().isocalendar()
    return f"{year}-W{week:02d}"


def _outcome_timestamp(row: RealizedOutcome) -> datetime | None:
    return _parse_utc_instant(row.ts_utc)


def _recent_performance_comparison(
    rows: list[RealizedOutcome],
    *,
    as_of: datetime,
) -> dict[str, Any]:
    end = as_of.astimezone(timezone.utc)
    recent_start = end - timedelta(days=RECENT_PERFORMANCE_WINDOW_DAYS)
    prior_start = recent_start - timedelta(days=RECENT_PERFORMANCE_WINDOW_DAYS)
    parsed_timestamps = [(row, _outcome_timestamp(row)) for row in rows]
    timestamp_parse_failures = sum(
        1 for _row, timestamp in parsed_timestamps if timestamp is None
    )
    timestamped = [
        (row, timestamp)
        for row, timestamp in parsed_timestamps
        if timestamp is not None
    ]
    recent_rows = [row for row, timestamp in timestamped if recent_start <= timestamp <= end]
    prior_rows = [row for row, timestamp in timestamped if prior_start <= timestamp < recent_start]
    historical_rows = [row for row, timestamp in timestamped if timestamp < recent_start]
    recent = _bucket_metrics(recent_rows)
    prior = _bucket_metrics(prior_rows)
    historical = _bucket_metrics(historical_rows)
    recent_trades = int(recent.get("trades") or 0)
    historical_trades = int(historical.get("trades") or 0)
    recent_expectancy = _optional_float(recent.get("expectancy_jpy_per_trade"))
    recent_net = _optional_float(recent.get("net_jpy"))
    historical_expectancy = _optional_float(
        historical.get("expectancy_jpy_per_trade")
    )
    timestamps_valid = timestamp_parse_failures == 0
    recent_sample_sufficient = (
        timestamps_valid and recent_trades >= MIN_SAMPLE_FOR_VERDICT
    )
    baseline_sample_sufficient = (
        timestamps_valid and historical_trades >= MIN_SAMPLE_FOR_VERDICT
    )
    positive = bool(
        timestamps_valid
        and recent_trades
        and recent_expectancy is not None
        and recent_expectancy > 0.0
        and recent_net is not None
        and recent_net > 0.0
    )
    if timestamp_parse_failures > 0:
        verdict = "TIMESTAMP_PARSE_FAILED"
    elif recent_trades == 0:
        verdict = "NO_RECENT_TRADES"
    elif not recent_sample_sufficient:
        verdict = "RECENT_POSITIVE_LOW_SAMPLE" if positive else "RECENT_NON_POSITIVE_LOW_SAMPLE"
    elif positive and not baseline_sample_sufficient:
        verdict = "POSITIVE_BASELINE_INSUFFICIENT"
    elif positive and historical_expectancy is not None and recent_expectancy > historical_expectancy:
        verdict = "OBSERVED_IMPROVEMENT_MIN_SAMPLE"
    elif positive:
        verdict = "POSITIVE_NOT_IMPROVED"
    else:
        verdict = "NOT_IMPROVED"
    return {
        "window_days": RECENT_PERFORMANCE_WINDOW_DAYS,
        "as_of_utc": end.isoformat(),
        "timestamp_parse_status": (
            "EMPTY"
            if not rows
            else "INVALID"
            if timestamp_parse_failures > 0
            else "VALID"
        ),
        "timestamp_input_rows": len(rows),
        "timestamp_parsed_rows": len(timestamped),
        "timestamp_parse_failures": timestamp_parse_failures,
        "recent_window": {
            "start_utc": recent_start.isoformat(),
            "end_utc": end.isoformat(),
            **recent,
        },
        "prior_window": {
            "start_utc": prior_start.isoformat(),
            "end_utc": recent_start.isoformat(),
            **prior,
        },
        "historical_baseline": {
            "end_exclusive_utc": recent_start.isoformat(),
            **historical,
        },
        "expectancy_delta_jpy_per_trade": (
            round(recent_expectancy - historical_expectancy, 1)
            if recent_expectancy is not None and historical_expectancy is not None
            else None
        ),
        "verdict": verdict,
        "recent_sample_sufficient": recent_sample_sufficient,
        "baseline_sample_sufficient": baseline_sample_sufficient,
        "observed_improvement": verdict == "OBSERVED_IMPROVEMENT_MIN_SAMPLE",
        # Raw JPY point estimates cannot distinguish better entry/exit edge
        # from a lot-size or NAV shift. Statistical and normalized R (or
        # JPY/1,000u) lower-bound evidence must be added before this can be
        # called proof.
        "improvement_proven": False,
        "proof_status": (
            "TIMESTAMP_PARSE_FAILED"
            if timestamp_parse_failures > 0
            else "NOT_AVAILABLE_RAW_JPY_POINT_ESTIMATE_ONLY"
        ),
        "normalized_edge_proof_available": False,
        "sample_gap": (
            MIN_SAMPLE_FOR_VERDICT
            if not timestamps_valid
            else max(0, MIN_SAMPLE_FOR_VERDICT - recent_trades)
        ),
        "baseline_sample_gap": (
            MIN_SAMPLE_FOR_VERDICT
            if not timestamps_valid
            else max(0, MIN_SAMPLE_FOR_VERDICT - historical_trades)
        ),
        "lifetime_status_unchanged": True,
    }


def _negative_exit_rows(by_exit: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for reason, metrics in by_exit.items():
        if not isinstance(metrics, dict):
            continue
        net = _optional_float(metrics.get("net_jpy"))
        trades = int(metrics.get("trades") or 0)
        if trades <= 0 or net is None or net >= 0:
            continue
        rows.append((reason, metrics))
    return sorted(rows, key=lambda item: float(item[1].get("net_jpy") or 0.0))


def _positive_exit_rows(by_exit: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for reason, metrics in by_exit.items():
        if not isinstance(metrics, dict):
            continue
        net = _optional_float(metrics.get("net_jpy"))
        trades = int(metrics.get("trades") or 0)
        if trades <= 0 or net is None or net <= 0:
            continue
        rows.append((reason, metrics))
    return sorted(rows, key=lambda item: float(item[1].get("net_jpy") or 0.0), reverse=True)


def _capture_repair_summary(
    *,
    status: str,
    overall: dict[str, Any],
    by_exit: dict[str, Any],
) -> dict[str, Any]:
    negative = _negative_exit_rows(by_exit)
    positive = _positive_exit_rows(by_exit)
    payoff = _optional_float(overall.get("payoff_ratio"))
    breakeven = _optional_float(overall.get("breakeven_payoff_at_win_rate"))
    summary: dict[str, Any] = {
        "status": status,
        "payoff_gap_to_breakeven": (
            round(max(0.0, breakeven - payoff), 3)
            if payoff is not None and breakeven is not None
            else None
        ),
        "top_negative_exit_reasons": [
            {
                "exit_reason": reason,
                "trades": int(metrics.get("trades") or 0),
                "net_jpy": metrics.get("net_jpy"),
                "expectancy_jpy_per_trade": metrics.get("expectancy_jpy_per_trade"),
                "win_rate": metrics.get("win_rate"),
                "payoff_ratio": metrics.get("payoff_ratio"),
            }
            for reason, metrics in negative[:EXIT_REPAIR_ITEM_LIMIT]
        ],
        "top_positive_exit_reasons": [
            {
                "exit_reason": reason,
                "trades": int(metrics.get("trades") or 0),
                "net_jpy": metrics.get("net_jpy"),
                "expectancy_jpy_per_trade": metrics.get("expectancy_jpy_per_trade"),
                "win_rate": metrics.get("win_rate"),
                "payoff_ratio": metrics.get("payoff_ratio"),
            }
            for reason, metrics in positive[:EXIT_REPAIR_ITEM_LIMIT]
        ],
    }
    if negative:
        reason, metrics = negative[0]
        summary["dominant_loss_exit_reason"] = reason
        summary["dominant_loss_exit_net_jpy"] = metrics.get("net_jpy")
        summary["dominant_loss_exit_expectancy_jpy_per_trade"] = metrics.get(
            "expectancy_jpy_per_trade"
        )
    if positive:
        reason, metrics = positive[0]
        summary["strongest_positive_exit_reason"] = reason
        summary["strongest_positive_exit_net_jpy"] = metrics.get("net_jpy")
    return summary


def _segment_repair_priorities(rows: list[RealizedOutcome]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[RealizedOutcome]] = {}
    for row in rows:
        key = (row.pair or "UNKNOWN", row.side or "UNKNOWN", row.method or "UNKNOWN")
        groups.setdefault(key, []).append(row)

    items: list[dict[str, Any]] = []
    for (pair, side, method), bucket in groups.items():
        overall = _bucket_metrics(bucket)
        tp_metrics = _bucket_metrics(
            [row for row in bucket if row.exit_reason == TAKE_PROFIT_EXIT_REASON]
        )
        market_close_metrics = _bucket_metrics(
            [row for row in bucket if row.exit_reason == MARKET_CLOSE_EXIT_REASON]
        )
        market_close_loss_rows = [
            row
            for row in bucket
            if row.exit_reason == MARKET_CLOSE_EXIT_REASON and row.realized_pl_jpy < 0
        ]

        tp_trades = int(tp_metrics.get("trades") or 0)
        tp_losses = int(tp_metrics.get("losses") or 0)
        tp_expectancy = _optional_float(tp_metrics.get("expectancy_jpy_per_trade"))
        tp_avg_win = _optional_float(tp_metrics.get("avg_win_jpy"))
        tp_proven = (
            tp_trades >= SCOPED_TP_PROOF_MIN_EXIT_TRADES
            and tp_expectancy is not None
            and tp_expectancy > 0
            and tp_avg_win is not None
            and tp_avg_win > 0
            and tp_losses <= 0
        )
        tp_positive_thin = (
            tp_trades > 0
            and not tp_proven
            and tp_expectancy is not None
            and tp_expectancy > 0
            and tp_avg_win is not None
            and tp_avg_win > 0
            and tp_losses <= 0
        )
        proof_gap = max(0, SCOPED_TP_PROOF_MIN_EXIT_TRADES - tp_trades)

        segment_net = _optional_float(overall.get("net_jpy"))
        market_close_net = _optional_float(market_close_metrics.get("net_jpy"))
        market_close_negative = market_close_net is not None and market_close_net < 0
        segment_negative = segment_net is not None and segment_net < 0

        if tp_proven and market_close_negative:
            priority_class = "PRESERVE_TP_PROVEN_REPAIR_MARKET_CLOSE_LEAK"
            rank = 0
            next_action = (
                "preserve attached-TP HARVEST entries for this exact shape, but repair "
                "or avoid its MARKET_ORDER_TRADE_CLOSE path before increasing exposure"
            )
        elif market_close_negative and tp_positive_thin:
            priority_class = "COLLECT_TP_PROOF_REPAIR_MARKET_CLOSE_LEAK"
            rank = 1
            next_action = (
                "collect more scoped broker-TP outcomes and repair MARKET_ORDER_TRADE_CLOSE "
                "leakage before treating this as high-rotation proof"
            )
        elif market_close_negative:
            priority_class = "REPAIR_MARKET_CLOSE_LEAK"
            rank = 2
            next_action = (
                "rank the close provenance for this segment; do not widen fresh risk until "
                "loss-side MARKET_ORDER_TRADE_CLOSE evidence is repaired or explicitly justified"
            )
        elif tp_proven:
            priority_class = "PRESERVE_TP_PROVEN_SHAPE"
            rank = 3
            next_action = (
                "preserve this attached-TP capture shape while keeping forecast, spread, "
                "strategy-profile, margin, and gateway checks active"
            )
        elif tp_positive_thin:
            priority_class = "COLLECT_SCOPED_TP_PROOF"
            rank = 4
            next_action = (
                "treat as evidence-collection candidate: positive broker-TP outcomes exist "
                "but the scoped sample is below the proof floor"
            )
        elif segment_negative:
            priority_class = "AVOID_OR_REPRICE_SEGMENT"
            rank = 5
            next_action = (
                "avoid or reprice this segment until entry/exit geometry produces positive "
                "realized expectancy"
            )
        else:
            priority_class = "MONITOR_LOW_SAMPLE" if int(overall.get("trades") or 0) < MIN_SAMPLE_FOR_VERDICT else "MONITOR"
            rank = 6
            next_action = "monitor; no realized TP proof or market-close repair priority dominates yet"

        items.append(
            {
                "evidence_ref": f"capture:segment:{pair}:{side}:{method}",
                "attribution_scope": SYSTEM_ATTRIBUTION_SCOPE,
                "operator_manual_excluded": True,
                "should_count_against_system_edge": True,
                "pair": pair,
                "side": side,
                "method": method,
                "priority_class": priority_class,
                "next_action": next_action,
                "trades": int(overall.get("trades") or 0),
                "wins": int(overall.get("wins") or 0),
                "losses": int(overall.get("losses") or 0),
                "win_rate": overall.get("win_rate"),
                "expectancy_jpy_per_trade": overall.get("expectancy_jpy_per_trade"),
                "net_jpy": overall.get("net_jpy"),
                "take_profit_trades": tp_trades,
                "take_profit_wins": int(tp_metrics.get("wins") or 0),
                "take_profit_losses": tp_losses,
                "take_profit_expectancy_jpy": tp_metrics.get("expectancy_jpy_per_trade"),
                "take_profit_net_jpy": tp_metrics.get("net_jpy"),
                "take_profit_proof_floor": SCOPED_TP_PROOF_MIN_EXIT_TRADES,
                "take_profit_proof_gap_trades": proof_gap,
                "take_profit_proven": tp_proven,
                "market_close_trades": int(market_close_metrics.get("trades") or 0),
                "market_close_losses": int(market_close_metrics.get("losses") or 0),
                "market_close_loss_net_jpy": round(
                    sum(row.realized_pl_jpy for row in market_close_loss_rows),
                    1,
                ),
                "market_close_expectancy_jpy": market_close_metrics.get(
                    "expectancy_jpy_per_trade"
                ),
                "market_close_net_jpy": market_close_metrics.get("net_jpy"),
                "market_close_loss_trade_ids": [
                    row.trade_id
                    for row in sorted(
                        market_close_loss_rows,
                        key=lambda item: (item.realized_pl_jpy, item.ts_utc),
                    )[:MARKET_CLOSE_LOSS_EXAMPLE_LIMIT]
                ],
                "market_close_loss_examples": _market_close_loss_examples(
                    market_close_loss_rows
                ),
                "_sort_rank": rank,
            }
        )

    def _sort_key(item: dict[str, Any]) -> tuple[float, float, float, int, str, str, str]:
        market_close_net = _optional_float(item.get("market_close_net_jpy"))
        segment_net = _optional_float(item.get("net_jpy"))
        proof_gap = int(item.get("take_profit_proof_gap_trades") or 0)
        return (
            float(item.get("_sort_rank") or 0),
            market_close_net if market_close_net is not None else 0.0,
            segment_net if segment_net is not None else 0.0,
            proof_gap,
            str(item.get("pair") or ""),
            str(item.get("side") or ""),
            str(item.get("method") or ""),
        )

    sorted_items = sorted(items, key=_sort_key)
    for item in sorted_items:
        item.pop("_sort_rank", None)
    return {
        "basis": "trader-attributed realized outcomes grouped by pair|side|method",
        "take_profit_exit_reason": TAKE_PROFIT_EXIT_REASON,
        "market_close_exit_reason": MARKET_CLOSE_EXIT_REASON,
        "scoped_tp_proof_min_exit_trades": SCOPED_TP_PROOF_MIN_EXIT_TRADES,
        "total_segments": len(sorted_items),
        "items": sorted_items[:SEGMENT_REPAIR_PRIORITY_LIMIT],
    }


def _capture_action_items(
    *,
    status: str,
    overall: dict[str, Any],
    by_exit: dict[str, Any],
    repair_summary: dict[str, Any],
) -> list[str]:
    if status == "EVIDENCE_UNREADABLE":
        return [
            "repair execution-ledger financing attribution before generating or sending fresh entries"
        ]
    if status == "LOW_SAMPLE":
        return ["collect more trader-attributed realized exits before changing exit policy"]
    items: list[str] = []
    payoff = _optional_float(overall.get("payoff_ratio"))
    breakeven = _optional_float(overall.get("breakeven_payoff_at_win_rate"))
    if status == "NEGATIVE_EXPECTANCY":
        if payoff is not None and breakeven is not None:
            items.append(
                "repair exit payoff asymmetry before treating the daily target as arithmetically reachable: "
                f"payoff_ratio={payoff:.3f} breakeven={breakeven:.3f}"
            )
        dominant_reason = str(repair_summary.get("dominant_loss_exit_reason") or "")
        dominant_net = _optional_float(repair_summary.get("dominant_loss_exit_net_jpy"))
        if dominant_reason:
            net_text = f"{dominant_net:.1f} JPY" if dominant_net is not None else "net loss"
            if dominant_reason == "MARKET_ORDER_TRADE_CLOSE":
                items.append(
                    "contain MARKET_ORDER_TRADE_CLOSE drag "
                    f"({net_text}): prefer attached TP, TP-rebalance, profit-side TAKE_PROFIT_MARKET, "
                    "and require hard Gate A/B evidence for loss-side CLOSE"
                )
            else:
                items.append(f"repair dominant negative exit bucket {dominant_reason} ({net_text})")
    strongest_positive = str(repair_summary.get("strongest_positive_exit_reason") or "")
    if strongest_positive:
        items.append(
            f"preserve profitable {strongest_positive} behavior while repairing negative exit buckets"
        )
    return items[:EXIT_REPAIR_ITEM_LIMIT]


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _market_close_loss_examples(rows: list[RealizedOutcome]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (item.realized_pl_jpy, item.ts_utc))[
        :MARKET_CLOSE_LOSS_EXAMPLE_LIMIT
    ]:
        examples.append(
            {
                "trade_id": row.trade_id,
                "ts_utc": row.ts_utc,
                "lane_id": row.lane_id,
                "pair": row.pair,
                "side": row.side,
                "method": row.method,
                "exit_reason": row.exit_reason,
                "realized_pl_jpy": round(row.realized_pl_jpy, 4),
                "close_family": "SYSTEM_GATEWAY_MARKET_CLOSE",
                "attribution_scope": SYSTEM_ATTRIBUTION_SCOPE,
                "operator_manual_excluded": True,
                "should_count_against_system_edge": True,
            }
        )
    return examples


def build_capture_economics(
    *,
    ledger_path: Path,
    output_path: Path = DEFAULT_CAPTURE_ECONOMICS,
    report_path: Path = DEFAULT_CAPTURE_ECONOMICS_REPORT,
    now: datetime | None = None,
) -> CaptureEconomicsSummary:
    raw_rows = read_attributed_net_outcomes(ledger_path) if ledger_path.exists() else []
    evidence_read_failed = raw_rows is None and ledger_path.exists()
    rows = raw_rows or []

    overall = _bucket_metrics(rows)
    by_exit: dict[str, Any] = {}
    by_week: dict[str, Any] = {}
    for reason in sorted({r.exit_reason for r in rows}):
        by_exit[reason] = _bucket_metrics([r for r in rows if r.exit_reason == reason])
    for week in sorted({_iso_week(r.ts_utc) for r in rows}):
        by_week[week] = _bucket_metrics([r for r in rows if _iso_week(r.ts_utc) == week])
    by_pair_side_exit = _nested_segment_metrics(rows, ("pair", "side", "exit_reason"))
    by_pair_side_method_exit = _nested_segment_metrics(rows, ("pair", "side", "method", "exit_reason"))

    trades = int(overall.get("trades") or 0)
    payoff = overall.get("payoff_ratio")
    breakeven = overall.get("breakeven_payoff_at_win_rate")
    expectancy = overall.get("expectancy_jpy_per_trade")
    if evidence_read_failed:
        status = "EVIDENCE_UNREADABLE"
    elif trades < MIN_SAMPLE_FOR_VERDICT:
        status = "LOW_SAMPLE"
    elif payoff is not None and breakeven is not None and payoff >= breakeven:
        status = "POSITIVE_EXPECTANCY"
    elif payoff is None and expectancy is not None and expectancy > 0:
        # Zero losses in the sample: payoff is undefined (division by the
        # empty loss side) but the expectancy is unambiguously positive.
        status = "POSITIVE_EXPECTANCY"
    else:
        status = "NEGATIVE_EXPECTANCY"
    repair_summary = _capture_repair_summary(status=status, overall=overall, by_exit=by_exit)
    segment_repair_priorities = _segment_repair_priorities(rows)
    action_items = _capture_action_items(
        status=status,
        overall=overall,
        by_exit=by_exit,
        repair_summary=repair_summary,
    )

    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    clock = clock.astimezone(timezone.utc)
    recent_performance = _recent_performance_comparison(rows, as_of=clock)
    generated_at = clock.isoformat()
    payload = {
        "generated_at_utc": generated_at,
        "status": status,
        "evidence_read_status": "FAILED" if evidence_read_failed else "READABLE",
        "min_sample_for_verdict": MIN_SAMPLE_FOR_VERDICT,
        "overall": overall,
        "by_exit_reason": by_exit,
        "by_pair_side_exit_reason": by_pair_side_exit,
        "by_pair_side_method_exit_reason": by_pair_side_method_exit,
        "by_iso_week": by_week,
        "recent_performance": recent_performance,
        "repair_summary": repair_summary,
        "segment_repair_priorities": segment_repair_priorities,
        "action_items": action_items,
        "note": (
            "Advisory audit (AGENT_CONTRACT §8): payoff_ratio must reach "
            "breakeven_payoff_at_win_rate before the daily 5% floor has an "
            "arithmetic route. When status is NEGATIVE_EXPECTANCY and avg_loss_jpy "
            "exceeds avg_win_jpy, generate-intents caps fresh NEW-entry loss at "
            "the observed average winner until payoff repair clears."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Capture Economics Report",
        "",
        f"- Generated at UTC: `{generated_at}`",
        f"- Status: `{status}`",
        f"- Trades (trader-attributed, realized): `{trades}`",
    ]
    if trades:
        lines += [
            f"- Win rate: `{(overall.get('win_rate') or 0) * 100:.1f}%`",
            f"- Avg win / avg loss: `{overall.get('avg_win_jpy')}` / `{overall.get('avg_loss_jpy')}` JPY",
            f"- Payoff ratio: `{overall.get('payoff_ratio')}` (breakeven at win rate: `{overall.get('breakeven_payoff_at_win_rate')}`)",
            f"- Expectancy: `{overall.get('expectancy_jpy_per_trade')}` JPY/trade, net `{overall.get('net_jpy')}` JPY",
            "",
            "## Recent Performance (Does Not Replace Lifetime Safety Status)",
            "",
            f"- Timestamp parse: `{recent_performance.get('timestamp_parse_status')}`; "
            f"parsed `{recent_performance.get('timestamp_parsed_rows')}` / "
            f"`{recent_performance.get('timestamp_input_rows')}`, failures "
            f"`{recent_performance.get('timestamp_parse_failures')}`",
            f"- Verdict: `{recent_performance.get('verdict')}`",
            f"- Recent {RECENT_PERFORMANCE_WINDOW_DAYS}d: "
            f"trades `{recent_performance['recent_window'].get('trades', 0)}`, "
            f"expectancy `{recent_performance['recent_window'].get('expectancy_jpy_per_trade')}` JPY, "
            f"net `{recent_performance['recent_window'].get('net_jpy')}` JPY",
            f"- Prior {RECENT_PERFORMANCE_WINDOW_DAYS}d: "
            f"trades `{recent_performance['prior_window'].get('trades', 0)}`, "
            f"expectancy `{recent_performance['prior_window'].get('expectancy_jpy_per_trade')}` JPY, "
            f"net `{recent_performance['prior_window'].get('net_jpy')}` JPY",
            f"- Historical baseline before recent window: "
            f"trades `{recent_performance['historical_baseline'].get('trades', 0)}`, "
            f"expectancy `{recent_performance['historical_baseline'].get('expectancy_jpy_per_trade')}` JPY; "
            f"delta `{recent_performance.get('expectancy_delta_jpy_per_trade')}` JPY/trade",
            f"- Improvement proven: `{recent_performance.get('improvement_proven')}`; "
            f"recent sample gap `{recent_performance.get('sample_gap')}`, "
            f"baseline sample gap `{recent_performance.get('baseline_sample_gap')}`",
            f"- Proof status: `{recent_performance.get('proof_status')}`; raw JPY point estimates are observation only until normalized edge and statistical lower bounds exist.",
            "",
            "## Repair Summary",
            "",
            f"- Dominant loss exit: `{repair_summary.get('dominant_loss_exit_reason') or 'none'}` "
            f"net `{repair_summary.get('dominant_loss_exit_net_jpy')}` JPY",
            f"- Strongest positive exit: `{repair_summary.get('strongest_positive_exit_reason') or 'none'}` "
            f"net `{repair_summary.get('strongest_positive_exit_net_jpy')}` JPY",
            f"- Payoff gap to breakeven: `{repair_summary.get('payoff_gap_to_breakeven')}`",
            "",
            "## Segment Repair Priorities",
            "",
            "| pair | side | method | priority | n | TP n/gap | market-close net | net |",
            "|---|---|---|---|---|---|---|---|",
            *[
                (
                    f"| `{item.get('pair')}` | `{item.get('side')}` | `{item.get('method')}` "
                    f"| `{item.get('priority_class')}` | {item.get('trades')} "
                    f"| {item.get('take_profit_trades')}/{item.get('take_profit_proof_gap_trades')} "
                    f"| {item.get('market_close_net_jpy')} | {item.get('net_jpy')} |"
                )
                for item in segment_repair_priorities.get("items", [])
            ],
            "",
            "## Action Items",
            "",
            *[f"- {item}" for item in action_items],
            "",
            "## By exit reason",
            "",
            "| exit_reason | n | win% | avg win | avg loss | net |",
            "|---|---|---|---|---|---|",
        ]
        for reason, m in by_exit.items():
            if not m.get("trades"):
                continue
            lines.append(
                f"| `{reason}` | {m['trades']} | {(m.get('win_rate') or 0) * 100:.0f}% "
                f"| {m.get('avg_win_jpy')} | {m.get('avg_loss_jpy')} | {m.get('net_jpy')} |"
            )
        lines += ["", "## By ISO week", "", "| week | n | win% | payoff | net |", "|---|---|---|---|---|"]
        for week, m in by_week.items():
            if not m.get("trades"):
                continue
            lines.append(
                f"| `{week}` | {m['trades']} | {(m.get('win_rate') or 0) * 100:.0f}% "
                f"| {m.get('payoff_ratio')} | {m.get('net_jpy')} |"
            )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")

    return CaptureEconomicsSummary(
        output_path=output_path,
        report_path=report_path,
        status=status,
        trades=trades,
        win_rate=overall.get("win_rate"),
        payoff_ratio=payoff,
        breakeven_payoff=breakeven,
        expectancy_jpy=overall.get("expectancy_jpy_per_trade"),
    )
