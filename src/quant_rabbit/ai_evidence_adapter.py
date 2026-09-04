"""Build a compact, sealed, point-in-time packet for the AI decision worker.

The adapter is deliberately read-only with respect to market and broker inputs.
It copies only allowlisted observations, never imports deterministic order
intents, and classifies non-system exposure as ``NO_TOUCH``.  Consumers still
have to enforce the ordinary decision, risk, ownership, and gateway contracts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTRACT = "QR_AI_EVIDENCE_PACKET_V1"

# 256 KiB is the product contract for one model input, not a market threshold.
# Replace only if the AI runtime's accepted packet contract changes.
MAX_PACKET_BYTES = 256 * 1024

# Five seconds accommodates ordinary local clock/read ordering without allowing
# a future market observation. Replace with measured host clock error if needed.
MAX_FUTURE_SKEW_SECONDS = 5

# The live G8 universe has 28 FX pairs. Four spare rows tolerate a reviewed
# universe extension; replace this with runtime universe metadata when exposed.
MAX_PAIRS = 32

# Four recent pivots retain the two latest high/low rail turns while bounding
# prompt size. Replace with a horizon-derived rail selector if the model needs it.
MAX_SWING_RAILS = 4

# Twenty headlines cover the current intraday event surface without copying an
# unbounded feed. Replace with an event-horizon selector when one is canonical.
MAX_NEWS_ITEMS = 20

# Sixty-four segment rows cover the current pair/side/method frontier. Replace
# with configured frontier membership if the eligible universe grows.
MAX_EDGE_ROWS = 64

# Forty-eight named cost observations cover spread/slippage/latency/swap inputs
# without admitting full audit arrays. Replace with a typed cost schema later.
MAX_COST_FACTS = 48

# A 16 MiB input ceiling is a parser/resource guard, not a market rule. Current
# bounded chart packets are substantially smaller; replace with producer schema
# limits if those become explicit.
MAX_SOURCE_BYTES = 16 * 1024 * 1024

# Broker truth is execution-adjacent and must be no older than five minutes;
# chart/matrix/target truth is one M15 decision interval; news may span the
# two-hour strategic horizon; realized cost audits may span one trading day.
# Callers can tighten these limits per runtime profile.
DEFAULT_MAX_AGE_SECONDS: dict[str, int] = {
    "broker_snapshot": 5 * 60,
    "pair_charts": 15 * 60,
    "market_context_matrix": 15 * 60,
    "news_health": 2 * 60 * 60,
    "news_snapshot": 2 * 60 * 60,
    "capture_economics": 24 * 60 * 60,
    "execution_timing": 24 * 60 * 60,
    "daily_target_state": 15 * 60,
}

# The live prompt carries four complementary horizons. M1 is too noisy for the
# ten-minute decision cadence and D is represented by the separate strategic
# overlay; excluding both keeps the full 28-pair packet below its hard 256 KiB
# boundary without truncating any selected timeframe.
TIMEFRAMES = ("M5", "M15", "H1", "H4")
REQUIRED_SOURCES = frozenset(
    {
        "broker_snapshot",
        "pair_charts",
        "market_context_matrix",
        "news_health",
        "news_snapshot",
        "daily_target_state",
    }
)
LEGACY_MARKERS = (
    "order_intents",
    "hierarchical_bot_regime",
    "fast_bot",
    "fast-bot",
    "old_bot",
    "legacy_strategy",
)


class EvidenceAdapterError(ValueError):
    """Raised when a bounded canonical packet cannot be produced safely."""


@dataclass(frozen=True)
class EvidencePaths:
    broker_snapshot: Path
    pair_charts: Path
    market_context_matrix: Path
    news_health: Path
    news_snapshot: Path
    daily_target_state: Path
    capture_economics: Path | None = None
    execution_timing: Path | None = None


@dataclass(frozen=True)
class EvidenceWriteResult:
    output_path: Path
    packet_sha256: str
    size_bytes: int
    written: bool


def build_ai_evidence_packet(
    paths: EvidencePaths,
    *,
    now_utc: datetime | None = None,
    max_age_seconds: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Read current artifacts and return one sealed, no-lookahead AI packet."""

    now = _aware_utc(now_utc or datetime.now(timezone.utc))
    age_limits = dict(DEFAULT_MAX_AGE_SECONDS)
    if max_age_seconds:
        for name, value in max_age_seconds.items():
            if name not in age_limits or isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise EvidenceAdapterError(f"invalid source age policy: {name}")
            age_limits[name] = value

    configured: dict[str, Path | None] = {
        "broker_snapshot": paths.broker_snapshot,
        "pair_charts": paths.pair_charts,
        "market_context_matrix": paths.market_context_matrix,
        "news_health": paths.news_health,
        "news_snapshot": paths.news_snapshot,
        "capture_economics": paths.capture_economics,
        "execution_timing": paths.execution_timing,
        "daily_target_state": paths.daily_target_state,
    }
    sources: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any] | None] = {}
    for name, path in configured.items():
        descriptor, payload = _load_source(
            name=name,
            path=path,
            required=name in REQUIRED_SOURCES,
            max_age_seconds=age_limits[name],
            now=now,
        )
        sources[name] = descriptor
        payloads[name] = payload

    _validate_matrix_binding(sources, payloads)
    issues = [
        {
            "source": name,
            "code": descriptor["issue_code"],
            "required": descriptor["required"],
        }
        for name, descriptor in sorted(sources.items())
        if descriptor["status"] != "READY"
    ]
    blocking_sources = sorted(
        name
        for name, descriptor in sources.items()
        if descriptor["required"] and descriptor["status"] != "READY"
    )

    broker = payloads["broker_snapshot"] if sources["broker_snapshot"]["status"] == "READY" else None
    charts = payloads["pair_charts"] if sources["pair_charts"]["status"] == "READY" else None
    matrix = (
        payloads["market_context_matrix"]
        if sources["market_context_matrix"]["status"] == "READY"
        else None
    )
    quote_rows = _quotes(broker or {}, now=now, max_age_seconds=age_limits["broker_snapshot"])
    exposure = _exposure_summary(broker or {})
    markets, market_issues = _markets(
        charts or {},
        matrix or {},
        quote_rows,
        now=now,
    )
    issues.extend(market_issues)

    ready_times = [
        str(item["as_of_utc"])
        for item in sources.values()
        if item["status"] == "READY" and item.get("as_of_utc")
    ]
    status = "BLOCKED" if blocking_sources else ("DEGRADED" if issues else "READY")
    broker_account = (broker or {}).get("account")
    if not isinstance(broker_account, Mapping):
        broker_account = {}
    broker_source = sources["broker_snapshot"]
    body: dict[str, Any] = {
        "contract": CONTRACT,
        "schema_version": 1,
        "status": status,
        "evidence_as_of_utc": max(ready_times) if ready_times else None,
        "blocking_sources": blocking_sources,
        "issues": issues,
        "sources": sources,
        "source_set_sha256": _sha256_json(sources),
        "broker_epoch": {
            "as_of_utc": broker_source.get("as_of_utc"),
            "source_sha256": broker_source.get("sha256"),
            "last_transaction_id": _text(broker_account.get("last_transaction_id")),
        },
        "ownership_contract": {
            "system_owner_values": ["SYSTEM", "TRADER"],
            "operator_manual_and_unknown_policy": "NO_TOUCH",
            "raw_client_extensions_used_for_reclassification": False,
        },
        "broker": {
            "account": _select(
                broker_account,
                (
                    "balance_jpy",
                    "nav_jpy",
                    "margin_available_jpy",
                    "margin_used_jpy",
                    "margin_closeout_percent",
                    "unrealized_pl_jpy",
                    "pl_jpy",
                    "financing_jpy",
                    "hedging_enabled",
                ),
            ),
            "home_conversions": {
                str(currency): number
                for currency, raw in sorted(
                    (
                        broker.get("home_conversions")
                        if isinstance(broker, Mapping)
                        and isinstance(broker.get("home_conversions"), Mapping)
                        else {}
                    ).items()
                )
                if (number := _number(raw)) is not None and number > 0
            },
            "quotes": quote_rows,
            "exposure": exposure,
        },
        "markets": markets,
        "regime_dimensions_are_orthogonal": True,
        "news": _news(payloads, sources),
        "costs": _costs(payloads, sources),
        "net_edge_inputs": _net_edge_inputs(payloads, sources),
        "portfolio": _portfolio(payloads, sources, broker_account, exposure),
        "authority": {
            "packet_grants_live_permission": False,
            "packet_grants_broker_mutation": False,
            "order_candidates_included": False,
        },
    }
    packet = _seal(body)
    size = len(_canonical_bytes(packet)) + 1
    if size > MAX_PACKET_BYTES:
        raise EvidenceAdapterError(f"canonical evidence packet exceeds {MAX_PACKET_BYTES} bytes")
    return packet


def write_ai_evidence_packet(
    paths: EvidencePaths,
    output_path: Path,
    *,
    now_utc: datetime | None = None,
    max_age_seconds: Mapping[str, int] | None = None,
) -> EvidenceWriteResult:
    """Atomically publish a canonical packet, skipping an unchanged rewrite."""

    packet = build_ai_evidence_packet(
        paths,
        now_utc=now_utc,
        max_age_seconds=max_age_seconds,
    )
    raw = _canonical_bytes(packet) + b"\n"
    if output_path.exists():
        try:
            if output_path.read_bytes() == raw:
                return EvidenceWriteResult(
                    output_path=output_path,
                    packet_sha256=str(packet["packet_sha256"]),
                    size_bytes=len(raw),
                    written=False,
                )
        except OSError as exc:
            raise EvidenceAdapterError(f"cannot read existing evidence packet: {output_path}") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_symlink():
        raise EvidenceAdapterError("evidence output must not be a symlink")
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
        directory_fd = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return EvidenceWriteResult(
        output_path=output_path,
        packet_sha256=str(packet["packet_sha256"]),
        size_bytes=len(raw),
        written=True,
    )


def _load_source(
    *,
    name: str,
    path: Path | None,
    required: bool,
    max_age_seconds: int,
    now: datetime,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    descriptor: dict[str, Any] = {
        "required": required,
        "status": "MISSING",
        "issue_code": "SOURCE_MISSING",
        "sha256": None,
        "size_bytes": None,
        "as_of_utc": None,
        "stale_after_utc": None,
        "evidence_ref": None,
    }
    if path is None or not path.is_file():
        return descriptor, None
    if _contains_legacy_marker(str(path)):
        descriptor.update(status="REJECTED", issue_code="SOURCE_NOT_ALLOWED")
        return descriptor, None
    try:
        raw = path.read_bytes()
    except OSError:
        descriptor.update(status="MALFORMED", issue_code="SOURCE_UNREADABLE")
        return descriptor, None
    digest = hashlib.sha256(raw).hexdigest()
    descriptor.update(sha256=digest, size_bytes=len(raw))
    if len(raw) > MAX_SOURCE_BYTES:
        descriptor.update(status="MALFORMED", issue_code="SOURCE_TOO_LARGE")
        return descriptor, None
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        descriptor.update(status="MALFORMED", issue_code="SOURCE_JSON_INVALID")
        return descriptor, None
    if not isinstance(payload, dict):
        descriptor.update(status="MALFORMED", issue_code="SOURCE_NOT_OBJECT")
        return descriptor, None
    as_of = _source_as_of(name, payload)
    if as_of is None:
        descriptor.update(status="MALFORMED", issue_code="SOURCE_CLOCK_MISSING_OR_INVALID")
        return descriptor, None
    stale_after = as_of + timedelta(seconds=max_age_seconds)
    descriptor.update(
        as_of_utc=as_of.isoformat(),
        stale_after_utc=stale_after.isoformat(),
        evidence_ref=f"source:{name}:{digest}",
    )
    if as_of > now + timedelta(seconds=MAX_FUTURE_SKEW_SECONDS):
        descriptor.update(status="FUTURE", issue_code="SOURCE_CLOCK_IN_FUTURE")
        return descriptor, None
    if now > stale_after:
        descriptor.update(status="STALE", issue_code="SOURCE_STALE")
        return descriptor, None
    descriptor.update(status="READY", issue_code=None)
    return descriptor, payload


def _source_as_of(name: str, payload: Mapping[str, Any]) -> datetime | None:
    keys = {
        "broker_snapshot": ("fetched_at_utc", "as_of_utc", "generated_at_utc"),
        "daily_target_state": ("as_of_utc", "generated_at_utc"),
    }.get(name, ("generated_at_utc", "as_of_utc", "fetched_at_utc"))
    for key in keys:
        parsed = _parse_utc(payload.get(key))
        if parsed is not None:
            return parsed
    if name == "broker_snapshot" and isinstance(payload.get("account"), Mapping):
        return _parse_utc(payload["account"].get("fetched_at_utc"))
    return None


def _validate_matrix_binding(
    sources: dict[str, dict[str, Any]],
    payloads: dict[str, dict[str, Any] | None],
) -> None:
    if sources["market_context_matrix"]["status"] != "READY":
        return
    matrix = payloads["market_context_matrix"] or {}
    binding = matrix.get("pair_charts_binding")
    if binding is None:
        return
    if not isinstance(binding, Mapping) or binding.get("sha256") != sources["pair_charts"].get("sha256"):
        sources["market_context_matrix"].update(
            status="MALFORMED",
            issue_code="SOURCE_BINDING_MISMATCH",
            evidence_ref=None,
        )
        payloads["market_context_matrix"] = None


def _quotes(
    broker: Mapping[str, Any],
    *,
    now: datetime,
    max_age_seconds: int,
) -> dict[str, dict[str, Any]]:
    raw_quotes = broker.get("quotes") if isinstance(broker.get("quotes"), Mapping) else {}
    rows: dict[str, dict[str, Any]] = {}
    for pair, raw in sorted(raw_quotes.items(), key=lambda item: str(item[0]))[:MAX_PAIRS]:
        if not _pair(pair) or not isinstance(raw, Mapping):
            continue
        bid = _number(raw.get("bid"))
        ask = _number(raw.get("ask"))
        timestamp = _parse_utc(raw.get("timestamp_utc") or raw.get("as_of_utc"))
        if bid is None or ask is None or bid <= 0 or ask < bid or timestamp is None:
            rows[str(pair)] = {"status": "MALFORMED"}
            continue
        status = "READY"
        if timestamp > now + timedelta(seconds=MAX_FUTURE_SKEW_SECONDS):
            status = "FUTURE"
        elif now > timestamp + timedelta(seconds=max_age_seconds):
            status = "STALE"
        pip_size = 0.01 if str(pair).endswith("_JPY") else 0.0001
        rows[str(pair)] = {
            "status": status,
            "as_of_utc": timestamp.isoformat(),
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2.0,
            "spread_price": ask - bid,
            "spread_pips": (ask - bid) / pip_size,
        }
    return rows


def _exposure_summary(broker: Mapping[str, Any]) -> dict[str, Any]:
    positions = broker.get("positions") if isinstance(broker.get("positions"), list) else []
    orders = broker.get("orders") if isinstance(broker.get("orders"), list) else []
    position_rows = [_exposure_row(row, position=True) for row in positions if isinstance(row, Mapping)]
    order_rows = [_exposure_row(row, position=False) for row in orders if isinstance(row, Mapping)]
    position_rows = [row for row in position_rows if row is not None]
    order_rows = [row for row in order_rows if row is not None]
    counts = {owner: sum(row["ownership"] == owner for row in position_rows) for owner in ("SYSTEM", "OPERATOR_MANUAL", "UNKNOWN")}
    return {
        "positions": position_rows,
        "pending_orders": order_rows,
        "position_count_by_ownership": counts,
        "system_position_count": counts["SYSTEM"],
        "no_touch_position_count": counts["OPERATOR_MANUAL"] + counts["UNKNOWN"],
        "gross_position_units": sum(abs(float(row.get("units") or 0.0)) for row in position_rows),
        "system_unrealized_pl_jpy": sum(float(row.get("unrealized_pl_jpy") or 0.0) for row in position_rows if row["ownership"] == "SYSTEM"),
        "no_touch_unrealized_pl_jpy": sum(float(row.get("unrealized_pl_jpy") or 0.0) for row in position_rows if row["ownership"] != "SYSTEM"),
    }


def _exposure_row(row: Mapping[str, Any], *, position: bool) -> dict[str, Any] | None:
    pair = row.get("pair")
    if pair is not None and not _pair(pair):
        pair = None
    ownership = _owner_class(row.get("owner"))
    result: dict[str, Any] = {
        "ownership": ownership,
        "mutation_policy": "GATEWAY_VALIDATION_REQUIRED" if ownership == "SYSTEM" else "NO_TOUCH",
        "pair": pair,
        "side": _text(row.get("side")),
        "units": _number(row.get("units")),
    }
    if position:
        result.update(
            trade_id=_text(row.get("trade_id")),
            entry_price=_number(row.get("entry_price") or row.get("avg_entry")),
            take_profit=_number(row.get("take_profit")),
            stop_loss=_number(row.get("stop_loss")),
            unrealized_pl_jpy=_number(row.get("unrealized_pl_jpy")),
        )
    else:
        result.update(
            order_id=_text(row.get("order_id")),
            trade_id=_text(row.get("trade_id")),
            order_type=_text(row.get("order_type")),
            price=_number(row.get("price")),
            state=_text(row.get("state")),
        )
    return result


def _markets(
    pair_charts: Mapping[str, Any],
    matrix: Mapping[str, Any],
    quotes: Mapping[str, Any],
    *,
    now: datetime,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    chart_rows = pair_charts.get("charts") if isinstance(pair_charts.get("charts"), list) else []
    chart_as_of = _parse_utc(pair_charts.get("generated_at_utc")) or now
    by_pair: dict[str, Mapping[str, Any]] = {}
    duplicate_pairs: set[str] = set()
    for chart in chart_rows:
        if not isinstance(chart, Mapping) or not _pair(chart.get("pair")):
            continue
        pair = str(chart["pair"])
        if pair in by_pair:
            duplicate_pairs.add(pair)
        else:
            by_pair[pair] = chart
    issues: list[dict[str, Any]] = []
    markets: dict[str, Any] = {}
    for pair in sorted(by_pair)[:MAX_PAIRS]:
        if pair in duplicate_pairs:
            issues.append({"source": "pair_charts", "code": "DUPLICATE_PAIR", "pair": pair, "required": True})
            continue
        chart = by_pair[pair]
        views = chart.get("views") if isinstance(chart.get("views"), list) else []
        by_tf: dict[str, list[Mapping[str, Any]]] = {tf: [] for tf in TIMEFRAMES}
        for view in views:
            if isinstance(view, Mapping) and view.get("granularity") in by_tf:
                by_tf[str(view["granularity"])].append(view)
        tf_rows: dict[str, Any] = {}
        for timeframe in TIMEFRAMES:
            candidates = by_tf[timeframe]
            if len(candidates) != 1:
                tf_rows[timeframe] = {"status": "MISSING" if not candidates else "MALFORMED_DUPLICATE"}
                continue
            tf_rows[timeframe] = _timeframe_summary(candidates[0], cutoff=min(now, chart_as_of))
        confluence = chart.get("confluence") if isinstance(chart.get("confluence"), Mapping) else {}
        session = chart.get("session") if isinstance(chart.get("session"), Mapping) else {}
        markets[pair] = {
            "quote": quotes.get(pair),
            "dominant_regime": _text(chart.get("dominant_regime")),
            "long_score": _number(chart.get("long_score")),
            "short_score": _number(chart.get("short_score")),
            "orthogonal_regime": {
                "higher_tf_alignment": _text(confluence.get("higher_tf_alignment")),
                "higher_tf_regime": _text(confluence.get("higher_tf_regime")),
                "score_balance": _text(confluence.get("score_balance")),
                "tf_agreement_score": _number(confluence.get("tf_agreement_score")),
                "price_percentile_24h": _number(confluence.get("price_percentile_24h")),
                "price_percentile_7d": _number(confluence.get("price_percentile_7d")),
                "atr_percentile_24h": _number(confluence.get("atr_percentile_24h")),
                "range_24h_expansion_outlier": _bool_or_none(confluence.get("range_24h_expansion_outlier")),
            },
            "session": _select(session, ("current_tag", "jp_holiday", "holiday_name", "next_killzone", "minutes_to_next_killzone")),
            "timeframes": tf_rows,
            "market_context": _matrix_pair(matrix, pair),
        }
    if len(by_pair) > MAX_PAIRS:
        issues.append({"source": "pair_charts", "code": "PAIR_LIMIT_EXCEEDED", "required": True})
    return markets, issues


def _timeframe_summary(view: Mapping[str, Any], *, cutoff: datetime) -> dict[str, Any]:
    indicators = view.get("indicators") if isinstance(view.get("indicators"), Mapping) else {}
    structure = view.get("structure") if isinstance(view.get("structure"), Mapping) else {}
    regime = view.get("regime_reading") if isinstance(view.get("regime_reading"), Mapping) else {}
    family = view.get("family_scores") if isinstance(view.get("family_scores"), Mapping) else {}
    market_state = view.get("market_state") if isinstance(view.get("market_state"), Mapping) else {}
    candle = _latest_complete_candle(view.get("recent_candles"), cutoff=cutoff)
    candle_cutoff = _parse_utc((candle or {}).get("t")) or cutoff
    statistical_regime = _select(regime, ("state", "hurst", "adx", "choppiness", "atr_percentile", "confidence", "source", "lookback_bars"))
    family_scores = _select(family, ("trend_score", "mean_rev_score", "breakout_score", "disagreement"))
    market_state_row = _select(
        market_state,
        (
            "phase", "direction", "direction_quality", "trend_strength", "volatility",
            "momentum", "noise", "structure", "trigger", "location", "value_zone",
            "extension", "mean_reversion_speed", "liquidity", "trend_maturity",
            "readiness", "strategy_family", "entry_mode", "invalidation_phase",
            "confidence", "evidence_complete",
        ),
    )
    atr_pips = _number(indicators.get("atr_pips"))
    legacy_regime = _text(view.get("regime"))
    last_event = _last_confirmed_event(structure, cutoff=candle_cutoff)
    swing_rails = _swing_rails(structure, cutoff=candle_cutoff)
    required_parts = {
        "latest_complete_ohlc": candle,
        "atr_pips": atr_pips,
        "regime": legacy_regime,
        "statistical_regime": statistical_regime or None,
        "market_state": market_state_row or None,
        "last_confirmed_structure_event": last_event,
        "recent_swing_rails": swing_rails or None,
    }
    missing = [key for key, value in required_parts.items() if value is None]
    return {
        "status": "READY" if not missing else "PARTIAL",
        "missing": missing,
        "latest_complete_ohlc": candle,
        "atr_pips": atr_pips,
        "regime": legacy_regime,
        "statistical_regime": statistical_regime,
        "family_scores": family_scores,
        "market_state": market_state_row,
        "last_confirmed_structure_event": last_event,
        "recent_swing_rails": swing_rails,
    }


def _latest_complete_candle(value: Any, *, cutoff: datetime) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return None
    candidates: list[tuple[datetime, dict[str, Any]]] = []
    for row in value:
        if not isinstance(row, Mapping) or row.get("complete") is not True:
            continue
        stamp = _parse_utc(row.get("t") or row.get("timestamp_utc"))
        ohlc = [_number(row.get(key)) for key in ("o", "h", "l", "c")]
        if stamp is None or stamp > cutoff or any(item is None for item in ohlc):
            continue
        o, high, low, close = ohlc
        if low is None or high is None or o is None or close is None or low > min(o, close) or high < max(o, close):
            continue
        candidates.append((stamp, {"t": stamp.isoformat(), "o": o, "h": high, "l": low, "c": close}))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _last_confirmed_event(structure: Mapping[str, Any], *, cutoff: datetime) -> dict[str, Any] | None:
    values: list[Any] = []
    if isinstance(structure.get("last_event"), Mapping):
        values.append(structure["last_event"])
    if isinstance(structure.get("structure_events"), list):
        values.extend(structure["structure_events"])
    candidates: list[tuple[datetime, dict[str, Any]]] = []
    for row in values:
        if not isinstance(row, Mapping) or row.get("close_confirmed") is not True:
            continue
        stamp = _parse_utc(row.get("timestamp") or row.get("timestamp_utc"))
        price = _number(row.get("broken_pivot_price"))
        if stamp is None or stamp > cutoff or price is None:
            continue
        candidates.append((stamp, {"timestamp_utc": stamp.isoformat(), "kind": _text(row.get("kind")), "broken_pivot_price": price}))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _swing_rails(structure: Mapping[str, Any], *, cutoff: datetime) -> list[dict[str, Any]]:
    values = structure.get("swings") if isinstance(structure.get("swings"), list) else []
    candidates: list[tuple[datetime, dict[str, Any]]] = []
    for row in values:
        if not isinstance(row, Mapping):
            continue
        stamp = _parse_utc(row.get("timestamp") or row.get("timestamp_utc"))
        price = _number(row.get("price"))
        side = _text(row.get("side"))
        if stamp is None or stamp > cutoff or price is None or side not in {"HIGH", "LOW"}:
            continue
        candidates.append((stamp, {"timestamp_utc": stamp.isoformat(), "side": side, "price": price}))
    return [row for _, row in sorted(candidates, key=lambda item: item[0])[-MAX_SWING_RAILS:]]


def _matrix_pair(matrix: Mapping[str, Any], pair: str) -> dict[str, Any] | None:
    pairs = matrix.get("pairs") if isinstance(matrix.get("pairs"), Mapping) else {}
    pair_row = pairs.get(pair)
    if not isinstance(pair_row, Mapping):
        return None
    result: dict[str, Any] = {}
    for side in ("LONG", "SHORT"):
        raw = pair_row.get(side)
        if not isinstance(raw, Mapping):
            result[side] = None
            continue
        result[side] = {
            "evidence_ref": _safe_text(raw.get("evidence_ref")),
            "support_count": _integer(raw.get("support_count")),
            "reject_count": _integer(raw.get("reject_count")),
            "warning_count": _integer(raw.get("warning_count")),
            "missing_count": _integer(raw.get("missing_count")),
            "horizon_conflict_count": _integer(raw.get("horizon_conflict_count")),
            "strongest_support": _safe_text(raw.get("strongest_support")),
            "strongest_reject": _safe_text(raw.get("strongest_reject")),
            "strongest_warning": _safe_text(raw.get("strongest_warning")),
            "correlation_facts": _correlation_facts(raw),
        }
    return result


def _correlation_facts(side: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket in ("supports", "rejects", "warnings"):
        values = side.get(bucket) if isinstance(side.get(bucket), list) else []
        for item in values:
            if not isinstance(item, Mapping):
                continue
            haystack = " ".join(str(item.get(key) or "") for key in ("code", "layer", "message")).lower()
            if "correl" not in haystack:
                continue
            rows.append({"bucket": bucket, "code": _safe_text(item.get("code")), "message": _safe_text(item.get("message"))})
    return rows[:MAX_SWING_RAILS]


def _news(payloads: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    health = payloads.get("news_health") if sources["news_health"]["status"] == "READY" else None
    snapshot = payloads.get("news_snapshot") if sources["news_snapshot"]["status"] == "READY" else None
    items: list[dict[str, Any]] = []
    raw_items = snapshot.get("items") if isinstance(snapshot, Mapping) and isinstance(snapshot.get("items"), list) else []
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            continue
        title = _safe_text(raw.get("title"))
        published = _parse_utc(raw.get("published_at_utc") or raw.get("timestamp_utc"))
        if title is None or published is None:
            continue
        items.append(
            {
                "published_at_utc": published.isoformat(),
                "title": title,
                "source": _safe_text(raw.get("source")),
                "pairs": _safe_text_list(raw.get("pairs")),
                "topics": _safe_text_list(raw.get("topics")),
                "impact": _safe_text(raw.get("impact") or raw.get("importance")),
            }
        )
    items.sort(key=lambda item: item["published_at_utc"], reverse=True)
    return {
        "health": _select(
            health if isinstance(health, Mapping) else {},
            ("status", "fresh", "item_count", "source_count", "coverage", "issues", "latest_published_at_utc"),
        ),
        "items": items[:MAX_NEWS_ITEMS],
    }


def _costs(payloads: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    capture = payloads.get("capture_economics") if sources["capture_economics"]["status"] == "READY" else None
    timing = payloads.get("execution_timing") if sources["execution_timing"]["status"] == "READY" else None
    broker = payloads.get("broker_snapshot") if sources["broker_snapshot"]["status"] == "READY" else None
    account = broker.get("account") if isinstance(broker, Mapping) and isinstance(broker.get("account"), Mapping) else {}
    return {
        "account_financing_jpy": _number(account.get("financing_jpy")),
        "capture_status": _text(capture.get("status")) if isinstance(capture, Mapping) else None,
        "capture_overall": _select(
            capture.get("overall") if isinstance(capture, Mapping) and isinstance(capture.get("overall"), Mapping) else {},
            ("trades", "wins", "losses", "win_rate", "avg_win_jpy", "avg_loss_jpy", "expectancy_jpy_per_trade", "payoff_ratio", "breakeven_payoff_at_win_rate", "net_jpy"),
        ),
        "spread_slippage_latency_swap_facts": _collect_cost_facts((capture, timing)),
    }


def _net_edge_inputs(payloads: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    capture = payloads.get("capture_economics") if sources["capture_economics"]["status"] == "READY" else None
    if not isinstance(capture, Mapping):
        return {"status": "UNAVAILABLE", "segments": []}
    ai_exact = capture.get("ai_entry_net_edge")
    exact_items = (
        ai_exact.get("items")
        if isinstance(ai_exact, Mapping) and isinstance(ai_exact.get("items"), list)
        else []
    )
    exact_segments: list[dict[str, Any]] = []
    for row in exact_items:
        if not isinstance(row, Mapping) or row.get("ai_entry_eligible") is not True:
            continue
        exact_segments.append(
            _select(
                row,
                (
                    "pair", "side", "method", "vehicle", "trades", "wins", "losses",
                    "net_jpy", "expectancy_jpy_per_trade", "avg_win_jpy", "avg_loss_jpy",
                    "unresolved_realized_trades", "unresolved_realized_net_jpy",
                    "win_rate_wilson95_lower", "wilson_stressed_expectancy_jpy",
                    "proof_class", "ai_entry_eligible",
                ),
            )
        )
    if exact_segments:
        return {
            "status": "READY",
            "proof_class": _text(ai_exact.get("proof_class")),
            "segments": exact_segments[:MAX_EDGE_ROWS],
        }

    raw = capture.get("segment_repair_priorities")
    items = raw.get("items") if isinstance(raw, Mapping) and isinstance(raw.get("items"), list) else []
    segments: list[dict[str, Any]] = []
    for row in items:
        if not isinstance(row, Mapping):
            continue
        segments.append(
            _select(
                row,
                (
                    "evidence_ref", "pair", "side", "method", "trades", "wins", "losses",
                    "win_rate", "avg_win_jpy", "avg_loss_jpy", "expectancy_jpy_per_trade",
                    "payoff_ratio", "breakeven_payoff_at_win_rate", "net_jpy",
                ),
            )
        )
    return {"status": _text(capture.get("status")), "segments": segments[:MAX_EDGE_ROWS]}


def _portfolio(
    payloads: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    account: Mapping[str, Any],
    exposure: Mapping[str, Any],
) -> dict[str, Any]:
    daily = payloads.get("daily_target_state") if sources["daily_target_state"]["status"] == "READY" else None
    daily_row = _select(
        daily if isinstance(daily, Mapping) else {},
        (
            "status", "campaign_day", "pace_state", "performance_basis", "sizing_basis",
            "current_equity_raw", "funding_adjusted_equity", "daily_risk_budget_jpy",
            "daily_loss_capacity_before_open_jpy", "open_risk_jpy", "remaining_risk_budget_jpy",
            "realized_loss_spent_jpy", "realized_pl_jpy", "unrealized_pl_jpy",
            "rolling_30d_multiplier_funding_adjusted", "remaining_to_4x_funding_adjusted",
            "required_calendar_daily_return_funding_adjusted", "required_active_day_return_funding_adjusted",
        ),
    )
    return {
        "margin": _select(account, ("nav_jpy", "margin_available_jpy", "margin_used_jpy", "margin_closeout_percent", "hedging_enabled")),
        "exposure": {
            "system_position_count": exposure.get("system_position_count"),
            "no_touch_position_count": exposure.get("no_touch_position_count"),
            "gross_position_units": exposure.get("gross_position_units"),
            "system_unrealized_pl_jpy": exposure.get("system_unrealized_pl_jpy"),
            "no_touch_unrealized_pl_jpy": exposure.get("no_touch_unrealized_pl_jpy"),
        },
        "daily_target": daily_row,
        "correlation_source": "market_context_matrix.pairs.*.*.correlation_facts",
    }


def _collect_cost_facts(values: Sequence[Any]) -> list[dict[str, Any]]:
    keywords = ("spread", "slippage", "latency", "lag", "swap", "financing", "cost")
    rows: list[dict[str, Any]] = []

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if len(rows) >= MAX_COST_FACTS or _contains_legacy_marker(".".join(path)):
            return
        if isinstance(value, Mapping):
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                visit(item, (*path, str(key)))
        elif isinstance(value, list):
            for item in value[:MAX_COST_FACTS]:
                visit(item, path)
        elif path and any(keyword in path[-1].lower() for keyword in keywords):
            scalar = _scalar(value)
            if scalar is not None:
                rows.append({"field": ".".join(path[-4:]), "value": scalar})

    for value in values:
        if value is not None:
            visit(value, ())
    return rows[:MAX_COST_FACTS]


def _select(value: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        if key not in value or _contains_legacy_marker(key):
            continue
        item = value[key]
        if isinstance(item, (str, int, float, bool)) or item is None:
            scalar = _scalar(item)
            if scalar is not None or item is None:
                result[key] = scalar
        elif key == "issues" and isinstance(item, list):
            result[key] = [_safe_text(part) for part in item if _safe_text(part) is not None]
    return result


def _owner_class(value: Any) -> str:
    owner = str(value or "").strip().upper()
    if owner in {"SYSTEM", "TRADER"}:
        return "SYSTEM"
    if owner in {"MANUAL", "OPERATOR_MANUAL"}:
        return "OPERATOR_MANUAL"
    return "UNKNOWN"


def _safe_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := _safe_text(item)) is not None]


def _safe_text(value: Any) -> str | None:
    text = _text(value)
    if text is None or _contains_legacy_marker(text):
        return None
    # Five hundred characters preserves a concise evidence statement. Replace
    # with typed producer bounds when every source publishes one.
    return text[:500]


def _contains_legacy_marker(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in LEGACY_MARKERS)


def _pair(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 7 and text[3] == "_" and text.replace("_", "").isalpha() and text == text.upper()


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return value


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _scalar(value: Any) -> str | int | float | bool | None:
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _number(value)
    return None


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise EvidenceAdapterError("now_utc must be timezone-aware")
    return value.astimezone(timezone.utc)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceAdapterError("evidence packet is not canonical JSON") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "packet_sha256"}
    return {**body, "packet_sha256": _sha256_json(body)}


__all__ = [
    "CONTRACT",
    "MAX_PACKET_BYTES",
    "EvidenceAdapterError",
    "EvidencePaths",
    "EvidenceWriteResult",
    "build_ai_evidence_packet",
    "write_ai_evidence_packet",
]
