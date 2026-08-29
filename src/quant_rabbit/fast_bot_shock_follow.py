"""Prospective-only, deterministic shock-follow fast-bot shadow lanes.

The module owns no broker write path.  It observes completed M1/M5 candles,
freezes STOP-entry hypotheses, and later scores them against exact OANDA S5
BID/ASK truth in ledgers separated from the ordinary fast bot and corrective
challenger.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle
from quant_rabbit.technical_forecast_forward_truth import fetch_frozen_s5_truth


CONFIG_CONTRACT = "QR_FAST_BOT_SHOCK_FOLLOW_CONFIG_V1"
SIGNAL_CONTRACT = "QR_FAST_BOT_SHOCK_FOLLOW_SIGNAL_V1"
OUTCOME_CONTRACT = "QR_FAST_BOT_SHOCK_FOLLOW_S5_OUTCOME_V1"
SCORECARD_CONTRACT = "QR_FAST_BOT_SHOCK_FOLLOW_SCORECARD_V1"
RUN_CONTRACT = "QR_FAST_BOT_SHOCK_FOLLOW_RUN_V1"
SCORING_POLICY = "QR_FAST_BOT_SHOCK_FOLLOW_STOP_S5_CONSERVATIVE_V1"
STRATEGIES = (
    "SHOCK_BREAKOUT_FOLLOW",
    "SHOCK_PULLBACK_CONTINUATION",
)
PAIRS = ("EUR_USD", "USD_JPY")
DIAGNOSTIC_ARMS = ("BASELINE", "VOL_SHOCK_VETO", "COMBINED")
STOP_REASONS = {
    "STOP_LOSS",
    "STOP_LOSS_GAP",
    "STOP_LOSS_AMBIGUOUS_FILL_S5",
    "STOP_LOSS_GAP_AMBIGUOUS_FILL_S5",
    "STOP_LOSS_AMBIGUOUS_SAME_S5",
}
NY = ZoneInfo("America/New_York")


def canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return {**body, "contract_sha256": canonical_sha(body)}


def sealed_valid(value: Mapping[str, Any], contract: str) -> bool:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return value.get("contract") == contract and value.get("contract_sha256") == canonical_sha(body)


def parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("aware timestamp is required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("aware datetime is required")
    return value.astimezone(timezone.utc)


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("contract") != CONFIG_CONTRACT:
        raise ValueError("shock-follow config contract mismatch")
    authority = config.get("authority") or {}
    if (
        authority.get("execution_authority") != "NONE"
        or authority.get("broker_http_methods_allowed") != ["GET"]
        or authority.get("broker_mutation_allowed") is not False
        or authority.get("live_permission") is not False
        or authority.get("promotion_allowed") is not False
        or authority.get("automatic_adoption_allowed") is not False
        or authority.get("automatic_parameter_change_allowed") is not False
    ):
        raise ValueError("shock-follow authority boundary mismatch")
    evidence = config.get("evidence") or {}
    if (
        evidence.get("lookahead_policy")
        != "COMPLETED_M1_M5_AVAILABLE_AT_OBSERVATION_ONLY"
        or evidence.get("retrospective_reinterpretation_allowed") is not False
        or evidence.get("historical_intervals_are_diagnostic_only") is not True
    ):
        raise ValueError("shock-follow evidence boundary mismatch")
    parse_utc(evidence.get("prospective_start_at_utc"))
    pairs = tuple(config.get("pairs") or [])
    if pairs != PAIRS:
        raise ValueError("shock-follow pair boundary mismatch")
    strategies = tuple(row.get("strategy_id") for row in config.get("strategies") or [])
    if strategies != STRATEGIES:
        raise ValueError("shock-follow strategy boundary mismatch")
    for row in config["strategies"]:
        if (
            row.get("order_type") != "STOP"
            or not 1 <= int(row.get("entry_ttl_seconds") or 0) <= 300
            or int(row.get("max_hold_seconds") or 0) <= 0
        ):
            raise ValueError("shock-follow STOP/TTL boundary mismatch")
    truth = config.get("truth") or {}
    if (
        truth.get("granularity") != "S5"
        or truth.get("price_component") != "BID_ASK"
        or not 1 <= int(truth.get("chunk_candle_limit") or 0) <= 5000
    ):
        raise ValueError("shock-follow S5 truth boundary mismatch")
    return config, canonical_sha(config)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{number}")
            rows.append(row)
    return rows


def market_is_closed(at_utc: datetime) -> bool:
    local = aware_utc(at_utc).astimezone(NY)
    weekday = local.weekday()
    if weekday == 5:
        return True
    if weekday == 4 and local.hour >= 17:
        return True
    if weekday == 6 and local.hour < 17:
        return True
    return False


def _chart_by_pair(packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("pair")): row
        for row in packet.get("charts", []) or []
        if isinstance(row, Mapping) and row.get("pair")
    }


def _view(chart: Mapping[str, Any], timeframe: str) -> Mapping[str, Any] | None:
    for row in chart.get("views", []) or []:
        if isinstance(row, Mapping) and str(row.get("granularity") or "").upper() == timeframe:
            return row
    return None


def _completed_candles(
    view: Mapping[str, Any] | None,
    *,
    timeframe_seconds: int,
    now: datetime,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not isinstance(view, Mapping):
        return result
    for raw in view.get("recent_candles", []) or []:
        if not isinstance(raw, Mapping) or raw.get("complete") is not True:
            continue
        try:
            started = parse_utc(raw.get("t"))
            values = {key: float(raw[key]) for key in ("o", "h", "l", "c")}
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
            continue
        if values["h"] < max(values["o"], values["c"]) or values["l"] > min(values["o"], values["c"]):
            continue
        closed = started + timedelta(seconds=timeframe_seconds)
        if closed > now:
            continue
        result.append({**values, "started_at": started, "closed_at": closed})
    result.sort(key=lambda row: row["started_at"])
    deduped = {row["started_at"]: row for row in result}
    return [deduped[key] for key in sorted(deduped)]


def _candle_integrity_passes(view: Mapping[str, Any] | None, *, pair: str, timeframe: str) -> bool:
    if not isinstance(view, Mapping):
        return False
    integrity = view.get("candle_integrity")
    if not isinstance(integrity, Mapping):
        return False
    return (
        integrity.get("schema") == "QR_TECHNICAL_CANDLE_INTEGRITY_V2"
        and integrity.get("source") == "OANDA_MBA"
        and integrity.get("pair") == pair
        and str(integrity.get("granularity") or "").upper() == timeframe
        and integrity.get("evaluation_status") == "PASS"
        and integrity.get("forecast_blocking") is False
        and integrity.get("provenance_complete") is True
        and integrity.get("coverage_complete") is True
        and integrity.get("recent_clean_coverage_complete") is True
    )


def _true_ranges(candles: Sequence[Mapping[str, Any]], factor: float) -> list[float]:
    values: list[float] = []
    for index, candle in enumerate(candles):
        previous = float(candles[index - 1]["c"]) if index else float(candle["o"])
        values.append(
            max(
                float(candle["h"]) - float(candle["l"]),
                abs(float(candle["h"]) - previous),
                abs(float(candle["l"]) - previous),
            )
            * factor
        )
    return values


def _prior_atr(candles: Sequence[Mapping[str, Any]], index: int, period: int, factor: float) -> float | None:
    if index < period:
        return None
    values = _true_ranges(candles[: index + 1], factor)
    prior = values[index - period:index]
    return statistics.fmean(prior) if len(prior) == period and min(prior) > 0.0 else None


def _direction(candle: Mapping[str, Any]) -> str | None:
    if float(candle["c"]) > float(candle["o"]):
        return "LONG"
    if float(candle["c"]) < float(candle["o"]):
        return "SHORT"
    return None


def _shock_bucket(expansion: float, edges: Sequence[float]) -> str:
    first, second, third = (float(value) for value in edges)
    if expansion < second:
        return f"SHOCK_{first:g}_TO_LT_{second:g}"
    if expansion < third:
        return f"SHOCK_{second:g}_TO_LT_{third:g}"
    return f"SHOCK_GE_{third:g}"


def _shock_features(
    candles: Sequence[Mapping[str, Any]],
    *,
    index: int,
    factor: float,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    policy = config["shock"]
    period = int(policy["prior_atr_period"])
    atr = _prior_atr(candles, index, period, factor)
    if atr is None:
        return None
    candle = candles[index]
    tr = _true_ranges(candles[: index + 1], factor)[index]
    body = abs(float(candle["c"]) - float(candle["o"])) * factor
    direction = _direction(candle)
    expansion = tr / atr
    impulse = body / atr
    body_fraction = body / tr if tr > 0.0 else 0.0
    if (
        direction is None
        or expansion < float(policy["m1_atr_expansion_min"])
        or impulse < float(policy["m1_impulse_body_atr_min"])
        or body_fraction < float(policy["directional_body_fraction_min"])
    ):
        return None
    return {
        "direction": direction,
        "prior_m1_atr_pips": round(atr, 6),
        "m1_true_range_pips": round(tr, 6),
        "m1_atr_expansion_ratio": round(expansion, 6),
        "m1_impulse_body_atr_ratio": round(impulse, 6),
        "m1_directional_body_fraction": round(body_fraction, 6),
        "shock_bucket": _shock_bucket(expansion, policy["bucket_edges"]),
    }


def _m5_alignment(
    candles: Sequence[Mapping[str, Any]],
    view: Mapping[str, Any] | None,
    *,
    factor: float,
    side: str,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    policy = config["shock"]
    period = int(policy["prior_atr_period"])
    index = len(candles) - 1
    atr = _prior_atr(candles, index, period, factor)
    if atr is None:
        return None
    candle = candles[index]
    tr = _true_ranges(candles, factor)[index]
    body = abs(float(candle["c"]) - float(candle["o"])) * factor
    market = view.get("market_state") if isinstance(view, Mapping) and isinstance(view.get("market_state"), Mapping) else {}
    expected = "UP" if side == "LONG" else "DOWN"
    if (
        _direction(candle) != side
        or str(market.get("direction") or "").upper() != expected
        or market.get("evidence_complete") is not True
        or tr / atr < float(policy["m5_atr_expansion_min"])
        or body / atr < float(policy["m5_directional_body_atr_min"])
    ):
        return None
    return {
        "prior_m5_atr_pips": round(atr, 6),
        "m5_true_range_pips": round(tr, 6),
        "m5_atr_expansion_ratio": round(tr / atr, 6),
        "m5_direction": expected,
        "m5_closed_at_utc": candle["closed_at"].isoformat(),
    }


def _positive(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def _price(pair: str, value: float) -> float:
    return round(value, 3 if pair.endswith("_JPY") else 5)


def _strategy_map(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["strategy_id"]): row for row in config["strategies"]}


def _signal_geometry(
    *,
    pair: str,
    side: str,
    strategy: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    atr_pips: float,
) -> dict[str, Any]:
    factor = float(instrument_pip_factor(pair))
    buffer = atr_pips * float(strategy["confirmation_buffer_atr"])
    entry = (
        float(confirmation["h"]) + buffer / factor
        if side == "LONG"
        else float(confirmation["l"]) - buffer / factor
    )
    stop_pips = atr_pips * float(strategy["stop_atr"])
    target_pips = atr_pips * float(strategy["take_profit_atr"])
    stop = entry - stop_pips / factor if side == "LONG" else entry + stop_pips / factor
    target = entry + target_pips / factor if side == "LONG" else entry - target_pips / factor
    rounded_entry = _price(pair, entry)
    rounded_stop = _price(pair, stop)
    rounded_target = _price(pair, target)
    actual_stop = abs(rounded_entry - rounded_stop) * factor
    actual_target = abs(rounded_target - rounded_entry) * factor
    return {
        "entry": rounded_entry,
        "stop_loss": rounded_stop,
        "take_profit": rounded_target,
        "stop_loss_pips": round(actual_stop, 6),
        "take_profit_pips": round(actual_target, 6),
        "reward_risk": round(actual_target / actual_stop, 6),
    }


def build_shock_follow_shadow(
    *,
    pair_charts: Mapping[str, Any],
    broker_snapshot: Mapping[str, Any],
    config: Mapping[str, Any],
    config_sha256: str,
    now_utc: datetime,
) -> dict[str, Any]:
    """Emit only patterns whose final confirmation bar was unknown previously."""

    now = aware_utc(now_utc)
    blockers: list[str] = []
    if market_is_closed(now):
        blockers.append("MARKET_CLOSED")
    start = parse_utc(config["evidence"]["prospective_start_at_utc"])
    packet_at = None
    snapshot_at = None
    try:
        packet_at = parse_utc(pair_charts.get("generated_at_utc"))
        snapshot_at = parse_utc(broker_snapshot.get("fetched_at_utc"))
    except ValueError:
        blockers.append("PACKET_OR_SNAPSHOT_TIMESTAMP_INVALID")
    market_policy = config["market"]
    if packet_at is None or not 0.0 <= (now - packet_at).total_seconds() <= float(market_policy["max_packet_age_seconds"]):
        blockers.append("PAIR_CHART_PACKET_STALE_OR_FUTURE")
    if snapshot_at is None or not 0.0 <= (now - snapshot_at).total_seconds() <= float(market_policy["max_quote_age_seconds"]):
        blockers.append("BROKER_SNAPSHOT_STALE_OR_FUTURE")
    if now < start:
        blockers.append("PROSPECTIVE_BOUNDARY_NOT_REACHED")
    charts = _chart_by_pair(pair_charts)
    quotes = broker_snapshot.get("quotes") if isinstance(broker_snapshot.get("quotes"), Mapping) else {}
    signals: list[dict[str, Any]] = []
    pair_rejections: dict[str, list[str]] = {}
    if not blockers:
        for pair in config["pairs"]:
            rejected: list[str] = []
            chart = charts.get(pair)
            m1_view = _view(chart or {}, "M1")
            m5_view = _view(chart or {}, "M5")
            if not _candle_integrity_passes(m1_view, pair=pair, timeframe="M1") or not _candle_integrity_passes(
                m5_view,
                pair=pair,
                timeframe="M5",
            ):
                rejected.append("M1_M5_CANDLE_INTEGRITY_NOT_PROVEN")
                pair_rejections[pair] = rejected
                continue
            m1 = _completed_candles(m1_view, timeframe_seconds=60, now=now)
            m5 = _completed_candles(m5_view, timeframe_seconds=300, now=now)
            period = int(config["shock"]["prior_atr_period"])
            if len(m1) < period + 3 or len(m5) < period + 1:
                rejected.append("COMPLETED_M1_M5_TRUTH_INSUFFICIENT")
                pair_rejections[pair] = rejected
                continue
            if not 0.0 <= (now - m1[-1]["closed_at"]).total_seconds() <= float(market_policy["max_m1_close_age_seconds"]):
                rejected.append("M1_CLOSED_CANDLE_STALE_OR_FUTURE")
            if not 0.0 <= (now - m5[-1]["closed_at"]).total_seconds() <= float(market_policy["max_m5_close_age_seconds"]):
                rejected.append("M5_CLOSED_CANDLE_STALE_OR_FUTURE")
            quote = quotes.get(pair) if isinstance(quotes.get(pair), Mapping) else {}
            bid = _positive(quote.get("bid"))
            ask = _positive(quote.get("ask"))
            try:
                quote_at = parse_utc(quote.get("timestamp_utc"))
            except ValueError:
                quote_at = None
            if bid is None or ask is None or ask <= bid or quote_at is None:
                rejected.append("QUOTE_TRUTH_INVALID")
            elif not 0.0 <= (now - quote_at).total_seconds() <= float(market_policy["max_quote_age_seconds"]):
                rejected.append("QUOTE_STALE_OR_FUTURE")
            factor = float(instrument_pip_factor(pair))
            latest_atr = _prior_atr(m1, len(m1) - 1, period, factor)
            spread = (ask - bid) * factor if bid is not None and ask is not None else None
            if latest_atr is None or spread is None or spread / latest_atr > float(config["shock"]["max_spread_to_m1_atr"]):
                rejected.append("SPREAD_SHOCK_OR_ATR_UNAVAILABLE")
            if rejected:
                pair_rejections[pair] = sorted(set(rejected))
                continue
            for strategy_id, anchor_index, confirmation_index in (
                ("SHOCK_BREAKOUT_FOLLOW", len(m1) - 2, len(m1) - 1),
                ("SHOCK_PULLBACK_CONTINUATION", len(m1) - 3, len(m1) - 1),
            ):
                shock = _shock_features(m1, index=anchor_index, factor=factor, config=config)
                if shock is None:
                    continue
                side = str(shock["direction"])
                alignment = _m5_alignment(m5, m5_view, factor=factor, side=side, config=config)
                if alignment is None:
                    continue
                anchor = m1[anchor_index]
                confirmation = m1[confirmation_index]
                if confirmation["closed_at"] < start:
                    continue
                strategy = _strategy_map(config)[strategy_id]
                confirmed = False
                pattern: dict[str, Any] = {}
                if strategy_id == "SHOCK_BREAKOUT_FOLLOW":
                    confirmed = (
                        float(confirmation["h"]) > float(anchor["h"])
                        and float(confirmation["c"]) > float(anchor["h"])
                        if side == "LONG"
                        else float(confirmation["l"]) < float(anchor["l"])
                        and float(confirmation["c"]) < float(anchor["l"])
                    )
                else:
                    pullback = m1[-2]
                    atr_price = float(shock["prior_m1_atr_pips"]) / factor
                    retrace = (
                        (float(anchor["h"]) - float(pullback["l"])) / atr_price
                        if side == "LONG"
                        else (float(pullback["h"]) - float(anchor["l"])) / atr_price
                    )
                    invalidation = float(strategy["invalidation_atr"]) * atr_price
                    bounded = float(strategy["pullback_atr_min"]) <= retrace <= float(strategy["pullback_atr_max"])
                    intact = (
                        float(pullback["l"]) >= float(anchor["l"]) - invalidation
                        if side == "LONG"
                        else float(pullback["h"]) <= float(anchor["h"]) + invalidation
                    )
                    reaccelerated = (
                        float(confirmation["c"]) > float(pullback["h"]) and _direction(confirmation) == "LONG"
                        if side == "LONG"
                        else float(confirmation["c"]) < float(pullback["l"]) and _direction(confirmation) == "SHORT"
                    )
                    confirmed = bounded and intact and reaccelerated
                    pattern = {
                        "pullback_atr_ratio": round(retrace, 6),
                        "pullback_started_at_utc": pullback["started_at"].isoformat(),
                        "opposite_break_invalidated": not intact,
                    }
                if not confirmed:
                    continue
                geometry = _signal_geometry(
                    pair=pair,
                    side=side,
                    strategy=strategy,
                    confirmation=confirmation,
                    atr_pips=float(shock["prior_m1_atr_pips"]),
                )
                if (side == "LONG" and geometry["entry"] <= ask) or (side == "SHORT" and geometry["entry"] >= bid):
                    continue
                identity = {
                    "config_sha256": config_sha256,
                    "pair": pair,
                    "strategy_id": strategy_id,
                    "side": side,
                    "shock_bucket": shock["shock_bucket"],
                    "shock_anchor_closed_at_utc": anchor["closed_at"].isoformat(),
                    "confirmation_closed_at_utc": confirmation["closed_at"].isoformat(),
                }
                body = {
                    "contract": SIGNAL_CONTRACT,
                    "schema_version": 1,
                    "signal_id": canonical_sha(identity)[:24],
                    "generated_at_utc": now.isoformat(),
                    **identity,
                    "strategy": strategy_id,
                    "order_type": "STOP",
                    "confirmation_vehicle": strategy["vehicle"],
                    "entry_ttl_seconds": int(strategy["entry_ttl_seconds"]),
                    "max_hold_seconds": int(strategy["max_hold_seconds"]),
                    **geometry,
                    **shock,
                    **alignment,
                    **pattern,
                    "spread_pips": round(float(spread), 6),
                    "spread_to_m1_atr": round(float(spread) / float(shock["prior_m1_atr_pips"]), 6),
                    "shock_anchor_started_at_utc": anchor["started_at"].isoformat(),
                    "confirmation_started_at_utc": confirmation["started_at"].isoformat(),
                    "quote_timestamp_utc": quote_at.isoformat(),
                    "quote_bid": _price(pair, float(bid)),
                    "quote_ask": _price(pair, float(ask)),
                    "evidence_mode": "PROSPECTIVE_FORWARD_ONLY",
                    "lookahead_used": False,
                    "retrospective_reinterpretation": False,
                    "normal_strategy_override": False,
                    "range_rotation_shock_policy": "RETAIN_EXISTING_VOL_SHOCK_VETO",
                    "truth_source_for_outcome": "OANDA_S5_BID_ASK",
                    "truth_chunk_candle_limit": int(config["truth"]["chunk_candle_limit"]),
                    "execution_authority": "NONE",
                    "broker_http_methods_allowed": ["GET"],
                    "broker_mutation_allowed": False,
                    "external_order_attempts": 0,
                    "external_orders": 0,
                    "shadow_only": True,
                    "live_permission": False,
                    "promotion_allowed": False,
                    "automatic_adoption_allowed": False,
                }
                signals.append(seal(body))
            if not any(row["pair"] == pair for row in signals):
                pair_rejections.setdefault(pair, []).append("NO_CONFIRMED_SHOCK_FOLLOW_PATTERN")
    status = "EMITTED" if signals else blockers[0] if blockers else "NO_CONFIRMED_SIGNAL"
    return seal(
        {
            "contract": "QR_FAST_BOT_SHOCK_FOLLOW_SHADOW_V1",
            "schema_version": 1,
            "generated_at_utc": now.isoformat(),
            "status": status,
            "config_sha256": config_sha256,
            "signals": sorted(signals, key=lambda row: (row["pair"], row["strategy_id"], row["side"])),
            "blockers": sorted(set(blockers)),
            "pair_rejections": {key: sorted(set(value)) for key, value in sorted(pair_rejections.items())},
            "range_rotation_shock_policy": "RETAIN_EXISTING_VOL_SHOCK_VETO",
            "normal_strategy_override": False,
            "historical_diagnostics_count_as_forward_evidence": False,
            "execution_authority": "NONE",
            "broker_http_methods_allowed": ["GET"],
            "broker_mutation_allowed": False,
            "external_order_attempts": 0,
            "external_orders": 0,
            "shadow_only": True,
            "live_permission": False,
            "promotion_allowed": False,
            "automatic_adoption_allowed": False,
        }
    )


def resolve_signal(
    signal: Mapping[str, Any],
    candles: Sequence[S5BidAskCandle],
    *,
    truth_chunk_sha256: Sequence[str],
    resolved_at_utc: datetime,
) -> dict[str, Any]:
    if not sealed_valid(signal, SIGNAL_CONTRACT):
        raise ValueError("shock-follow signal seal mismatch")
    generated = parse_utc(signal["generated_at_utc"])
    resolved = aware_utc(resolved_at_utc)
    ttl = int(signal["entry_ttl_seconds"])
    hold = int(signal["max_hold_seconds"])
    maturity = generated + timedelta(seconds=ttl + hold)
    if resolved < maturity:
        raise ValueError("shock-follow signal is not mature")
    hashes = [str(value) for value in truth_chunk_sha256]
    if not hashes or not all(len(value) == 64 and all(char in "0123456789abcdef" for char in value) for value in hashes):
        raise ValueError("valid S5 truth chunk hashes are required")
    first = _ceil_s5(generated)
    horizon = _floor_s5(maturity)
    timestamps = [row.timestamp_utc for row in candles]
    expected_slots = int((horizon - first).total_seconds() // 5)
    chunk_limit = int(signal["truth_chunk_candle_limit"])
    expected_receipts = math.ceil(expected_slots / chunk_limit)
    if (
        horizon <= first
        or len(hashes) != expected_receipts
        or len(set(timestamps)) != len(timestamps)
        or any(timestamp < first or timestamp >= horizon or _floor_s5(timestamp) != timestamp for timestamp in timestamps)
    ):
        raise ValueError("invalid S5 truth coverage")
    pair = str(signal["pair"])
    side = str(signal["side"])
    factor = float(instrument_pip_factor(pair))
    entry = float(signal["entry"])
    stop = float(signal["stop_loss"])
    target = float(signal["take_profit"])
    stop_pips = float(signal["stop_loss_pips"])
    fill_deadline = generated + timedelta(seconds=ttl)
    fill_at: datetime | None = None
    fill_price: float | None = None
    exit_at: datetime | None = None
    exit_reason = "UNFILLED"
    realized = 0.0
    ambiguous = False
    path: list[S5BidAskCandle] = []
    for candle in sorted(candles, key=lambda row: row.timestamp_utc):
        if fill_at is not None and candle.timestamp_utc >= fill_at + timedelta(seconds=hold):
            break
        newly_filled = False
        if fill_at is None:
            touched = candle.ask_h >= entry if side == "LONG" else candle.bid_l <= entry
            if not touched or candle.timestamp_utc + timedelta(seconds=5) > fill_deadline:
                continue
            fill_at = candle.timestamp_utc
            fill_price = max(entry, candle.ask_o) if side == "LONG" else min(entry, candle.bid_o)
            newly_filled = True
        path.append(candle)
        tp_hit = candle.bid_h >= target if side == "LONG" else candle.ask_l <= target
        sl_hit = candle.bid_l <= stop if side == "LONG" else candle.ask_h >= stop
        if newly_filled and (tp_hit or sl_hit):
            ambiguous = True
            fill_risk = (
                (float(fill_price) - stop) * factor
                if side == "LONG"
                else (stop - float(fill_price)) * factor
            )
            opening = (
                (candle.bid_o - float(fill_price)) * factor
                if side == "LONG"
                else (float(fill_price) - candle.ask_o) * factor
            )
            realized = min(-fill_risk, opening)
            exit_reason = "STOP_LOSS_GAP_AMBIGUOUS_FILL_S5" if realized < -fill_risk - 1e-9 else "STOP_LOSS_AMBIGUOUS_FILL_S5"
            exit_at = candle.timestamp_utc
            break
        if tp_hit and sl_hit:
            ambiguous = True
            realized = -(
                (float(fill_price) - stop) * factor
                if side == "LONG"
                else (stop - float(fill_price)) * factor
            )
            exit_reason = "STOP_LOSS_AMBIGUOUS_SAME_S5"
            exit_at = candle.timestamp_utc
            break
        if sl_hit:
            fill_risk = (
                (float(fill_price) - stop) * factor
                if side == "LONG"
                else (stop - float(fill_price)) * factor
            )
            opening = (
                (candle.bid_o - float(fill_price)) * factor
                if side == "LONG"
                else (float(fill_price) - candle.ask_o) * factor
            )
            realized = min(-fill_risk, opening)
            exit_reason = "STOP_LOSS_GAP" if realized < -fill_risk - 1e-9 else "STOP_LOSS"
            exit_at = candle.timestamp_utc
            break
        if tp_hit:
            realized = (
                (target - float(fill_price)) * factor
                if side == "LONG"
                else (float(fill_price) - target) * factor
            )
            exit_reason = "TAKE_PROFIT"
            exit_at = candle.timestamp_utc
            break
    if fill_at is not None and exit_at is None:
        realized = -(
            (float(fill_price) - stop) * factor
            if side == "LONG"
            else (stop - float(fill_price)) * factor
        )
        exit_reason = "TIMEOUT_FULL_STOP_LOSS"
        exit_at = fill_at + timedelta(seconds=hold)
    mfe = 0.0
    mae = 0.0
    for candle in path:
        if side == "LONG":
            mfe = max(mfe, (candle.bid_h - float(fill_price)) * factor)
            mae = max(mae, (float(fill_price) - candle.bid_l) * factor)
        else:
            mfe = max(mfe, (float(fill_price) - candle.ask_l) * factor)
            mae = max(mae, (candle.ask_h - float(fill_price)) * factor)
    slippage = (
        ((fill_price - entry) * factor if side == "LONG" else (entry - fill_price) * factor)
        if fill_price is not None
        else 0.0
    )
    body = {
        "contract": OUTCOME_CONTRACT,
        "schema_version": 1,
        "scoring_policy": SCORING_POLICY,
        "signal_id": signal["signal_id"],
        "signal_sha256": signal["contract_sha256"],
        "config_sha256": signal["config_sha256"],
        "pair": pair,
        "strategy": signal["strategy_id"],
        "side": side,
        "shock_bucket": signal["shock_bucket"],
        "signal_generated_at_utc": generated.isoformat(),
        "resolved_at_utc": resolved.isoformat(),
        "maturity_at_utc": maturity.isoformat(),
        "filled": fill_at is not None,
        "fill_at_utc": fill_at.isoformat() if fill_at else None,
        "entry_fill_price": _price(pair, fill_price) if fill_price is not None else None,
        "entry_slippage_pips": round(max(0.0, slippage), 6),
        "exit_at_utc": exit_at.isoformat() if exit_at else None,
        "exit_reason": exit_reason,
        "realized_pips": round(realized, 6),
        "after_cost_net_pips": round(realized, 6),
        "mfe_pips": round(max(0.0, mfe), 6),
        "mae_pips": round(max(0.0, mae), 6),
        "ambiguous_same_s5": ambiguous,
        "truth_source": "OANDA_S5_BID_ASK",
        "truth_chunk_sha256": hashes,
        "evidence_mode": "PROSPECTIVE_FORWARD_ONLY",
        "historical_diagnostic": False,
        "execution_authority": "NONE",
        "broker_http_methods_used": ["GET"],
        "broker_mutation": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "shadow_only": True,
        "live_permission": False,
        "promotion_allowed": False,
        "automatic_adoption_allowed": False,
    }
    return seal(body)


def _ceil_s5(value: datetime) -> datetime:
    value = aware_utc(value)
    timestamp = int(value.timestamp())
    remainder = timestamp % 5
    if remainder or value.microsecond:
        timestamp += 5 - remainder if remainder else 5
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def _floor_s5(value: datetime) -> datetime:
    value = aware_utc(value)
    timestamp = int(value.timestamp())
    return datetime.fromtimestamp(timestamp - timestamp % 5, tz=timezone.utc)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (str(row.get("signal_generated_at_utc") or row.get("generated_at_utc")), str(row.get("signal_id"))))
    eligible = [row for row in ordered if row.get("vetoed") is not True]
    filled = [row for row in eligible if row.get("filled") is True]
    values = [float(row.get("after_cost_net_pips", row.get("realized_pips", 0.0))) for row in filled]
    wins = [value for value in values if value > 0.0]
    losses = [value for value in values if value < 0.0]
    consecutive = maximum = 0
    for value in values:
        consecutive = consecutive + 1 if value < 0.0 else 0
        maximum = max(maximum, consecutive)
    tail_count = max(1, math.ceil(len(values) * 0.05)) if values else 0
    tail = sorted(values)[:tail_count]
    gross_loss = abs(sum(losses))
    return {
        "signal_count": len(ordered),
        "eligible_count": len(eligible),
        "vetoed_count": len(ordered) - len(eligible),
        "filled_count": len(filled),
        "fill_rate": round(len(filled) / len(eligible), 6) if eligible else None,
        "win_rate": round(len(wins) / len(filled), 6) if filled else None,
        "net_pips": round(sum(values), 6),
        "profit_factor": round(sum(wins) / gross_loss, 6) if gross_loss else "INF" if wins else None,
        "max_consecutive_losses": maximum,
        "tail_5pct_loss_pips": round(sum(tail), 6),
        "worst_fill_pips": round(min(values), 6) if values else None,
        "mean_mfe_pips": _mean(float(row.get("mfe_pips") or 0.0) for row in filled),
        "mean_mae_pips": _mean(float(row.get("mae_pips") or 0.0) for row in filled),
        "mean_entry_slippage_pips": _mean(float(row.get("entry_slippage_pips") or 0.0) for row in filled),
    }


def _mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return round(statistics.fmean(rows), 6) if rows else None


def _groups(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    return [
        {**dict(zip(keys, identity)), **_aggregate(group)}
        for identity, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0]))
    ]


def build_scorecard(
    *,
    signals: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    corrective_rows: Sequence[Mapping[str, Any]],
    config_sha256: str,
    generated_at_utc: datetime,
) -> dict[str, Any]:
    valid_signals = [row for row in signals if sealed_valid(row, SIGNAL_CONTRACT) and row.get("config_sha256") == config_sha256]
    by_sha = {str(row["contract_sha256"]): row for row in valid_signals}
    valid_outcomes = [
        row
        for row in outcomes
        if sealed_valid(row, OUTCOME_CONTRACT)
        and row.get("config_sha256") == config_sha256
        and str(row.get("signal_sha256")) in by_sha
        and row.get("evidence_mode") == "PROSPECTIVE_FORWARD_ONLY"
    ]
    diagnostic = [
        row
        for row in corrective_rows
        if sealed_valid(row, "QR_FAST_BOT_CORRECTIVE_CHALLENGER_ROW_V1")
        and row.get("arm_id") in DIAGNOSTIC_ARMS
        and row.get("vol_shock") is True
    ]
    diagnostic_comparison = [
        {
            "arm_id": arm,
            "evidence_mode": "HISTORICAL_DIAGNOSTIC_ONLY",
            "counts_as_forward_evidence": False,
            **_aggregate([row for row in diagnostic if row.get("arm_id") == arm]),
        }
        for arm in DIAGNOSTIC_ARMS
    ]
    body = {
        "contract": SCORECARD_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": aware_utc(generated_at_utc).isoformat(),
        "config_sha256": config_sha256,
        "status": "COLLECTING_PROSPECTIVE_SHOCK_EVIDENCE",
        "prospective": {
            "emitted_signal_count": len(valid_signals),
            "resolved_signal_count": len(valid_outcomes),
            "overall": _aggregate(valid_outcomes),
            "pair_strategy_side_shock_bucket": _groups(valid_outcomes, ["pair", "strategy", "side", "shock_bucket"]),
            "by_pair": [
                {"pair": pair, **_aggregate([row for row in valid_outcomes if row.get("pair") == pair])}
                for pair in PAIRS
            ],
            "by_strategy": [
                {
                    "strategy": strategy,
                    **_aggregate([row for row in valid_outcomes if row.get("strategy") == strategy]),
                }
                for strategy in STRATEGIES
            ],
            "tracked_dimensions": ["pair", "strategy", "side", "shock_bucket"],
            "tracked_pairs": list(PAIRS),
            "tracked_strategies": list(STRATEGIES),
        },
        "historical_diagnostic_reference": {
            "source": "EXISTING_CORRECTIVE_CHALLENGER_LEDGER",
            "comparison": diagnostic_comparison,
            "pair_arm": _groups(diagnostic, ["pair", "arm_id"]),
            "counts_as_forward_evidence": False,
            "adoption_eligible": False,
        },
        "profitability_claim_allowed": False,
        "forward_evidence_passed": False,
        "automatic_adoption_allowed": False,
        "promotion_allowed": False,
        "live_permission": False,
        "execution_authority": "NONE",
        "broker_mutation": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "manual_tagless_positions_policy": "NO_TOUCH",
        "existing_tp_sl_policy": "NO_TOUCH",
        "limitations": [
            "NO_PROFITABILITY_CLAIM",
            "PROSPECTIVE_SAMPLE_NOT_YET_SUFFICIENT",
            "HISTORICAL_DIAGNOSTIC_EXCLUDED_FROM_PROMOTION",
            "S5_EXTREMA_CANNOT_ORDER_INTRABAR_MFE_MAE",
            "AUTOMATIC_ADOPTION_AND_LIVE_PERMISSION_ALWAYS_FALSE",
        ],
    }
    return seal(body)


def append_sealed_rows(path: Path, rows: Sequence[Mapping[str, Any]], *, contract: str, identity_key: str) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        seen: set[str] = set()
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not sealed_valid(value, contract):
                raise ValueError(f"invalid {contract} row at line {number}")
            identity = str(value.get(identity_key) or "")
            if not identity or identity in seen:
                raise ValueError(f"duplicate {contract} identity at line {number}")
            seen.add(identity)
        handle.seek(0, os.SEEK_END)
        appended = 0
        for row in rows:
            identity = str(row.get(identity_key) or "")
            if identity and identity not in seen and sealed_valid(row, contract):
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
                seen.add(identity)
                appended += 1
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return appended


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_incremental(
    *,
    pair_charts_path: Path,
    broker_snapshot_path: Path,
    signal_ledger_path: Path,
    outcome_ledger_path: Path,
    scorecard_path: Path,
    corrective_ledger_path: Path,
    config_path: Path,
    max_due: int = 12,
    now: datetime | None = None,
    client: OandaReadOnlyClient | None = None,
    truth_fetcher: Callable[[OandaReadOnlyClient, Mapping[str, Any]], tuple[Sequence[S5BidAskCandle], Sequence[str]]] | None = None,
) -> dict[str, Any]:
    if not 1 <= max_due <= 100:
        raise ValueError("max_due must be inside 1..100")
    clock = aware_utc(now or datetime.now(timezone.utc))
    config, config_sha = load_config(config_path)
    charts = json.loads(pair_charts_path.read_text(encoding="utf-8"))
    snapshot = json.loads(broker_snapshot_path.read_text(encoding="utf-8"))
    shadow = build_shock_follow_shadow(
        pair_charts=charts,
        broker_snapshot=snapshot,
        config=config,
        config_sha256=config_sha,
        now_utc=clock,
    )
    signal_appended = append_sealed_rows(
        signal_ledger_path,
        shadow["signals"],
        contract=SIGNAL_CONTRACT,
        identity_key="signal_id",
    )
    signals = load_jsonl(signal_ledger_path)
    outcomes = load_jsonl(outcome_ledger_path)
    resolved = {str(row.get("signal_sha256")) for row in outcomes if sealed_valid(row, OUTCOME_CONTRACT)}
    due = [
        row
        for row in signals
        if sealed_valid(row, SIGNAL_CONTRACT)
        and row.get("config_sha256") == config_sha
        and str(row.get("contract_sha256")) not in resolved
        and parse_utc(row["generated_at_utc"]) + timedelta(seconds=int(row["entry_ttl_seconds"]) + int(row["max_hold_seconds"])) <= clock
    ]
    due.sort(key=lambda row: (str(row["generated_at_utc"]), str(row["signal_id"])))
    due = due[:max_due]
    broker_read = False
    new_outcomes: list[dict[str, Any]] = []
    if due:
        broker = client or OandaReadOnlyClient()
        broker_read = True

        def default_fetch(value: OandaReadOnlyClient, signal: Mapping[str, Any]) -> tuple[Sequence[S5BidAskCandle], Sequence[str]]:
            generated = parse_utc(signal["generated_at_utc"])
            maturity = generated + timedelta(seconds=int(signal["entry_ttl_seconds"]) + int(signal["max_hold_seconds"]))
            return fetch_frozen_s5_truth(
                value,
                pair=str(signal["pair"]),
                time_from=generated,
                time_to=maturity,
                chunk_candle_limit=int(signal["truth_chunk_candle_limit"]),
            )

        fetch = truth_fetcher or default_fetch
        for signal in due:
            candles, hashes = fetch(broker, signal)
            new_outcomes.append(resolve_signal(signal, candles, truth_chunk_sha256=hashes, resolved_at_utc=clock))
    outcome_appended = append_sealed_rows(
        outcome_ledger_path,
        new_outcomes,
        contract=OUTCOME_CONTRACT,
        identity_key="signal_sha256",
    )
    all_outcomes = load_jsonl(outcome_ledger_path)
    corrective = load_jsonl(corrective_ledger_path)
    scorecard = build_scorecard(
        signals=signals,
        outcomes=all_outcomes,
        corrective_rows=corrective,
        config_sha256=config_sha,
        generated_at_utc=clock,
    )
    write_json_atomic(scorecard_path, scorecard)
    return {
        "contract": RUN_CONTRACT,
        "generated_at_utc": clock.isoformat(),
        "config_sha256": config_sha,
        "shadow_status": shadow["status"],
        "emitted_signal_count": len(shadow["signals"]),
        "signal_ledger_appended": signal_appended,
        "due_signal_count": len(due),
        "outcome_ledger_appended": outcome_appended,
        "broker_read": broker_read,
        "broker_http_methods_used": ["GET"] if broker_read else [],
        "scorecard_path": str(scorecard_path),
        "signal_ledger_path": str(signal_ledger_path),
        "outcome_ledger_path": str(outcome_ledger_path),
        "execution_authority": "NONE",
        "broker_mutation": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "live_permission": False,
        "promotion_allowed": False,
        "automatic_adoption_allowed": False,
    }


__all__ = [
    "CONFIG_CONTRACT",
    "OUTCOME_CONTRACT",
    "PAIRS",
    "SCORECARD_CONTRACT",
    "SIGNAL_CONTRACT",
    "STRATEGIES",
    "build_scorecard",
    "build_shock_follow_shadow",
    "canonical_sha",
    "load_config",
    "market_is_closed",
    "resolve_signal",
    "run_incremental",
    "seal",
    "sealed_valid",
]
