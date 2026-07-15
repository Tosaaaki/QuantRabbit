"""Frozen forward-shadow contract for the causal technical forecast candidate.

The research selector found a positive 24-hour directional cohort and an
exploratory passive LIMIT geometry.  The exploratory holdout was inspected
before this candidate was chosen, so none of that history is promotion proof.
This module starts a new, append-only forward cohort without creating order
intents or broker permissions.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_rabbit.instruments import instrument_pip_factor


FORWARD_SHADOW_CONTRACT = "QR_TECHNICAL_FORECAST_FORWARD_SHADOW_V1"
FORWARD_CANDIDATE_CONTRACT = "QR_TECHNICAL_FORECAST_FORWARD_CANDIDATE_V1"


def load_forward_candidate(path: Path) -> dict[str, Any]:
    """Load and validate one frozen non-live candidate."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read forward candidate: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("forward candidate must be a JSON object")
    issues = validate_forward_candidate(payload)
    if issues:
        raise ValueError("invalid forward candidate: " + "; ".join(issues))
    return payload


def validate_forward_candidate(value: object) -> tuple[str, ...]:
    """Return fail-closed candidate contract issues."""

    if not isinstance(value, Mapping):
        return ("candidate is not an object",)
    issues: list[str] = []
    if value.get("contract") != FORWARD_CANDIDATE_CONTRACT:
        issues.append("contract mismatch")
    if value.get("schema_version") != 1:
        issues.append("schema_version must be 1")
    if value.get("status") != "LOCKED_FORWARD_SHADOW":
        issues.append("status must be LOCKED_FORWARD_SHADOW")
    if value.get("shadow_enabled") is not True:
        issues.append("shadow_enabled must be true")
    if value.get("live_order_enabled") is not False:
        issues.append("live_order_enabled must be false")
    if value.get("promotion_allowed") is not False:
        issues.append("promotion_allowed must be false")
    for key in ("locked_at_utc", "forward_evaluation_not_before_utc"):
        if _parse_utc(value.get(key)) is None:
            issues.append(f"{key} must be an aware UTC timestamp")
    disclosure = value.get("selection_disclosure")
    if not isinstance(disclosure, Mapping) or disclosure.get(
        "initial_holdout_inspected_before_lock"
    ) is not True:
        issues.append("initial holdout inspection must be disclosed")
    selector = value.get("selector")
    if not isinstance(selector, Mapping):
        issues.append("selector must be an object")
    else:
        expected = {
            "horizon_min": 1440,
            "feature": "return_12",
            "feature_timeframe": "M5",
            "orientation": "DIRECT",
            "schedule_hour_utc": 23,
            "schedule_minute_utc": 55,
            "collection_lateness_max_seconds": 60,
        }
        for key, required in expected.items():
            if selector.get(key) != required:
                issues.append(f"selector.{key} must be {required!r}")
        if not _positive(selector.get("spread_cap_pips")):
            issues.append("selector.spread_cap_pips must be positive")
        fraction = _finite(selector.get("opportunity_fraction"))
        if fraction is None or not 0.0 < fraction <= 1.0:
            issues.append("selector.opportunity_fraction must be inside (0, 1]")
    vehicle = value.get("vehicle")
    if not isinstance(vehicle, Mapping):
        issues.append("vehicle must be an object")
    else:
        expected_vehicle = {
            "order_type": "LIMIT",
            "entry_reference": "JOIN_EXECUTABLE_NEAR_SIDE",
            "entry_ttl_min": 5,
            "max_hold_min": 1435,
            "take_profit_pips": 15,
            "stop_loss_pips": 30,
            "market_entry_allowed": False,
            "time_close_allowed": False,
        }
        for key, required in expected_vehicle.items():
            if vehicle.get(key) != required:
                issues.append(f"vehicle.{key} must be {required!r}")
    blockers = value.get("promotion_blockers")
    if not isinstance(blockers, list) or "NEW_FORWARD_COHORT_REQUIRED" not in blockers:
        issues.append("NEW_FORWARD_COHORT_REQUIRED blocker is mandatory")
    return tuple(issues)


def forward_collection_window(
    candidate: Mapping[str, Any],
    observed_at_utc: datetime,
) -> dict[str, Any]:
    """Return the latest scheduled collection window and whether it is open."""

    issues = validate_forward_candidate(candidate)
    if issues:
        raise ValueError("invalid forward candidate: " + "; ".join(issues))
    observed = _aware_utc(observed_at_utc)
    selector = candidate["selector"]
    terminal = observed.replace(
        hour=int(selector["schedule_hour_utc"]),
        minute=int(selector["schedule_minute_utc"]),
        second=0,
        microsecond=0,
    )
    decision_at = terminal + timedelta(minutes=5)
    if decision_at > observed:
        terminal -= timedelta(days=1)
        decision_at -= timedelta(days=1)
    closes_at = decision_at + timedelta(
        seconds=int(selector["collection_lateness_max_seconds"])
    )
    not_before = _parse_utc(candidate.get("forward_evaluation_not_before_utc"))
    assert not_before is not None
    if decision_at < not_before:
        status = "BEFORE_FORWARD_LOCK"
    elif observed > closes_at:
        status = "DECISION_WINDOW_MISSED"
    else:
        status = "OPEN"
    return {
        "status": status,
        "terminal_m5_timestamp_utc": terminal.isoformat(),
        "decision_at_utc": decision_at.isoformat(),
        "collection_closes_at_utc": closes_at.isoformat(),
        "open": status == "OPEN",
    }


def build_forward_shadow(
    candidate: Mapping[str, Any],
    pair_charts: Mapping[str, Any],
    quotes: Mapping[str, Any],
    *,
    candidate_sha256: str,
    observed_at_utc: datetime | None = None,
) -> dict[str, Any]:
    """Build one decision-time, read-only shadow snapshot.

    A signal is emitted only after the complete 23:55 UTC M5 candle.  The
    strongest 30% of spread-eligible pairs are selected by absolute 12-bar
    return and retain its sign (DIRECT orientation).
    """

    issues = validate_forward_candidate(candidate)
    if issues:
        raise ValueError("invalid forward candidate: " + "; ".join(issues))
    observed = _aware_utc(observed_at_utc or datetime.now(timezone.utc))
    chart_generated = _parse_utc(pair_charts.get("generated_at_utc"))
    raw_charts = pair_charts.get("charts")
    base = {
        "contract": FORWARD_SHADOW_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": observed.isoformat(),
        "candidate_sha256": _sha256_text(candidate_sha256),
        "shadow_only": True,
        "live_ready": False,
        "promotion_allowed": False,
        "broker_mutation_allowed": False,
        "order_intents": [],
        "signals": [],
    }
    if chart_generated is None or not isinstance(raw_charts, list):
        return _seal({**base, "status": "INVALID_PAIR_CHARTS", "issues": ["missing valid generated_at_utc or charts"]})

    selector = candidate["selector"]
    window = forward_collection_window(candidate, observed)
    terminal = _parse_utc(window["terminal_m5_timestamp_utc"])
    decision_at = _parse_utc(window["decision_at_utc"])
    assert terminal is not None and decision_at is not None
    decision_id = _stable_digest(
        {
            "candidate_sha256": candidate_sha256,
            "terminal_m5_timestamp_utc": terminal.isoformat(),
        }
    )
    base.update(
        {
            "terminal_m5_timestamp_utc": terminal.isoformat(),
            "decision_at_utc": decision_at.isoformat(),
            "collection_closes_at_utc": window["collection_closes_at_utc"],
            "decision_id": decision_id,
        }
    )
    if window["status"] != "OPEN":
        return _seal({**base, "status": window["status"]})
    rows: list[dict[str, Any]] = []
    row_issues: list[str] = []
    for chart in raw_charts:
        row, issue = _chart_row(chart, quotes, terminal=terminal)
        if issue is not None:
            row_issues.append(issue)
            continue
        assert row is not None
        rows.append(row)

    public_rows = [_public_row(row) for row in rows]
    common = {
        **base,
        "pair_charts_generated_at_utc": chart_generated.isoformat(),
        "eligible_input_rows": public_rows,
        "input_issues": sorted(row_issues),
    }
    if not rows:
        return _seal({**common, "status": "NO_VALID_TECHNICAL_ROWS"})
    not_before = _parse_utc(candidate.get("forward_evaluation_not_before_utc"))
    assert not_before is not None
    if decision_at < not_before:
        return _seal({**common, "status": "BEFORE_FORWARD_LOCK"})
    if observed < decision_at:
        return _seal({**common, "status": "DECISION_CANDLE_NOT_COMPLETE"})
    if chart_generated < decision_at:
        return _seal({**common, "status": "PAIR_CHARTS_PREDATE_DECISION"})

    spread_cap = float(selector["spread_cap_pips"])
    eligible = [
        row
        for row in rows
        if row["spread_pips"] <= spread_cap
        and row["quote_timestamp"] >= decision_at
        and row["quote_timestamp"] <= observed + timedelta(seconds=60)
    ]
    if not eligible:
        return _seal({**common, "status": "NO_SPREAD_FRESH_ELIGIBLE_PAIRS"})
    fraction = float(selector["opportunity_fraction"])
    count = max(1, int(math.ceil(len(eligible) * fraction)))
    ranked = sorted(
        eligible,
        key=lambda row: (abs(row["technical_score_pips"]), row["pair"]),
        reverse=True,
    )[:count]
    vehicle = candidate["vehicle"]
    signals = [
        _signal(
            row,
            decision_id=decision_id,
            decision_at=decision_at,
            vehicle=vehicle,
        )
        for row in ranked
    ]
    return _seal(
        {
            **common,
            "status": "EMITTED",
            "eligible_pair_count": len(eligible),
            "selected_pair_count": len(signals),
            "signals": signals,
        }
    )


def append_shadow_once(path: Path, shadow: Mapping[str, Any]) -> bool:
    """Append an emitted decision once, deduplicated under an advisory lock."""

    if shadow.get("status") != "EMITTED":
        return False
    decision_id = _sha256_text(shadow.get("decision_id"))
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            with path.open(encoding="utf-8") as current:
                for line in current:
                    try:
                        previous = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if previous.get("decision_id") == decision_id:
                        return False
        with path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(dict(shadow)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return True


def write_shadow_atomic(path: Path, shadow: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(shadow), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _chart_row(
    chart: object,
    quotes: Mapping[str, Any],
    *,
    terminal: datetime,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(chart, Mapping):
        return None, "chart is not an object"
    pair = str(chart.get("pair") or "").upper()
    views = chart.get("views")
    if not pair or not isinstance(views, list):
        return None, f"{pair or 'UNKNOWN'}: missing views"
    view = next(
        (
            item
            for item in views
            if isinstance(item, Mapping) and item.get("granularity") == "M5"
        ),
        None,
    )
    candles = view.get("recent_candles") if isinstance(view, Mapping) else None
    if not isinstance(candles, list) or len(candles) < 13:
        return None, f"{pair}: fewer than 13 M5 candles"
    parsed_all: list[tuple[datetime, float]] = []
    for candle in candles:
        if not isinstance(candle, Mapping) or candle.get("complete") is not True:
            return None, f"{pair}: incomplete M5 lookback"
        timestamp = _parse_utc(candle.get("t"))
        close = _finite(candle.get("c"))
        if timestamp is None or close is None or close <= 0.0:
            return None, f"{pair}: invalid M5 candle"
        parsed_all.append((timestamp, close))
    parsed_all.sort(key=lambda item: item[0])
    terminal_indices = [
        index for index, item in enumerate(parsed_all) if item[0] == terminal
    ]
    if len(terminal_indices) != 1 or terminal_indices[0] < 12:
        return None, f"{pair}: scheduled M5 terminal or 12-bar lookback unavailable"
    terminal_index = terminal_indices[0]
    parsed = parsed_all[terminal_index - 12 : terminal_index + 1]
    if any(
        right[0] - left[0] != timedelta(minutes=5)
        for left, right in zip(parsed, parsed[1:])
    ):
        return None, f"{pair}: non-contiguous M5 return_12 lookback"
    quote = quotes.get(pair)
    bid = _finite(_field(quote, "bid"))
    ask = _finite(_field(quote, "ask"))
    quote_timestamp = _parse_utc(_field(quote, "timestamp_utc"))
    if bid is None or ask is None or quote_timestamp is None or not 0.0 < bid < ask:
        return None, f"{pair}: invalid current bid/ask"
    pip_factor = instrument_pip_factor(pair)
    score = (parsed[-1][1] - parsed[0][1]) * pip_factor
    return (
        {
            "pair": pair,
            "terminal_time": terminal,
            "technical_score_pips": score,
            "bid": bid,
            "ask": ask,
            "spread_pips": (ask - bid) * pip_factor,
            "quote_timestamp": quote_timestamp,
            "pip_factor": pip_factor,
        },
        None,
    )


def _public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pair": row["pair"],
        "terminal_m5_timestamp_utc": row["terminal_time"].isoformat(),
        "technical_score_pips": round(float(row["technical_score_pips"]), 6),
        "spread_pips": round(float(row["spread_pips"]), 6),
        "quote_timestamp_utc": row["quote_timestamp"].isoformat(),
    }


def _signal(
    row: Mapping[str, Any],
    *,
    decision_id: str,
    decision_at: datetime,
    vehicle: Mapping[str, Any],
) -> dict[str, Any]:
    direction = "UP" if float(row["technical_score_pips"]) >= 0.0 else "DOWN"
    side = "LONG" if direction == "UP" else "SHORT"
    entry = float(row["bid"] if direction == "UP" else row["ask"])
    pip_size = 1.0 / float(row["pip_factor"])
    reward = float(vehicle["take_profit_pips"])
    risk = float(vehicle["stop_loss_pips"])
    target = entry + reward * pip_size if direction == "UP" else entry - reward * pip_size
    stop = entry - risk * pip_size if direction == "UP" else entry + risk * pip_size
    body = {
        "decision_id": decision_id,
        "pair": row["pair"],
        "predicted_direction": direction,
        "side": side,
        "horizon_min": 1440,
        "selected_rule": "return_12",
        "orientation": "DIRECT",
        "technical_score_pips": round(float(row["technical_score_pips"]), 6),
        "spread_pips_at_decision": round(float(row["spread_pips"]), 6),
        "quote_timestamp_utc": row["quote_timestamp"].isoformat(),
        "order_type": "LIMIT",
        "entry_price": entry,
        "take_profit_price": target,
        "stop_loss_price": stop,
        "entry_ttl_min": int(vehicle["entry_ttl_min"]),
        "entry_expires_at_utc": (
            decision_at + timedelta(minutes=float(vehicle["entry_ttl_min"]))
        ).isoformat(),
        "max_hold_min": int(vehicle["max_hold_min"]),
        "take_profit_pips": reward,
        "stop_loss_pips": risk,
        "shadow_only": True,
        "live_ready": False,
        "broker_mutation_allowed": False,
    }
    return {**body, "signal_sha256": _stable_digest(body)}


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    return {**body, "shadow_sha256": _stable_digest(body)}


def _field(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _parse_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        try:
            return _aware_utc(value)
        except ValueError:
            return None
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive(value: Any) -> bool:
    parsed = _finite(value)
    return parsed is not None and parsed > 0.0


def _sha256_text(value: Any) -> str:
    text = str(value or "")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError("candidate_sha256 must be lowercase SHA-256")
    return text


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "FORWARD_CANDIDATE_CONTRACT",
    "FORWARD_SHADOW_CONTRACT",
    "append_shadow_once",
    "build_forward_shadow",
    "forward_collection_window",
    "load_forward_candidate",
    "validate_forward_candidate",
    "write_shadow_atomic",
]
