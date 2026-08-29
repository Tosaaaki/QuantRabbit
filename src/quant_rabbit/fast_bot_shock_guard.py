"""Durable, side-relative shock freeze and protective-stop shadow gate.

This module has no broker client and no execution authority.  It transforms an
existing sealed shadow proposal into a separately sealed, zero-authority
candidate set.  The baseline control ledger remains untouched.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_rabbit.instruments import instrument_pip_factor


CONFIG_CONTRACT = "QR_FAST_BOT_SHOCK_GUARD_CONFIG_V1"
STATE_CONTRACT = "QR_FAST_BOT_SHOCK_GUARD_STATE_V1"
RECEIPT_CONTRACT = "QR_FAST_BOT_SHOCK_GUARD_RECEIPT_V1"
SHADOW_CONTRACT = "QR_FAST_BOT_SHOCK_GUARDED_SHADOW_V1"
DECISION_CONTRACT = "QR_FAST_BOT_SHOCK_GUARD_DECISION_V1"
SCORECARD_CONTRACT = "QR_FAST_BOT_SHOCK_GUARD_SCORECARD_V1"
PROTECTIVE_STOP_CONTRACT = "QR_FAST_BOT_PROTECTIVE_STOP_V1"

NORMAL = "NORMAL"
SHOCK_FREEZE = "SHOCK_FREEZE"
CONTINUATION_CONFIRMED = "CONTINUATION_CONFIRMED"
FAILED_CONTINUATION = "FAILED_CONTINUATION"
COOLDOWN = "COOLDOWN"
STATES = frozenset(
    {NORMAL, SHOCK_FREEZE, CONTINUATION_CONFIRMED, FAILED_CONTINUATION, COOLDOWN}
)
BLOCK_ALL_STATES = frozenset({SHOCK_FREEZE, FAILED_CONTINUATION})


def canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def seal(value: Mapping[str, Any], *, key: str = "contract_sha256") -> dict[str, Any]:
    body = {name: item for name, item in value.items() if name != key}
    return {**body, key: canonical_sha(body)}


def sealed_valid(value: Mapping[str, Any], contract: str, *, key: str = "contract_sha256") -> bool:
    body = {name: item for name, item in value.items() if name != key}
    return value.get("contract") == contract and value.get(key) == canonical_sha(body)


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("contract") != CONFIG_CONTRACT:
        raise ValueError("shock guard config contract mismatch")
    authority = value.get("authority") or {}
    if (
        authority.get("execution_authority") != "NONE"
        or authority.get("broker_http_methods_allowed") != ["GET"]
        or authority.get("broker_mutation_allowed") is not False
        or authority.get("live_permission") is not False
        or authority.get("promotion_allowed") is not False
        or authority.get("automatic_reversal_allowed") is not False
    ):
        raise ValueError("shock guard authority boundary mismatch")
    detection = value.get("detection") or {}
    resolution = value.get("resolution") or {}
    if (
        int(detection.get("window_minutes") or 0) != 15
        or float(detection.get("minimum_impulse_pips") or 0.0) != 18.0
        or float(detection.get("minimum_atr_multiple") or 0.0) != 2.0
        or int(resolution.get("freeze_minutes") or 0) != 5
        or float(resolution.get("adverse_atr_fraction") or 0.0) != 0.25
    ):
        raise ValueError("shock guard preregistered central cell drift")
    return value, canonical_sha(value)


def _parse_utc(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone-aware timestamp required")
    return parsed.astimezone(timezone.utc)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timezone-aware timestamp required")
    return value.astimezone(timezone.utc)


def _finite_positive(value: Any, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_state(path: Path, *, pair: str, now: datetime) -> tuple[dict[str, Any], bool]:
    if not path.exists():
        return _normal_state(pair=pair, now=now), True
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _integrity_freeze_state(pair=pair, now=now, reason="STATE_RESTORE_INVALID"), False
    if not isinstance(value, dict) or not sealed_valid(value, STATE_CONTRACT):
        return _integrity_freeze_state(pair=pair, now=now, reason="STATE_RESTORE_INVALID"), False
    if value.get("pair") != pair or value.get("state") not in STATES:
        return _integrity_freeze_state(pair=pair, now=now, reason="STATE_RESTORE_INVALID"), False
    return value, True


def _normal_state(*, pair: str, now: datetime) -> dict[str, Any]:
    return seal(
        {
            "contract": STATE_CONTRACT,
            "schema_version": 1,
            "pair": pair,
            "state": NORMAL,
            "event_id": None,
            "shock_direction": None,
            "observed_at_utc": now.isoformat(),
            "decision_due_at_utc": None,
            "cooldown_until_utc": None,
            "last_transition_at_utc": now.isoformat(),
            "last_complete_m1_at_utc": None,
            "resolution": None,
            "thresholds": {},
            "evidence": {},
            "fail_closed_reason": None,
            "execution_authority": "NONE",
            "broker_mutation_allowed": False,
            "external_order_attempts": 0,
            "external_orders": 0,
        }
    )


def _integrity_freeze_state(*, pair: str, now: datetime, reason: str) -> dict[str, Any]:
    event_id = f"qrs-integrity:{canonical_sha([pair, reason, now.isoformat()])[:24]}"
    return seal(
        {
            "contract": STATE_CONTRACT,
            "schema_version": 1,
            "pair": pair,
            "state": SHOCK_FREEZE,
            "event_id": event_id,
            "shock_direction": None,
            "observed_at_utc": now.isoformat(),
            "decision_due_at_utc": None,
            "cooldown_until_utc": None,
            "last_transition_at_utc": now.isoformat(),
            "last_complete_m1_at_utc": None,
            "resolution": None,
            "thresholds": {},
            "evidence": {"data_integrity": reason},
            "fail_closed_reason": reason,
            "execution_authority": "NONE",
            "broker_mutation_allowed": False,
            "external_order_attempts": 0,
            "external_orders": 0,
        }
    )


def _views(pair_charts: Mapping[str, Any], pair: str) -> dict[str, Mapping[str, Any]]:
    for chart in pair_charts.get("charts") or []:
        if isinstance(chart, Mapping) and str(chart.get("pair") or "").upper() == pair:
            return {
                str(view.get("granularity") or "").upper(): view
                for view in chart.get("views") or []
                if isinstance(view, Mapping)
            }
    return {}


def _complete_candles(view: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in (view or {}).get("recent_candles") or []:
        if not isinstance(item, Mapping) or item.get("complete") is not True:
            continue
        try:
            row = {
                "at": _parse_utc(item.get("t") or item.get("time")),
                "open": float(item["o"]),
                "high": float(item["h"]),
                "low": float(item["l"]),
                "close": float(item["c"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(row[name]) for name in ("open", "high", "low", "close")):
            rows.append(row)
    return sorted(rows, key=lambda item: item["at"])


def _view_atr(view: Mapping[str, Any] | None) -> float | None:
    indicators = (view or {}).get("indicators")
    try:
        value = float((indicators or {}).get("atr_pips"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def _direction(view: Mapping[str, Any] | None) -> str:
    market = (view or {}).get("market_state")
    raw = str((market or {}).get("direction") or "").upper()
    return raw if raw in {"UP", "DOWN"} else "NEUTRAL"


def observe_market(
    *,
    pair_charts: Mapping[str, Any],
    pair: str,
    now_utc: datetime,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build causal M1 shock evidence and separate MTF alignment fields."""

    now = _aware(now_utc)
    pair = pair.upper()
    views = _views(pair_charts, pair)
    rows = _complete_candles(views.get("M1"))
    window = int(config["detection"]["window_minutes"])
    if len(rows) < window + 1:
        return {"valid": False, "reason": "M1_HISTORY_INSUFFICIENT", "pair": pair}
    latest = rows[-1]
    age = (now - (latest["at"] + timedelta(minutes=1))).total_seconds()
    if age < -1.0 or age > float(config["integrity"]["maximum_complete_bar_age_seconds"]):
        return {"valid": False, "reason": "M1_STALE_OR_FUTURE", "pair": pair}
    recent = rows[-(window + 1) :]
    gaps = [
        (right["at"] - left["at"]).total_seconds()
        for left, right in zip(recent, recent[1:])
    ]
    if any(gap != 60.0 for gap in gaps):
        return {"valid": False, "reason": "M1_GAP", "pair": pair}
    atr = _view_atr(views.get(str(config["detection"]["atr_timeframe"]).upper()))
    if atr is None:
        return {"valid": False, "reason": "ATR_UNAVAILABLE", "pair": pair}
    factor = float(instrument_pip_factor(pair))
    impulse = (recent[-1]["close"] - recent[0]["close"]) * factor
    direction = "UP" if impulse > 0.0 else "DOWN" if impulse < 0.0 else "FLAT"
    magnitude = abs(impulse)
    initial_high = max(row["high"] for row in recent)
    initial_low = min(row["low"] for row in recent)
    alignment = {tf: _direction(views.get(tf)) for tf in ("M1", "M5", "M15", "H1")}
    short_reversal = direction in {"UP", "DOWN"} and any(
        alignment[tf] not in {direction, "NEUTRAL"} for tf in ("M1", "M5")
    )
    htf_continuation = direction in {"UP", "DOWN"} and all(
        alignment[tf] == direction for tf in ("M15", "H1")
    )
    return {
        "valid": True,
        "pair": pair,
        "latest_complete_m1_at_utc": latest["at"].isoformat(),
        "window_start_at_utc": recent[0]["at"].isoformat(),
        "window_end_at_utc": latest["at"].isoformat(),
        "impulse_direction": direction,
        "impulse_pips": round(impulse, 6),
        "impulse_magnitude_pips": round(magnitude, 6),
        "atr_pips": round(atr, 6),
        "atr_multiple": round(magnitude / atr, 6),
        "initial_high": initial_high,
        "initial_low": initial_low,
        "latest_close": latest["close"],
        "timeframe_alignment": alignment,
        "short_term_reversal": short_reversal,
        "higher_timeframe_continuation": htf_continuation,
        "post_window": [
            {
                "at": row["at"].isoformat(),
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
            }
            for row in rows
        ],
    }


def _shock_detected(observation: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    return bool(
        observation.get("valid") is True
        and observation.get("impulse_direction") in {"UP", "DOWN"}
        and float(observation["impulse_magnitude_pips"])
        >= float(config["detection"]["minimum_impulse_pips"])
        and float(observation["atr_multiple"])
        >= float(config["detection"]["minimum_atr_multiple"])
    )


def _post_event_rows(observation: Mapping[str, Any], observed_at: datetime) -> list[dict[str, Any]]:
    return [
        row
        for row in observation.get("post_window") or []
        if isinstance(row, Mapping) and _parse_utc(row.get("at")) > observed_at
    ]


def advance_state(
    *,
    prior: Mapping[str, Any],
    observation: Mapping[str, Any],
    now_utc: datetime,
    config: Mapping[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    """Advance the durable state using only observations available at ``now``."""

    now = _aware(now_utc)
    pair = str(prior.get("pair") or observation.get("pair") or "").upper()
    if observation.get("valid") is not True:
        frozen = _integrity_freeze_state(
            pair=pair,
            now=now,
            reason=str(observation.get("reason") or "OBSERVATION_INVALID"),
        )
        frozen["thresholds"] = {"config_sha256": config_sha256}
        return seal({key: value for key, value in frozen.items() if key != "contract_sha256"})

    prior_state = str(prior.get("state") or NORMAL)
    latest = str(observation["latest_complete_m1_at_utc"])
    if prior_state == NORMAL and _shock_detected(observation, config):
        direction = str(observation["impulse_direction"])
        event_id = f"qrs:{canonical_sha([pair, latest, direction, config_sha256])[:24]}"
        decision_due = now + timedelta(minutes=int(config["resolution"]["freeze_minutes"]))
        body = {
            "contract": STATE_CONTRACT,
            "schema_version": 1,
            "pair": pair,
            "state": SHOCK_FREEZE,
            "event_id": event_id,
            "shock_direction": direction,
            "observed_at_utc": now.isoformat(),
            "decision_due_at_utc": decision_due.isoformat(),
            "cooldown_until_utc": None,
            "last_transition_at_utc": now.isoformat(),
            "last_complete_m1_at_utc": latest,
            "resolution": None,
            "thresholds": {
                "config_sha256": config_sha256,
                "window_minutes": config["detection"]["window_minutes"],
                "minimum_impulse_pips": config["detection"]["minimum_impulse_pips"],
                "minimum_atr_multiple": config["detection"]["minimum_atr_multiple"],
                "freeze_minutes": config["resolution"]["freeze_minutes"],
                "adverse_atr_fraction": config["resolution"]["adverse_atr_fraction"],
            },
            "evidence": dict(observation),
            "fail_closed_reason": None,
            "execution_authority": "NONE",
            "broker_mutation_allowed": False,
            "external_order_attempts": 0,
            "external_orders": 0,
        }
        return seal(body)

    if prior_state == SHOCK_FREEZE:
        if prior.get("fail_closed_reason"):
            # Clean causal history releases an integrity-only freeze to a
            # bounded cooldown only after a later complete bar proves forward
            # progress.  The first clean read after restart remains frozen.
            prior_latest = prior.get("last_complete_m1_at_utc")
            if not prior_latest or latest == prior_latest:
                return seal(
                    {
                        **{k: v for k, v in prior.items() if k != "contract_sha256"},
                        "last_complete_m1_at_utc": latest,
                    }
                )
            cooldown = now + timedelta(minutes=int(config["resolution"]["cooldown_minutes"]))
            return _transition(prior, state=COOLDOWN, now=now, cooldown=cooldown, resolution="DATA_RECOVERED")
        due = _parse_utc(prior.get("decision_due_at_utc"))
        if now < due:
            return seal({**{k: v for k, v in prior.items() if k != "contract_sha256"}, "last_complete_m1_at_utc": latest})
        observed = _parse_utc(prior.get("observed_at_utc"))
        post = _post_event_rows(observation, observed)
        minimum = int(config["resolution"]["minimum_post_event_complete_m1_bars"])
        if len(post) < minimum:
            return seal({**{k: v for k, v in prior.items() if k != "contract_sha256"}, "last_complete_m1_at_utc": latest})
        direction = str(prior.get("shock_direction"))
        initial = prior.get("evidence") or {}
        atr = float(initial.get("atr_pips") or 0.0)
        adverse_threshold = atr * float(config["resolution"]["adverse_atr_fraction"])
        factor = float(instrument_pip_factor(pair))
        if direction == "UP":
            new_extreme = max(float(row["high"]) for row in post) > float(initial["initial_high"])
            adverse = (float(initial["initial_high"]) - min(float(row["low"]) for row in post)) * factor
        else:
            new_extreme = min(float(row["low"]) for row in post) < float(initial["initial_low"])
            adverse = (max(float(row["high"]) for row in post) - float(initial["initial_low"])) * factor
        resolution_evidence = {
            "post_event_complete_m1_bars": len(post),
            "new_shock_direction_extreme": new_extreme,
            "adverse_excursion_pips": round(adverse, 6),
            "adverse_threshold_pips": round(adverse_threshold, 6),
            "short_term_reversal": observation.get("short_term_reversal"),
            "higher_timeframe_continuation": observation.get("higher_timeframe_continuation"),
            "timeframe_alignment": observation.get("timeframe_alignment"),
        }
        if not new_extreme and adverse >= adverse_threshold:
            return _classified(prior, state=FAILED_CONTINUATION, now=now, evidence=resolution_evidence, config=config)
        if new_extreme and adverse < adverse_threshold:
            return _classified(prior, state=CONTINUATION_CONFIRMED, now=now, evidence=resolution_evidence, config=config)
        return seal(
            {
                **{k: v for k, v in prior.items() if k != "contract_sha256"},
                "last_complete_m1_at_utc": latest,
                "evidence": {**dict(initial), "resolution_pending": resolution_evidence},
                "fail_closed_reason": "FIVE_MINUTE_BOUNDARY_AMBIGUOUS",
            }
        )

    if prior_state in {CONTINUATION_CONFIRMED, FAILED_CONTINUATION}:
        cooldown = _parse_utc(prior.get("cooldown_until_utc"))
        return _transition(prior, state=COOLDOWN, now=now, cooldown=cooldown, resolution=str(prior.get("resolution")))

    if prior_state == COOLDOWN:
        cooldown = _parse_utc(prior.get("cooldown_until_utc"))
        if now >= cooldown:
            return _normal_state(pair=pair, now=now)
        return seal({**{k: v for k, v in prior.items() if k != "contract_sha256"}, "last_complete_m1_at_utc": latest})

    return _integrity_freeze_state(pair=pair, now=now, reason="STATE_UNKNOWN")


def _classified(
    prior: Mapping[str, Any],
    *,
    state: str,
    now: datetime,
    evidence: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    cooldown = now + timedelta(minutes=int(config["resolution"]["cooldown_minutes"]))
    body = {
        **{key: value for key, value in prior.items() if key != "contract_sha256"},
        "state": state,
        "last_transition_at_utc": now.isoformat(),
        "cooldown_until_utc": cooldown.isoformat(),
        "resolution": state,
        "evidence": {**dict(prior.get("evidence") or {}), "five_minute_resolution": dict(evidence)},
        "fail_closed_reason": None,
    }
    return seal(body)


def _transition(
    prior: Mapping[str, Any],
    *,
    state: str,
    now: datetime,
    cooldown: datetime,
    resolution: str,
) -> dict[str, Any]:
    return seal(
        {
            **{key: value for key, value in prior.items() if key != "contract_sha256"},
            "state": state,
            "last_transition_at_utc": now.isoformat(),
            "cooldown_until_utc": cooldown.isoformat(),
            "resolution": resolution,
            "fail_closed_reason": None,
        }
    )


def protective_stop_candidates(
    *,
    pair: str,
    side: str,
    entry: float,
    atr_pips: float,
    spread_pips: float,
    recent_swing_price: float,
    observed_at_utc: datetime,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return the bounded, preregistered SL geometry comparison set."""

    factor = float(instrument_pip_factor(pair))
    entry = _finite_positive(entry, "entry")
    atr = _finite_positive(atr_pips, "atr_pips")
    spread = _finite_positive(spread_pips, "spread_pips")
    side = side.upper()
    if side not in {"LONG", "SHORT"}:
        raise ValueError("side must be LONG or SHORT")
    swing_distance = (
        (entry - float(recent_swing_price)) * factor
        if side == "LONG"
        else (float(recent_swing_price) - entry) * factor
    )
    swing_width = max(float(config["protective_stop"]["minimum_stop_pips"]), swing_distance + spread)
    widths = [
        ("FIXED_3_2", 3.2),
        ("ATR_1_0", atr),
        ("ATR_1_5", atr * 1.5),
        ("ATR_2_0", atr * 2.0),
        ("SWING_SPREAD_BUFFER", swing_width),
        ("CONSERVATIVE_ATR_SWING", max(atr * 1.5, swing_width)),
    ]
    rows: list[dict[str, Any]] = []
    for geometry, width in widths:
        width = round(max(float(config["protective_stop"]["minimum_stop_pips"]), width), 6)
        price = entry - width / factor if side == "LONG" else entry + width / factor
        rows.append(
            seal(
                {
                    "contract": PROTECTIVE_STOP_CONTRACT,
                    "schema_version": 1,
                    "geometry_id": geometry,
                    "pair": pair,
                    "side": side,
                    "entry": entry,
                    "stop_loss": price,
                    "stop_loss_pips": width,
                    "observed_at_utc": _aware(observed_at_utc).isoformat(),
                    "maximum_age_seconds": int(config["protective_stop"]["maximum_age_seconds"]),
                    "attached_required": True,
                    "guaranteed": False,
                    "gap_slippage_ledger_required": True,
                    "widen_during_shock_allowed": False,
                    "remove_after_entry_allowed": False,
                    "shadow_only": True,
                    "live_permission": False,
                }
            )
        )
    return rows


def size_units_for_stop(
    *, max_loss_jpy: float, stop_loss_pips: float, pip_value_jpy_per_unit: float
) -> int:
    """Inverse-risk size: a wider stop can never increase units."""

    budget = _finite_positive(max_loss_jpy, "max_loss_jpy")
    width = _finite_positive(stop_loss_pips, "stop_loss_pips")
    pip_value = _finite_positive(pip_value_jpy_per_unit, "pip_value_jpy_per_unit")
    return max(0, math.floor(budget / (width * pip_value)))


def validate_protective_stop(
    signal: Mapping[str, Any], *, now_utc: datetime
) -> tuple[bool, str | None, float | None]:
    """Validate attached SL presence, seal, freshness and side-relative price."""

    stop = signal.get("protective_stop")
    if not isinstance(stop, Mapping) or not sealed_valid(stop, PROTECTIVE_STOP_CONTRACT):
        return False, "PROTECTIVE_STOP_MISSING_OR_UNSEALED", None
    try:
        now = _aware(now_utc)
        observed = _parse_utc(stop.get("observed_at_utc"))
        maximum_age = int(stop.get("maximum_age_seconds"))
        entry = _finite_positive(signal.get("entry"), "entry")
        stop_price = _finite_positive(signal.get("stop_loss"), "stop_loss")
        width = _finite_positive(signal.get("stop_loss_pips"), "stop_loss_pips")
        factor = float(instrument_pip_factor(str(signal.get("pair") or "")))
        side = str(signal.get("side") or "").upper()
    except (TypeError, ValueError):
        return False, "PROTECTIVE_STOP_VALUE_INVALID", None
    if observed > now or (now - observed).total_seconds() > maximum_age:
        return False, "PROTECTIVE_STOP_STALE_OR_FUTURE", None
    if stop.get("attached_required") is not True or signal.get("attached_stop_loss_required") is not True:
        return False, "PROTECTIVE_STOP_NOT_REQUIRED", None
    if stop.get("guaranteed") is not False or stop.get("gap_slippage_ledger_required") is not True:
        return False, "PROTECTIVE_STOP_GAP_CONTRACT_INVALID", None
    if float(stop.get("stop_loss") or 0.0) != stop_price or float(stop.get("stop_loss_pips") or 0.0) != width:
        return False, "PROTECTIVE_STOP_SIGNAL_MISMATCH", None
    actual = (entry - stop_price) * factor if side == "LONG" else (stop_price - entry) * factor
    if side not in {"LONG", "SHORT"} or actual <= 0.0 or abs(actual - width) > 0.01:
        return False, "PROTECTIVE_STOP_PRICE_INVALID", None
    return True, None, width


def _receipt(state: Mapping[str, Any], *, now: datetime, config_sha256: str) -> dict[str, Any]:
    expiry_raw = state.get("cooldown_until_utc") or state.get("decision_due_at_utc")
    expiry = _parse_utc(expiry_raw) if expiry_raw else now + timedelta(seconds=90)
    if expiry <= now:
        expiry = now + timedelta(seconds=90)
    evidence = state.get("evidence") or {}
    resolution = evidence.get("five_minute_resolution") or evidence.get("resolution_pending") or {}
    return seal(
        {
            "contract": RECEIPT_CONTRACT,
            "schema_version": 1,
            "event_id": state.get("event_id"),
            "state": state.get("state"),
            "resolution": state.get("resolution"),
            "shock_direction": state.get("shock_direction"),
            "observed_at_utc": state.get("observed_at_utc"),
            "expires_at_utc": expiry.isoformat(),
            "config_sha256": config_sha256,
            "fail_closed_reason": state.get("fail_closed_reason"),
            "short_term_reversal": resolution.get("short_term_reversal", evidence.get("short_term_reversal")),
            "higher_timeframe_continuation": resolution.get(
                "higher_timeframe_continuation", evidence.get("higher_timeframe_continuation")
            ),
            "timeframe_alignment": resolution.get("timeframe_alignment", evidence.get("timeframe_alignment", {})),
            "automatic_reversal_allowed": False,
            "execution_authority": "NONE",
            "broker_mutation_allowed": False,
            "external_order_attempts": 0,
            "external_orders": 0,
        }
    )


def validate_shock_guard_receipt(
    signal: Mapping[str, Any], *, now_utc: datetime
) -> tuple[bool, str | None]:
    receipt = signal.get("shock_guard")
    if not isinstance(receipt, Mapping) or not sealed_valid(receipt, RECEIPT_CONTRACT):
        return False, "SHOCK_GUARD_RECEIPT_MISSING_OR_UNSEALED"
    try:
        if _aware(now_utc) > _parse_utc(receipt.get("expires_at_utc")):
            return False, "SHOCK_GUARD_RECEIPT_STALE"
    except ValueError:
        return False, "SHOCK_GUARD_RECEIPT_TIME_INVALID"
    state = str(receipt.get("state") or "")
    if state in BLOCK_ALL_STATES or receipt.get("fail_closed_reason"):
        return False, f"SHOCK_GUARD_{state or 'FAIL_CLOSED'}"
    if state == COOLDOWN and receipt.get("resolution") != CONTINUATION_CONFIRMED:
        return False, "SHOCK_GUARD_COOLDOWN_BLOCKS_ENTRY"
    if state == CONTINUATION_CONFIRMED:
        if str(signal.get("method") or "") != "TREND_CONTINUATION":
            return False, "SHOCK_GUARD_CONTINUATION_TREND_ONLY"
        signal_direction = "UP" if str(signal.get("side") or "") == "LONG" else "DOWN"
        if signal_direction != receipt.get("shock_direction") or receipt.get("higher_timeframe_continuation") is not True:
            return False, "SHOCK_GUARD_CONTINUATION_ALIGNMENT_REQUIRED"
    return True, None


def guard_shadow(
    *,
    shadow: Mapping[str, Any],
    state: Mapping[str, Any],
    pair_charts: Mapping[str, Any],
    config: Mapping[str, Any],
    config_sha256: str,
    now_utc: datetime,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    now = _aware(now_utc)
    receipt = _receipt(state, now=now, config_sha256=config_sha256)
    views = _views(pair_charts, str(state.get("pair") or "EUR_USD"))
    m1 = _complete_candles(views.get("M1"))
    spread_default = float(config["protective_stop"]["fallback_spread_pips"])
    selected_geometry = str(config["protective_stop"]["selected_shadow_geometry"])
    admitted: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for raw_signal in shadow.get("signals") or []:
        if not isinstance(raw_signal, Mapping):
            continue
        signal = dict(raw_signal)
        pair = str(signal.get("pair") or "").upper()
        if pair != state.get("pair"):
            # The initial guard is pair-scoped.  Other pairs retain the same
            # mandatory protective-stop contract without inheriting EUR/USD state.
            local_receipt = seal(
                {
                    **{key: value for key, value in receipt.items() if key != "contract_sha256"},
                    "event_id": None,
                    "state": NORMAL,
                    "resolution": None,
                    "shock_direction": None,
                    "fail_closed_reason": None,
                }
            )
        else:
            local_receipt = receipt
        side = str(signal.get("side") or "").upper()
        swing = (
            min(row["low"] for row in m1[-10:]) if side == "LONG" and m1
            else max(row["high"] for row in m1[-10:]) if m1
            else float(signal.get("stop_loss") or signal.get("entry"))
        )
        candidates = protective_stop_candidates(
            pair=pair,
            side=side,
            entry=float(signal["entry"]),
            atr_pips=float(signal.get("m5_atr_pips") or 0.0),
            spread_pips=float(signal.get("spread_pips") or spread_default),
            recent_swing_price=float(swing),
            observed_at_utc=now,
            config=config,
        )
        selected = next(row for row in candidates if row["geometry_id"] == selected_geometry)
        signal.update(
            stop_loss=selected["stop_loss"],
            stop_loss_pips=selected["stop_loss_pips"],
            protective_stop=selected,
            protective_stop_candidates=candidates,
            shock_guard=local_receipt,
        )
        signal["reward_risk"] = round(float(signal["take_profit_pips"]) / float(signal["stop_loss_pips"]), 6)
        body = {key: value for key, value in signal.items() if key != "signal_sha256"}
        signal["signal_sha256"] = canonical_sha(body)
        allowed, reason = validate_shock_guard_receipt(signal, now_utc=now)
        stop_ok, stop_reason, _ = validate_protective_stop(signal, now_utc=now)
        if not stop_ok:
            allowed = False
            reason = stop_reason
        decision = seal(
            {
                "contract": DECISION_CONTRACT,
                "schema_version": 1,
                "decision_id": f"qrgd:{canonical_sha([signal.get('signal_id'), local_receipt.get('contract_sha256')])[:24]}",
                "signal_id": signal.get("signal_id"),
                "signal_sha256": signal.get("signal_sha256"),
                "pair": pair,
                "side": side,
                "method": signal.get("method"),
                "state": local_receipt.get("state"),
                "entry_allowed": allowed,
                "rejection_reason": reason,
                "drain_intent": (
                    {
                        "scope": "BOT_OWNED_ONLY",
                        "fraction": float(config["inventory"]["paper_shadow_drain_fraction"]),
                        "execution_scope": "PAPER_SHADOW_ONLY",
                        "manual_tagless_policy": "NO_TOUCH",
                    }
                    if pair == state.get("pair") and local_receipt.get("state") in BLOCK_ALL_STATES
                    else None
                ),
                "countertrend_candidate": {
                    "ledger_scope": "SHADOW_ONLY_SEPARATE",
                    "promotion_allowed": False,
                    "automatic_reversal_allowed": False,
                },
                "protective_stop_geometry": selected_geometry,
                "protective_stop_pips": selected["stop_loss_pips"],
                "gap_slippage_ledger_required": True,
                "execution_authority": "NONE",
                "broker_mutation_allowed": False,
                "external_order_attempts": 0,
                "external_orders": 0,
            }
        )
        decisions.append(decision)
        if allowed:
            admitted.append(signal)
    body = {
        "contract": SHADOW_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": now.isoformat(),
        "status": "EMITTED" if admitted else "ENTRY_FROZEN_OR_NO_SIGNAL",
        "source_shadow_sha256": shadow.get("contract_sha256"),
        "shock_guard_state": state.get("state"),
        "shock_guard_event_id": state.get("event_id"),
        "signals": admitted,
        "decision_count": len(decisions),
        "entry_rejection_count": sum(1 for row in decisions if not row["entry_allowed"]),
        "manual_tagless_policy": "NO_TOUCH",
        "existing_tp_sl_policy": "NO_TOUCH",
        "execution_authority": "NONE",
        "shadow_only": True,
        "live_permission": False,
        "promotion_allowed": False,
        "broker_mutation_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
    }
    return seal(body), decisions


def _append_once(path: Path, rows: Sequence[Mapping[str, Any]], *, id_key: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = path.with_suffix(path.suffix + ".lock")
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing: set[str] = set()
        if path.exists():
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        value = json.loads(line)
                        existing.add(str(value.get(id_key) or ""))
        appended = 0
        with path.open("a", encoding="utf-8") as target:
            for row in rows:
                identity = str(row.get(id_key) or "")
                if identity and identity not in existing:
                    target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    existing.add(identity)
                    appended += 1
            target.flush()
            os.fsync(target.fileno())
    return appended


def run_guard_cycle(
    *,
    pair_charts: Mapping[str, Any],
    shadow: Mapping[str, Any],
    config: Mapping[str, Any],
    config_sha256: str,
    state_path: Path,
    decision_ledger_path: Path,
    scorecard_path: Path,
    output_path: Path,
    now_utc: datetime,
    pair: str = "EUR_USD",
) -> dict[str, Any]:
    now = _aware(now_utc)
    prior, restored = _load_state(state_path, pair=pair, now=now)
    observation = observe_market(pair_charts=pair_charts, pair=pair, now_utc=now, config=config)
    state = advance_state(
        prior=prior,
        observation=observation,
        now_utc=now,
        config=config,
        config_sha256=config_sha256,
    )
    _atomic_json(state_path, state)
    guarded, decisions = guard_shadow(
        shadow=shadow,
        state=state,
        pair_charts=pair_charts,
        config=config,
        config_sha256=config_sha256,
        now_utc=now,
    )
    _atomic_json(output_path, guarded)
    appended = _append_once(decision_ledger_path, decisions, id_key="decision_id")
    total = 0
    rejected = 0
    if decision_ledger_path.exists():
        with decision_ledger_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    total += 1
                    rejected += json.loads(line).get("entry_allowed") is False
    scorecard = seal(
        {
            "contract": SCORECARD_CONTRACT,
            "schema_version": 1,
            "generated_at_utc": now.isoformat(),
            "decision_count": total,
            "entry_rejection_count": rejected,
            "entry_rejection_rate": round(rejected / total, 6) if total else 0.0,
            "current_state": state.get("state"),
            "current_event_id": state.get("event_id"),
            "restart_restore_valid": restored,
            "selected_protective_stop_geometry": config["protective_stop"]["selected_shadow_geometry"],
            "automatic_reversal_allowed": False,
            "execution_authority": "NONE",
            "broker_mutation_allowed": False,
            "external_order_attempts": 0,
            "external_orders": 0,
        }
    )
    _atomic_json(scorecard_path, scorecard)
    return {
        "status": guarded["status"],
        "state": state.get("state"),
        "event_id": state.get("event_id"),
        "signal_count": len(guarded.get("signals") or []),
        "decision_count": len(decisions),
        "decision_ledger_appended": appended,
        "shadow_output": str(output_path),
        "state_path": str(state_path),
        "decision_ledger_path": str(decision_ledger_path),
        "scorecard_path": str(scorecard_path),
        "execution_authority": "NONE",
        "broker_mutation": False,
        "external_order_attempts": 0,
        "external_orders": 0,
    }
