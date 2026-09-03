"""Sealed, GET-only corrective counterfactuals for fast-bot shadow outcomes.

The module never mutates broker state or the baseline ledgers.  Every arm is
scored on the exact OANDA S5 BID/ASK truth chunk already sealed by the primary
outcome, and is persisted under a content-addressed configuration identity.
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

from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.fast_bot_entry_edge import entry_edge_snapshot_valid
from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle
from quant_rabbit.technical_forecast_forward_truth import fetch_frozen_s5_truth


CONFIG_CONTRACT = "QR_FAST_BOT_CORRECTIVE_CHALLENGER_CONFIG_V1"
CONFIG_CONTRACT_V2 = "QR_FAST_BOT_CORRECTIVE_CHALLENGER_CONFIG_V2"
CONFIG_CONTRACT_V3 = "QR_FAST_BOT_CORRECTIVE_CHALLENGER_CONFIG_V3"
CONFIG_CONTRACT_V4 = "QR_FAST_BOT_CORRECTIVE_CHALLENGER_CONFIG_V4"
CONFIG_CONTRACTS = {
    CONFIG_CONTRACT,
    CONFIG_CONTRACT_V2,
    CONFIG_CONTRACT_V3,
    CONFIG_CONTRACT_V4,
}
ROW_CONTRACT = "QR_FAST_BOT_CORRECTIVE_CHALLENGER_ROW_V1"
SCORECARD_CONTRACT = "QR_FAST_BOT_CORRECTIVE_CHALLENGER_SCORECARD_V1"
PREREGISTRATION_CONTRACT = "QR_FAST_BOT_SHADOW_CHANGE_PREREGISTRATION_V1"
SCORING_POLICY = "QR_FAST_BOT_CORRECTIVE_SAME_S5_CONSERVATIVE_V1"
ARM_ORDER = (
    "BASELINE",
    "VOL_SHOCK_VETO",
    "ATR_NORMALIZED_GEOMETRY",
    "COMBINED",
    "LANE_COOLDOWN",
    "EURUSD_RANGE_ROTATION_EXCLUDE",
)
ARM_ORDER_V3 = (*ARM_ORDER, "M1_TRIGGERED_ONLY")
ARM_ORDER_V4 = (*ARM_ORDER_V3, "CAUSAL_ENTRY_EDGE_ONLY")
STOP_REASONS = {
    "STOP_LOSS",
    "STOP_LOSS_GAP",
    "STOP_LOSS_AMBIGUOUS_FILL_S5",
    "STOP_LOSS_GAP_AMBIGUOUS_FILL_S5",
    "STOP_LOSS_AMBIGUOUS_SAME_S5",
    "STOP_LOSS_GAP_AMBIGUOUS_SAME_S5",
}


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
    stored = str(value.get("contract_sha256") or "")
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return value.get("contract") == contract and stored == canonical_sha(body)


def parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("aware timestamp is required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("contract") not in CONFIG_CONTRACTS:
        raise ValueError("corrective challenger config contract mismatch")
    expected_arm_order = arm_order_for_config(config)
    if tuple(str(row.get("arm_id")) for row in config.get("arms", [])) != expected_arm_order:
        raise ValueError("corrective challenger arm set or order mismatch")
    authority = config.get("authority") or {}
    if (
        authority.get("execution_authority") != "NONE"
        or authority.get("broker_http_methods_allowed") != ["GET"]
        or authority.get("broker_mutation_allowed") is not False
        or authority.get("live_permission") is not False
        or authority.get("promotion_allowed") is not False
        or authority.get("automatic_parameter_change_allowed") is not False
        or config.get("lookahead_policy") != "ENTRY_TIME_FEATURES_ONLY"
    ):
        raise ValueError("corrective challenger authority boundary mismatch")
    geometry = config.get("geometry") or {}
    if float(geometry.get("reward_risk_minimum") or 0.0) < 1.0:
        raise ValueError("ATR geometry must have reward/risk >= 1")
    inventory = config.get("inventory") or {}
    # 990 seconds is the immutable proposal lifecycle already emitted by the
    # fast bot: 90 seconds to enter plus a 900-second maximum hold.  This is an
    # observation-only challenger contract, not a newly tuned market number.
    if (
        tuple(str(value) for value in inventory.get("lane_fields", ()))
        != ("pair", "side", "method", "horizon_lane")
        or int(inventory.get("reservation_seconds") or 0) != 990
        or inventory.get("reservation_source") != "ENTRY_TTL_90_PLUS_MAX_HOLD_900"
        or inventory.get("selection_policy") != "FIRST_ELIGIBLE_SIGNAL_RESERVES_LANE"
    ):
        raise ValueError("corrective challenger lane reservation contract mismatch")
    preregistration = config.get("preregistration") or {}
    success = preregistration.get("success_criteria") or {}
    # These floors intentionally mirror the already-published fast-bot
    # forward-admission contract. Changing them after observing this cohort
    # would invalidate the preregistration; a later experiment needs a new
    # hypothesis version and configuration identity.
    common_preregistration_invalid = (
        preregistration.get("contract") != PREREGISTRATION_CONTRACT
        or preregistration.get("change_scope")
        != "RESIDENT_SHADOW_CHALLENGER_ONLY"
        or preregistration.get("automatic_adoption_allowed") is not False
        or preregistration.get("once_only_activation_identity")
        != "SOURCE_BUNDLE_SHA256_PLUS_CONFIG_SHA256"
        or int(success.get("minimum_resolved_forward_fills") or 0) != 100
        or int(success.get("minimum_active_forward_days") or 0) != 10
        or float(success.get("minimum_profit_factor") or 0.0) != 1.25
        or float(success.get("minimum_net_delta_pips_exclusive", -1.0)) != 0.0
        or float(success.get("minimum_daily_delta_lower_pips_exclusive", -1.0))
        != 0.0
        or int(success.get("maximum_loss_streak_delta_vs_baseline", -1)) != 0
        or not preregistration.get("effective_conditions")
        or not preregistration.get("adverse_conditions")
        or not preregistration.get("stop_conditions")
    )
    if common_preregistration_invalid:
        raise ValueError("corrective challenger preregistration mismatch")
    if config.get("contract") == CONFIG_CONTRACT:
        if (
            preregistration.get("hypothesis_id")
            != "same-lane-cooldown-loss-compression"
            or preregistration.get("hypothesis_version") != 1
            or preregistration.get("target_arm_id") != "LANE_COOLDOWN"
        ):
            raise ValueError("corrective challenger V1 preregistration mismatch")
    elif config.get("contract") == CONFIG_CONTRACT_V2:
        early_stop = preregistration.get("early_futility_stop") or {}
        selection = preregistration.get("selection_evidence") or {}
        cutoff = parse_utc(preregistration.get("eligibility_cutoff_utc"))
        if (
            config.get("schema_version") != 2
            or preregistration.get("hypothesis_id")
            != "atr-normalized-geometry-loss-compression"
            or preregistration.get("hypothesis_version") != 2
            or preregistration.get("target_arm_id") != "ATR_NORMALIZED_GEOMETRY"
            or preregistration.get("single_factor_change")
            != "GEOMETRY_ONLY_ATR_RR1_BOUNDED"
            or cutoff.second != 0
            or cutoff.microsecond != 0
            or int(early_stop.get("minimum_target_fills") or 0) != 10
            or early_stop.get("requires_all")
            != [
                "TARGET_NET_PIPS_NOT_ABOVE_BASELINE",
                "TARGET_PROFIT_FACTOR_NOT_ABOVE_BASELINE",
            ]
            or selection.get("selection_basis")
            != "LOSS_COMPRESSION_HYPOTHESIS_ONLY"
            or selection.get("profitability_claim_allowed") is not False
            or not sealed_hash_fields_valid(
                selection,
                (
                    "observed_config_sha256",
                    "observed_scorecard_sha256",
                ),
            )
            or not git_commit_valid(selection.get("observed_release_commit"))
        ):
            raise ValueError("corrective challenger V2 preregistration mismatch")
    elif config.get("contract") == CONFIG_CONTRACT_V3:
        early_stop = preregistration.get("early_futility_stop") or {}
        selection = preregistration.get("selection_evidence") or {}
        cutoff = parse_utc(preregistration.get("eligibility_cutoff_utc"))
        if (
            config.get("schema_version") != 3
            or preregistration.get("hypothesis_id")
            != "execution-m1-triggered-entry-quality"
            or preregistration.get("hypothesis_version") != 3
            or preregistration.get("target_arm_id") != "M1_TRIGGERED_ONLY"
            or preregistration.get("single_factor_change")
            != "ENTRY_CONFIRMATION_ONLY_M1_TRIGGERED"
            or cutoff.second != 0
            or cutoff.microsecond != 0
            or int(early_stop.get("minimum_target_fills") or 0) != 10
            or early_stop.get("requires_all")
            != [
                "TARGET_NET_PIPS_NOT_ABOVE_BASELINE",
                "TARGET_PROFIT_FACTOR_NOT_ABOVE_BASELINE",
            ]
            or selection.get("selection_basis")
            != "FORWARD_ENTRY_TIMING_REPAIR_HYPOTHESIS_ONLY"
            or selection.get("profitability_claim_allowed") is not False
            or not sealed_hash_fields_valid(
                selection,
                (
                    "observed_config_sha256",
                    "observed_scorecard_sha256",
                ),
            )
            or not git_commit_valid(selection.get("observed_release_commit"))
        ):
            raise ValueError("corrective challenger V3 preregistration mismatch")
    else:
        early_stop = preregistration.get("early_futility_stop") or {}
        selection = preregistration.get("selection_evidence") or {}
        cutoff = parse_utc(preregistration.get("eligibility_cutoff_utc"))
        if (
            config.get("schema_version") != 4
            or preregistration.get("hypothesis_id")
            != "method-aware-causal-entry-edge"
            or preregistration.get("hypothesis_version") != 4
            or preregistration.get("target_arm_id") != "CAUSAL_ENTRY_EDGE_ONLY"
            or preregistration.get("single_factor_change")
            != "ENTRY_SELECTION_ONLY_METHOD_AWARE_CAUSAL_EDGE"
            or cutoff.second != 0
            or cutoff.microsecond != 0
            or int(early_stop.get("minimum_target_fills") or 0) != 10
            or early_stop.get("requires_all")
            != [
                "TARGET_NET_PIPS_NOT_ABOVE_BASELINE",
                "TARGET_PROFIT_FACTOR_NOT_ABOVE_BASELINE",
            ]
            or selection.get("selection_basis")
            != "FORWARD_ROOT_ENTRY_DECISION_REPAIR_ONLY"
            or selection.get("profitability_claim_allowed") is not False
            or not sealed_hash_fields_valid(
                selection,
                (
                    "observed_config_sha256",
                    "observed_scorecard_sha256",
                ),
            )
            or not git_commit_valid(selection.get("observed_release_commit"))
        ):
            raise ValueError("corrective challenger V4 preregistration mismatch")
    return config, canonical_sha(config)


def arm_order_for_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    contract = config.get("contract")
    if contract == CONFIG_CONTRACT_V4:
        return ARM_ORDER_V4
    if contract == CONFIG_CONTRACT_V3:
        return ARM_ORDER_V3
    return ARM_ORDER


def sealed_hash_fields_valid(value: Mapping[str, Any], fields: Sequence[str]) -> bool:
    return all(
        isinstance(value.get(field), str)
        and len(str(value[field])) == 64
        and all(character in "0123456789abcdef" for character in str(value[field]))
        for field in fields
    )


def git_commit_valid(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{number}")
            rows.append(value)
    return rows


def _time_bucket(at: datetime, seconds: int) -> str:
    epoch = int(at.timestamp())
    start = datetime.fromtimestamp(epoch - epoch % seconds, tz=timezone.utc)
    return start.isoformat()


def causal_features(
    signals: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build strictly-prior ATR features, collapsing same-time duplicate arms."""

    shock = config["shock"]
    window = int(shock["prior_window_seconds"])
    minimum = int(shock["minimum_prior_observations"])
    absolute = float(shock["absolute_m5_atr_pips"])
    ratio_limit = float(shock["causal_atr_ratio"])
    bucket_seconds = int(shock["time_bucket_seconds"])
    by_pair_time: dict[str, dict[datetime, list[float]]] = {}
    for signal in signals:
        pair = str(signal.get("pair") or "")
        signal_id = str(signal.get("signal_id") or "")
        if not pair or not signal_id or signal.get("m5_atr_pips") is None:
            continue
        at = parse_utc(signal["generated_at_utc"])
        by_pair_time.setdefault(pair, {}).setdefault(at, []).append(float(signal["m5_atr_pips"]))
    histories: dict[str, list[tuple[datetime, float]]] = {}
    for pair, by_time in by_pair_time.items():
        histories[pair] = [
            (at, statistics.median(values))
            for at, values in sorted(by_time.items())
        ]
    result: dict[str, dict[str, Any]] = {}
    for signal in signals:
        signal_id = str(signal.get("signal_id") or "")
        pair = str(signal.get("pair") or "")
        if not signal_id or not pair or signal.get("m5_atr_pips") is None:
            continue
        at = parse_utc(signal["generated_at_utc"])
        atr = float(signal["m5_atr_pips"])
        prior = [
            value
            for prior_at, value in histories.get(pair, [])
            if prior_at < at and 0.0 < (at - prior_at).total_seconds() <= window
        ]
        median = statistics.median(prior) if len(prior) >= minimum else None
        ratio = atr / median if median and median > 0.0 else None
        absolute_hit = atr >= absolute
        ratio_hit = ratio is not None and ratio >= ratio_limit
        is_shock = absolute_hit or ratio_hit
        result[signal_id] = {
            "prior_atr_observations": len(prior),
            "prior_atr_median_pips": round(median, 6) if median is not None else None,
            "causal_atr_ratio": round(ratio, 6) if ratio is not None else None,
            "vol_shock": is_shock,
            "vol_shock_reasons": [
                reason
                for reason, active in (
                    ("ABSOLUTE_M5_ATR", absolute_hit),
                    ("CAUSAL_ATR_RATIO", ratio_hit),
                )
                if active
            ],
            "rapid_time_bucket_utc": _time_bucket(at, bucket_seconds) if is_shock else "NON_SHOCK",
        }
    inventory = config.get("inventory") or {}
    lane_fields = tuple(str(value) for value in inventory.get("lane_fields", ()))
    reservation_seconds = int(inventory.get("reservation_seconds") or 0)
    if not lane_fields or reservation_seconds <= 0:
        raise ValueError("causal inventory reservation is required")
    reserved_until: dict[tuple[str, ...], datetime] = {}
    reserving_signal: dict[tuple[str, ...], str] = {}
    for signal in sorted(
        signals,
        key=lambda row: (parse_utc(row["generated_at_utc"]), str(row.get("signal_id") or "")),
    ):
        signal_id = str(signal.get("signal_id") or "")
        if not signal_id or signal_id not in result:
            continue
        generated = parse_utc(signal["generated_at_utc"])
        lane = tuple(str(signal.get(field) or "") for field in lane_fields)
        if any(not value for value in lane):
            raise ValueError(f"lane identity is incomplete for {signal_id}")
        until = reserved_until.get(lane)
        blocked = until is not None and generated < until
        result[signal_id].update(
            {
                "lane_identity": list(lane),
                "lane_cooldown_veto": blocked,
                "lane_reserved_by_signal_id": reserving_signal.get(lane) if blocked else None,
                "lane_reservation_until_utc": until.isoformat() if blocked and until else None,
            }
        )
        if not blocked:
            reserved_until[lane] = generated + timedelta(seconds=reservation_seconds)
            reserving_signal[lane] = signal_id
    return result


def _price_geometry(
    signal: Mapping[str, Any],
    *,
    stop_loss_pips: float,
    take_profit_pips: float,
) -> tuple[float, float]:
    entry = float(signal["entry"])
    pair = str(signal["pair"])
    factor = float(instrument_pip_factor(str(signal["pair"])))
    precision = 3 if pair.endswith("_JPY") else 5
    if signal["side"] == "LONG":
        return (
            round(entry - stop_loss_pips / factor, precision),
            round(entry + take_profit_pips / factor, precision),
        )
    if signal["side"] == "SHORT":
        return (
            round(entry + stop_loss_pips / factor, precision),
            round(entry - take_profit_pips / factor, precision),
        )
    raise ValueError("side must be LONG or SHORT")


def score_path(
    signal: Mapping[str, Any],
    candles: Sequence[S5BidAskCandle],
    *,
    stop_loss_pips: float,
    take_profit_pips: float,
) -> dict[str, Any]:
    generated = parse_utc(signal["generated_at_utc"])
    fill_deadline = generated + timedelta(seconds=int(signal["entry_ttl_seconds"]))
    hold = int(signal["max_hold_seconds"])
    side = str(signal["side"])
    entry = float(signal["entry"])
    factor = float(instrument_pip_factor(str(signal["pair"])))
    stop, target = _price_geometry(
        signal,
        stop_loss_pips=stop_loss_pips,
        take_profit_pips=take_profit_pips,
    )
    fill_at: datetime | None = None
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
            touched = candle.ask_l <= entry if side == "LONG" else candle.bid_h >= entry
            if not touched or candle.timestamp_utc + timedelta(seconds=5) > fill_deadline:
                continue
            fill_at = candle.timestamp_utc
            newly_filled = True
        path.append(candle)
        if side == "LONG":
            target_hit = candle.bid_h >= target
            stop_hit = candle.bid_l <= stop
        else:
            target_hit = candle.ask_l <= target
            stop_hit = candle.ask_h >= stop
        if newly_filled and (target_hit or stop_hit):
            ambiguous = True
            opening = (candle.bid_o - entry) * factor if side == "LONG" else (entry - candle.ask_o) * factor
            realized = min(-stop_loss_pips, opening)
            gap = realized < -stop_loss_pips - 1e-9
            exit_reason = "STOP_LOSS_GAP_AMBIGUOUS_FILL_S5" if gap else "STOP_LOSS_AMBIGUOUS_FILL_S5"
            exit_at = candle.timestamp_utc
            break
        if target_hit and stop_hit:
            ambiguous = True
            exit_reason = "STOP_LOSS_AMBIGUOUS_SAME_S5"
            realized = -stop_loss_pips
            exit_at = candle.timestamp_utc
            break
        if stop_hit:
            opening = (candle.bid_o - entry) * factor if side == "LONG" else (entry - candle.ask_o) * factor
            realized = min(-stop_loss_pips, opening)
            exit_reason = "STOP_LOSS_GAP" if realized < -stop_loss_pips - 1e-9 else "STOP_LOSS"
            exit_at = candle.timestamp_utc
            break
        if target_hit:
            exit_reason = "TAKE_PROFIT"
            realized = take_profit_pips
            exit_at = candle.timestamp_utc
            break
    if fill_at is not None and exit_at is None:
        exit_reason = "HORIZON_FULL_STOP_LOSS"
        realized = -stop_loss_pips
        exit_at = fill_at + timedelta(seconds=hold)
    mfe = 0.0
    mae = 0.0
    for candle in path:
        if side == "LONG":
            mfe = max(mfe, (candle.bid_h - entry) * factor)
            mae = max(mae, (entry - candle.bid_l) * factor)
        else:
            mfe = max(mfe, (entry - candle.ask_l) * factor)
            mae = max(mae, (candle.ask_h - entry) * factor)
    return {
        "filled": fill_at is not None,
        "fill_at_utc": fill_at.isoformat() if fill_at else None,
        "exit_at_utc": exit_at.isoformat() if exit_at else None,
        "exit_reason": exit_reason,
        "realized_pips": round(realized, 6),
        "mfe_pips": round(max(0.0, mfe), 6),
        "mae_pips": round(max(0.0, mae), 6),
        "time_to_stop_seconds": (
            (exit_at - fill_at).total_seconds()
            if fill_at is not None and exit_at is not None and exit_reason in STOP_REASONS
            else None
        ),
        "ambiguous_same_s5": ambiguous,
    }


def _worst_lane(signal: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    lane = config["worst_lane"]
    return all(str(signal.get(key)) == str(lane[key]) for key in ("pair", "method", "side"))


def _eurusd_range(signal: Mapping[str, Any]) -> bool:
    return signal.get("pair") == "EUR_USD" and signal.get("method") == "RANGE_ROTATION"


def arm_specs(
    signal: Mapping[str, Any],
    features: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    atr = float(signal["m5_atr_pips"])
    emitted_sl = float(signal["stop_loss_pips"])
    emitted_tp = float(signal["take_profit_pips"])
    geometry = config["geometry"]
    atr_sl = min(
        float(geometry["stop_cap_pips"]),
        max(float(geometry["stop_floor_pips"]), atr * float(geometry["atr_stop_multiplier"])),
    )
    atr_tp = max(atr_sl * float(geometry["reward_risk_minimum"]), atr_sl)
    weight = 1.0
    for row in sorted(
        config["volatility_lot_throttle"],
        key=lambda item: float(item["minimum_m5_atr_pips"]),
        reverse=True,
    ):
        if atr >= float(row["minimum_m5_atr_pips"]):
            weight = float(row["unit_weight"])
            break
    shock = bool(features.get("vol_shock"))
    worst = _worst_lane(signal, config)
    lane_blocked = bool(features.get("lane_cooldown_veto"))
    arms = [
        {"arm_id": "BASELINE", "vetoed": False, "veto_reason": None, "stop_loss_pips": emitted_sl, "take_profit_pips": emitted_tp, "unit_weight": 1.0},
        {"arm_id": "VOL_SHOCK_VETO", "vetoed": shock, "veto_reason": "VOL_SHOCK" if shock else None, "stop_loss_pips": emitted_sl, "take_profit_pips": emitted_tp, "unit_weight": 1.0},
        {"arm_id": "ATR_NORMALIZED_GEOMETRY", "vetoed": False, "veto_reason": None, "stop_loss_pips": round(atr_sl, 6), "take_profit_pips": round(atr_tp, 6), "unit_weight": 1.0},
        {"arm_id": "COMBINED", "vetoed": shock or worst, "veto_reason": "VOL_SHOCK" if shock else "WORST_LANE" if worst else None, "stop_loss_pips": round(atr_sl, 6), "take_profit_pips": round(atr_tp, 6), "unit_weight": weight},
        {"arm_id": "LANE_COOLDOWN", "vetoed": lane_blocked, "veto_reason": "LANE_RESERVED" if lane_blocked else None, "stop_loss_pips": emitted_sl, "take_profit_pips": emitted_tp, "unit_weight": 1.0},
        {"arm_id": "EURUSD_RANGE_ROTATION_EXCLUDE", "vetoed": _eurusd_range(signal), "veto_reason": "EURUSD_RANGE_ROTATION" if _eurusd_range(signal) else None, "stop_loss_pips": emitted_sl, "take_profit_pips": emitted_tp, "unit_weight": 1.0},
    ]
    if config.get("contract") in {CONFIG_CONTRACT_V3, CONFIG_CONTRACT_V4}:
        confirmation = signal.get("entry_confirmation")
        confirmed = bool(
            isinstance(confirmation, Mapping)
            and confirmation.get("contract") == "QR_FAST_BOT_ENTRY_CONFIRMATION_V1"
            and confirmation.get("policy") == "EXECUTION_M1_MUST_BE_TRIGGERED"
            and confirmation.get("m1_readiness") == "TRIGGERED"
            and confirmation.get("m1_triggered") is True
        )
        arms.append(
            {
                "arm_id": "M1_TRIGGERED_ONLY",
                "vetoed": not confirmed,
                "veto_reason": None if confirmed else "M1_NOT_TRIGGERED_AT_EMISSION",
                "stop_loss_pips": emitted_sl,
                "take_profit_pips": emitted_tp,
                "unit_weight": 1.0,
            }
        )
    if config.get("contract") == CONFIG_CONTRACT_V4:
        snapshot = signal.get("entry_edge_snapshot")
        accepted = bool(
            entry_edge_snapshot_valid(snapshot)
            and snapshot.get("accepted") is True
            and snapshot.get("lookahead_used") is False
            and snapshot.get("outcome_fields_used") == []
            and snapshot.get("execution_authority") == "NONE"
            and snapshot.get("live_permission") is False
        )
        arms.append(
            {
                "arm_id": "CAUSAL_ENTRY_EDGE_ONLY",
                "vetoed": not accepted,
                "veto_reason": None if accepted else "ENTRY_EDGE_NOT_CONFIRMED_AT_EMISSION",
                "stop_loss_pips": emitted_sl,
                "take_profit_pips": emitted_tp,
                "unit_weight": 1.0,
            }
        )
    return arms


def _atr_bucket(value: float) -> str:
    if value < 3.0:
        return "ATR_LT_3"
    if value < 4.0:
        return "ATR_3_TO_LT_4"
    if value < 5.0:
        return "ATR_4_TO_LT_5"
    return "ATR_GE_5"


def _regime_bucket(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "REGIME_UNKNOWN"
    return "REGIME_NEGATIVE" if score < 0 else "REGIME_POSITIVE" if score > 0 else "REGIME_ZERO"


def build_rows(
    signal: Mapping[str, Any],
    outcome: Mapping[str, Any],
    candles: Sequence[S5BidAskCandle],
    truth_hashes: Sequence[str],
    features: Mapping[str, Any],
    config: Mapping[str, Any],
    config_sha256: str,
    *,
    evaluated_at_utc: datetime,
) -> list[dict[str, Any]]:
    expected = [str(value) for value in outcome.get("truth_chunk_sha256", [])]
    if list(truth_hashes) != expected:
        raise ValueError(f"truth chunk hash mismatch for {signal.get('signal_id')}")
    result: list[dict[str, Any]] = []
    for spec in arm_specs(signal, features, config):
        if spec["vetoed"]:
            scored = {
                "filled": False,
                "fill_at_utc": None,
                "exit_at_utc": None,
                "exit_reason": "ENTRY_VETO",
                "realized_pips": 0.0,
                "mfe_pips": 0.0,
                "mae_pips": 0.0,
                "time_to_stop_seconds": None,
                "ambiguous_same_s5": False,
            }
        else:
            scored = score_path(
                signal,
                candles,
                stop_loss_pips=float(spec["stop_loss_pips"]),
                take_profit_pips=float(spec["take_profit_pips"]),
            )
        if spec["arm_id"] == "BASELINE":
            if (
                scored["filled"] is not outcome.get("filled")
                or scored["exit_reason"] != outcome.get("exit_reason")
                or not math.isclose(float(scored["realized_pips"]), float(outcome.get("realized_pips") or 0.0), abs_tol=1e-6)
            ):
                raise ValueError(f"baseline replay mismatch for {signal.get('signal_id')}")
        generated = parse_utc(signal["generated_at_utc"])
        body = {
            "contract": ROW_CONTRACT,
            "schema_version": 1,
            "scoring_policy": SCORING_POLICY,
            "config_sha256": config_sha256,
            "arm_id": spec["arm_id"],
            "row_identity": canonical_sha([signal["signal_sha256"], config_sha256, spec["arm_id"]]),
            "signal_id": signal["signal_id"],
            "signal_sha256": signal["signal_sha256"],
            "outcome_sha256": outcome.get("contract_sha256"),
            "evaluated_at_utc": evaluated_at_utc.astimezone(timezone.utc).isoformat(),
            "generated_at_utc": generated.isoformat(),
            "pair": signal["pair"],
            "strategy": signal["method"],
            "side": signal["side"],
            "m5_atr_pips": round(float(signal["m5_atr_pips"]), 6),
            "atr_bucket": _atr_bucket(float(signal["m5_atr_pips"])),
            "spread_pips": round(float(signal["spread_pips"]), 6),
            "spread_bucket": f"{round(float(signal['spread_pips']), 1):.1f}P",
            "regime_score": signal.get("regime_score"),
            "regime_bucket": _regime_bucket(signal.get("regime_score")),
            **dict(features),
            **spec,
            **scored,
            "after_cost_net_pips": round(float(scored["realized_pips"]) * float(spec["unit_weight"]), 6),
            "truth_source": "OANDA_S5_BID_ASK",
            "truth_chunk_sha256": list(truth_hashes),
            "truth_hash_match": True,
            "leftover_inventory": 0,
            "execution_authority": "NONE",
            "broker_http_methods_used": ["GET"],
            "broker_mutation": False,
            "external_order_attempts": 0,
            "external_orders": 0,
            "live_permission": False,
            "automatic_parameter_change_allowed": False,
        }
        result.append(seal(body))
    return result


def _mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return round(statistics.fmean(rows), 6) if rows else None


def _median(values: Iterable[float]) -> float | None:
    rows = list(values)
    return round(statistics.median(rows), 6) if rows else None


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (str(row.get("generated_at_utc")), str(row.get("signal_id"))))
    eligible = [row for row in ordered if row.get("vetoed") is not True]
    filled = [row for row in eligible if row.get("filled") is True]
    raw = [float(row["realized_pips"]) for row in filled]
    net = [float(row["after_cost_net_pips"]) for row in filled]
    wins = [value for value in net if value > 0.0]
    losses = [value for value in net if value < 0.0]
    gross_loss = abs(sum(losses))
    consecutive = 0
    maximum_consecutive = 0
    for value in net:
        consecutive = consecutive + 1 if value < 0.0 else 0
        maximum_consecutive = max(maximum_consecutive, consecutive)
    tail_count = max(1, math.ceil(len(net) * 0.05)) if net else 0
    tail = sorted(net)[:tail_count]
    return {
        "signal_count": len(ordered),
        "eligible_count": len(eligible),
        "vetoed_count": len(ordered) - len(eligible),
        "filled_count": len(filled),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(filled), 6) if filled else None,
        "raw_net_pips": round(sum(raw), 6),
        "after_cost_net_pips": round(sum(net), 6),
        "profit_factor": round(sum(wins) / gross_loss, 6) if gross_loss else "INF" if wins else None,
        "max_consecutive_losses": maximum_consecutive,
        "tail_5pct_count": tail_count,
        "tail_5pct_loss_pips": round(sum(tail), 6),
        "worst_fill_pips": round(min(net), 6) if net else None,
        "mean_mfe_pips": _mean(float(row["mfe_pips"]) for row in filled),
        "mean_mae_pips": _mean(float(row["mae_pips"]) for row in filled),
        "mean_time_to_stop_seconds": _mean(float(row["time_to_stop_seconds"]) for row in filled if row.get("time_to_stop_seconds") is not None),
        "median_time_to_stop_seconds": _median(float(row["time_to_stop_seconds"]) for row in filled if row.get("time_to_stop_seconds") is not None),
        "leftover_inventory": sum(int(row.get("leftover_inventory") or 0) for row in filled),
    }


def _group(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    return [
        {**dict(zip(keys, key)), **aggregate(group)}
        for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0]))
    ]


def build_scorecard(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    generated_at_utc: datetime,
) -> dict[str, Any]:
    matching = [row for row in rows if row.get("config_sha256") == config_sha256]
    for row in matching:
        if not sealed_valid(row, ROW_CONTRACT):
            raise ValueError("challenger ledger row seal mismatch")
    baseline = [row for row in matching if row.get("arm_id") == "BASELINE"]
    arm_order = arm_order_for_config(config)
    comparison = [
        {"arm_id": arm, **aggregate([row for row in matching if row.get("arm_id") == arm])}
        for arm in arm_order
    ]
    evidence_bearing = [row for row in comparison if int(row["filled_count"]) > 0]
    best_by_profit_factor = (
        max(
            evidence_bearing,
            key=lambda row: (_profit_factor_value(row["profit_factor"]), float(row["after_cost_net_pips"])),
        )["arm_id"]
        if evidence_bearing
        else None
    )
    best_by_net = (
        max(evidence_bearing, key=lambda row: float(row["after_cost_net_pips"]))["arm_id"]
        if evidence_bearing
        else None
    )
    collection_control = _collection_control(config, comparison)
    body = {
        "contract": SCORECARD_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": generated_at_utc.astimezone(timezone.utc).isoformat(),
        "config_sha256": config_sha256,
        "comparison": comparison,
        "loss_attribution": {
            "combined": _group(baseline, ["pair", "strategy", "atr_bucket", "spread_bucket", "regime_bucket", "side", "rapid_time_bucket_utc"]),
            "marginal": {
                key: _group(baseline, [key])
                for key in ("pair", "strategy", "atr_bucket", "spread_bucket", "regime_bucket", "side", "vol_shock", "rapid_time_bucket_utc")
            },
        },
        "best_so_far": best_by_profit_factor,
        "best_so_far_selection_metric": "PROFIT_FACTOR_THEN_AFTER_COST_NET_PIPS",
        "best_after_cost_net_pips_arm": best_by_net,
        "collection_control": collection_control,
        "positive_claim_allowed": False,
        "adoption_allowed": False,
        "automatic_parameter_change_allowed": False,
        "inventory_controller_evaluated": True,
        "external_order_attempts": 0,
        "external_orders": 0,
        "broker_mutation": False,
        "execution_authority": "NONE",
        "manual_tagless_positions_policy": "NO_TOUCH",
        "existing_tp_sl_policy": "NO_TOUCH",
        "limitations": [
            "SMALL_SINGLE_MARKET_WINDOW_BEST_SO_FAR_IS_NOT_PROFITABILITY_PROOF",
            "SHOCK_THRESHOLDS_ARE_FROZEN_POST_ATTRIBUTION_CHALLENGERS_NOT_LIVE_PARAMETERS",
            "S5_EXTREMA_CANNOT_ORDER_INTRABAR_MFE_MAE",
            "LANE_COOLDOWN_IS_CAUSAL_PROSPECTIVE_SHADOW_ONLY",
        ],
    }
    return seal(body)


def _profit_factor_value(value: Any) -> float:
    if value == "INF":
        return math.inf
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return -math.inf


def _collection_control(
    config: Mapping[str, Any], comparison: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    preregistration = config.get("preregistration") or {}
    early_stop = preregistration.get("early_futility_stop")
    if not isinstance(early_stop, Mapping):
        return {
            "status": "ACTIVE_NO_EARLY_FUTILITY_STOP",
            "halted": False,
            "new_signal_collection_allowed": True,
        }
    by_arm = {str(row.get("arm_id")): row for row in comparison}
    baseline = by_arm.get("BASELINE") or {}
    target = by_arm.get(str(preregistration.get("target_arm_id") or "")) or {}
    minimum = int(early_stop["minimum_target_fills"])
    floor_met = int(target.get("filled_count") or 0) >= minimum
    net_not_above = float(target.get("after_cost_net_pips") or 0.0) <= float(
        baseline.get("after_cost_net_pips") or 0.0
    )
    pf_not_above = _profit_factor_value(target.get("profit_factor")) <= _profit_factor_value(
        baseline.get("profit_factor")
    )
    halted = floor_met and net_not_above and pf_not_above
    return {
        "status": "HALTED_DUAL_METRIC_FUTILITY" if halted else "COLLECTING_FORWARD_EVIDENCE",
        "halted": halted,
        "new_signal_collection_allowed": not halted,
        "minimum_target_fills": minimum,
        "target_filled_count": int(target.get("filled_count") or 0),
        "target_net_pips_not_above_baseline": net_not_above,
        "target_profit_factor_not_above_baseline": pf_not_above,
        "requires_all": list(early_stop["requires_all"]),
    }


def append_rows_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
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
            if not isinstance(value, dict) or not sealed_valid(value, ROW_CONTRACT):
                raise ValueError(f"invalid challenger ledger row at line {number}")
            identity = str(value.get("row_identity") or "")
            if not identity or identity in seen:
                raise ValueError(f"duplicate challenger ledger identity at line {number}")
            seen.add(identity)
        handle.seek(0, os.SEEK_END)
        appended = 0
        for row in rows:
            identity = str(row.get("row_identity") or "")
            if not identity or identity in seen or not sealed_valid(row, ROW_CONTRACT):
                continue
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
    shadow_ledger_path: Path,
    outcome_ledger_path: Path,
    challenger_ledger_path: Path,
    scorecard_path: Path,
    config_path: Path,
    max_due: int = 12,
    client: OandaReadOnlyClient | None = None,
    truth_fetcher: Callable[[OandaReadOnlyClient, Mapping[str, Any], Mapping[str, Any]], tuple[Sequence[S5BidAskCandle], Sequence[str]]] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not 1 <= max_due <= 1000:
        raise ValueError("max_due must be inside 1..1000")
    config, config_sha = load_config(config_path)
    run_contract = {
        CONFIG_CONTRACT: "QR_FAST_BOT_CORRECTIVE_CHALLENGER_RUN_V1",
        CONFIG_CONTRACT_V2: "QR_FAST_BOT_CORRECTIVE_CHALLENGER_RUN_V2",
        CONFIG_CONTRACT_V3: "QR_FAST_BOT_CORRECTIVE_CHALLENGER_RUN_V3",
        CONFIG_CONTRACT_V4: "QR_FAST_BOT_CORRECTIVE_CHALLENGER_RUN_V4",
    }[config["contract"]]
    all_signals = load_jsonl(shadow_ledger_path)
    cutoff_raw = (config.get("preregistration") or {}).get("eligibility_cutoff_utc")
    cutoff = parse_utc(cutoff_raw) if cutoff_raw else None
    signals = [
        signal
        for signal in all_signals
        if cutoff is None or parse_utc(signal.get("generated_at_utc")) > cutoff
    ]
    pre_cutoff_signal_count = len(all_signals) - len(signals)
    outcomes = load_jsonl(outcome_ledger_path)
    existing = load_jsonl(challenger_ledger_path)
    evaluated = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    current_scorecard = build_scorecard(
        existing,
        config=config,
        config_sha256=config_sha,
        generated_at_utc=evaluated,
    )
    if current_scorecard["collection_control"]["halted"]:
        write_json_atomic(scorecard_path, current_scorecard)
        return {
            "contract": run_contract,
            "status": "HALTED_DUAL_METRIC_FUTILITY",
            "config_contract": config["contract"],
            "config_sha256": config_sha,
            "target_arm_id": config["preregistration"]["target_arm_id"],
            "eligibility_cutoff_utc": cutoff.isoformat() if cutoff else None,
            "pre_cutoff_signal_count": pre_cutoff_signal_count,
            "due_signal_count": 0,
            "processed_signal_count": 0,
            "appended_row_count": 0,
            "remaining_in_selected_batch": 0,
            "challenger_ledger_path": str(challenger_ledger_path),
            "scorecard_path": str(scorecard_path),
            "best_so_far": current_scorecard.get("best_so_far"),
            "collection_control": current_scorecard["collection_control"],
            "execution_authority": "NONE",
            "broker_http_methods_used": [],
            "broker_mutation": False,
            "external_order_attempts": 0,
            "external_orders": 0,
        }
    existing_identity = {str(row.get("row_identity")) for row in existing}
    by_signal = {str(row.get("signal_id")): row for row in signals if row.get("signal_id")}
    features = causal_features(signals, config)
    due: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for outcome in outcomes:
        signal = by_signal.get(str(outcome.get("signal_id") or ""))
        if signal is None or signal.get("signal_sha256") != outcome.get("signal_sha256"):
            continue
        expected = canonical_sha([signal["signal_sha256"], config_sha, "BASELINE"])
        if expected not in existing_identity:
            due.append((signal, outcome))
    due.sort(key=lambda row: (str(row[0].get("generated_at_utc")), str(row[0].get("signal_id"))))
    due = due[:max_due]
    broker = client or OandaReadOnlyClient()

    def default_fetch(
        value: OandaReadOnlyClient,
        signal: Mapping[str, Any],
        outcome: Mapping[str, Any],
    ) -> tuple[Sequence[S5BidAskCandle], Sequence[str]]:
        return fetch_frozen_s5_truth(
            value,
            pair=str(signal["pair"]),
            time_from=parse_utc(outcome["truth_request_from_utc"]),
            time_to=parse_utc(outcome["truth_request_to_utc"]),
            chunk_candle_limit=4500,
        )

    fetch = truth_fetcher or default_fetch
    new_rows: list[dict[str, Any]] = []
    for signal, outcome in due:
        candles, hashes = fetch(broker, signal, outcome)
        new_rows.extend(
            build_rows(
                signal,
                outcome,
                candles,
                hashes,
                features.get(str(signal["signal_id"]), {}),
                config,
                config_sha,
                evaluated_at_utc=evaluated,
            )
        )
    appended = append_rows_once(challenger_ledger_path, new_rows)
    all_rows = load_jsonl(challenger_ledger_path)
    scorecard = build_scorecard(
        all_rows,
        config=config,
        config_sha256=config_sha,
        generated_at_utc=evaluated,
    )
    write_json_atomic(scorecard_path, scorecard)
    remaining = max(0, len(due) - len({row["signal_id"] for row in new_rows if row["arm_id"] == "BASELINE"}))
    return {
        "contract": run_contract,
        "status": scorecard["collection_control"]["status"],
        "config_contract": config["contract"],
        "config_sha256": config_sha,
        "target_arm_id": config["preregistration"]["target_arm_id"],
        "eligibility_cutoff_utc": cutoff.isoformat() if cutoff else None,
        "pre_cutoff_signal_count": pre_cutoff_signal_count,
        "due_signal_count": len(due),
        "processed_signal_count": len({row["signal_id"] for row in new_rows if row["arm_id"] == "BASELINE"}),
        "appended_row_count": appended,
        "remaining_in_selected_batch": remaining,
        "challenger_ledger_path": str(challenger_ledger_path),
        "scorecard_path": str(scorecard_path),
        "best_so_far": scorecard.get("best_so_far"),
        "collection_control": scorecard["collection_control"],
        "execution_authority": "NONE",
        "broker_http_methods_used": ["GET"] if due else [],
        "broker_mutation": False,
        "external_order_attempts": 0,
        "external_orders": 0,
    }
