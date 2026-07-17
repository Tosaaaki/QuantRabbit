"""Exact, read-only truth for sealed fast-bot causal episodes.

The V1 episode ledger remains immutable.  A V2 episode handoff freezes the
complete broker snapshot and the complete prospective learning shadow before
future S5 truth exists.  This module projects a separate vehicle ledger from a
newly CONFIRMED episode, resolves every frozen candidate/arm on one exact
bid/ask path, and publishes a diagnostic episode-cluster scorecard.

Nothing in this module can create an order, mutate broker state, promote a
rule, or grant live permission.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import stat
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from quant_rabbit.analysis.market_state import TAXONOMY_SECTIONS
from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.fast_bot_episode import (
    MAX_LEDGER_BYTES as MAX_EPISODE_LEDGER_BYTES,
    MAX_LEDGER_EVENTS as MAX_EPISODE_LEDGER_EVENTS,
    verify_episode_ledger,
)
from quant_rabbit.fast_bot_learning import (
    CELL_ORDER,
    LEARNING_SELECTION_POLICY,
    LEARNING_SHADOW_CONTRACT,
)
from quant_rabbit.fast_bot_learning_truth import (
    SCORING_POLICY as LEARNING_SCORING_POLICY,
    TRUTH_CHUNK_CANDLE_LIMIT,
    _learning_seat_deep_valid,
    _outcome_valid_for_seat,
    _seat_maturity,
    _truth_path_sha,
    _validate_truth_path,
    resolve_fast_bot_learning_seat,
)
from quant_rabbit.fast_bot_technical_hypotheses import (
    build_fast_bot_technical_hypotheses,
    technical_hypothesis_shadow_valid,
)
from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle
from quant_rabbit.technical_forecast_forward_truth import fetch_frozen_s5_truth


EPISODE_HANDOFF_V2_CONTRACT = "QR_FAST_BOT_EPISODE_HANDOFF_V2"
VEHICLE_CONTRACT = "QR_FAST_BOT_EPISODE_VEHICLE_V1"
OUTCOME_CONTRACT = "QR_FAST_BOT_EPISODE_S5_OUTCOME_V1"
SCORECARD_CONTRACT = "QR_FAST_BOT_EPISODE_CLUSTER_SCORECARD_V1"
TRUTH_ADAPTER_CONTRACT = "QR_FAST_BOT_EPISODE_TRUTH_ADAPTER_V1"
VEHICLE_POLICY = "SEALED_PROSPECTIVE_LEARNING_SEAT_V1"
SCORING_POLICY = "QR_FAST_BOT_EPISODE_LEARNING_S5_CONSERVATIVE_AND_MARK_V1"

ROUTE_ALIGNED = "ROUTE_ALIGNED"
INVERSE_DIRECTION = "INVERSE_DIRECTION"
SAME_SIDE_OTHER_METHOD = "SAME_SIDE_OTHER_METHOD"
INVERSE_SIDE_OTHER_METHOD = "INVERSE_SIDE_OTHER_METHOD"
EPISODE_ROLES = {
    ROUTE_ALIGNED,
    INVERSE_DIRECTION,
    SAME_SIDE_OTHER_METHOD,
    INVERSE_SIDE_OTHER_METHOD,
}

MAX_VEHICLE_LEDGER_BYTES = 64 * 1024 * 1024
MAX_OUTCOME_LEDGER_BYTES = 128 * 1024 * 1024
MAX_LEDGER_ROWS = 16_384
MAX_DUE_PER_RUN = 12
MAX_ERROR_COUNT = 20
MAX_UNSCORED_EXAMPLES = 32
ROUND_DIGITS = 6
TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D")
TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14_400,
    "D": 86_400,
}
MARKET_STATE_ENUM_FEATURES = {
    key: frozenset(values) for key, values in TAXONOMY_SECTIONS.items()
}
MARKET_STATE_FEATURES = tuple(MARKET_STATE_ENUM_FEATURES) + (
    "confidence",
    "evidence_complete",
)
NUMERIC_INDICATOR_FEATURES = frozenset(
    {
        "close",
        "sma_20",
        "adx",
        "adx_14",
        "adx_percentile_100",
        "plus_di_14",
        "minus_di_14",
        "ema_12",
        "ema_20",
        "ema_50",
        "ema_slope_5",
        "ema_slope_20",
        "ema_gap_5_24",
        "ema_gap_12_48",
        "macd",
        "macd_signal",
        "macd_hist",
        "macd_hist_scaled",
        "rsi_14",
        "stoch_rsi",
        "stoch_rsi_k",
        "stoch_rsi_d",
        "williams_r",
        "williams_r_14",
        "cci_14",
        "mfi_14",
        "bb_position",
        "bb_width",
        "bollinger_position",
        "bollinger_width",
        "donchian_position",
        "donchian_high",
        "donchian_low",
        "donchian_width_pips",
        "atr",
        "atr_14",
        "atr_pips",
        "atr_percentile",
        "atr_percentile_100",
        "atr_percentile_24h",
        "choppiness",
        "choppiness_14",
        "hurst",
        "hurst_100",
        "hurst_returns",
        "half_life",
        "half_life_60",
        "roc",
        "roc_5",
        "roc_10",
        "roc_14",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_span_pips",
        "keltner_width",
        "bb_squeeze",
        "bb_width_percentile_100",
        "aroon_up_14",
        "aroon_down_14",
        "aroon_osc",
        "aroon_osc_14",
        "vortex_plus_14",
        "vortex_minus_14",
        "supertrend",
        "supertrend_value",
        "supertrend_dir",
        "psar_value",
        "psar_dir",
        "hull_ma_20",
        "kama_10",
        "alma_20",
        "linreg_slope_20",
        "linreg_r2_20",
        "linreg_channel_upper",
        "linreg_channel_lower",
        "ichimoku_tenkan",
        "ichimoku_kijun",
        "ichimoku_span_a",
        "ichimoku_span_b",
        "ichimoku_cloud_pos",
        "zscore",
        "zscore_20",
        "z_score_20",
        "realized_vol_20",
        "vwap",
        "vwap_gap",
        "vwap_gap_pips",
        "avwap_anchor",
        "avwap_upper_1sd",
        "avwap_lower_1sd",
        "avwap_upper_2sd",
        "avwap_lower_2sd",
        "avwap_swing_high",
        "avwap_swing_low",
        "price_vs_ema50_per_atr",
    }
)
ENUM_INDICATOR_FEATURES = frozenset(
    {
        "ema_order",
        "atr_regime",
        "regime_quantile",
    }
)
INDICATOR_SERIES_FEATURES = frozenset(
    {
        "rsi_14",
        "macd_hist",
        "adx_14",
        "atr_pips",
        "ema_12_minus_50_pips",
    }
)
MAX_INDICATOR_SERIES_VALUES = 30
HYPOTHESIS_FAMILIES = (
    "TREND",
    "PULLBACK",
    "BREAKOUT",
    "BREAKOUT_FAILURE",
    "RANGE",
    "EXHAUSTION",
)
MAX_TECHNICAL_FEATURE_SNAPSHOT_BYTES = 64 * 1024


class TruthLockBusyError(RuntimeError):
    """Raised when another episode truth cycle already owns the run lock."""


def run_fast_bot_episode_truth_cycle(
    *,
    handoffs: Sequence[Mapping[str, Any]] = (),
    episode_ledger_path: Path,
    source_archive_dir: Path,
    vehicle_ledger_path: Path,
    outcome_ledger_path: Path,
    scorecard_path: Path,
    lock_path: Path | None = None,
    client_factory: Callable[[], Any] = OandaReadOnlyClient,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Project confirmed vehicles and resolve mature ones under one lock.

    ``handoffs`` must be the exact V2 handoffs consumed by the episode worker.
    Passing no handoffs is valid and lets later Guardian cycles resolve vehicles
    whose longest frozen arm has just matured.
    """

    now = _aware_utc((clock or (lambda: datetime.now(timezone.utc)))())
    base = {
        "contract": TRUTH_ADAPTER_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": now.isoformat(),
        "diagnostic_only": True,
        "order_authority": "NONE",
        "automatic_promotion_allowed": False,
        "promotion_allowed": False,
        "primary_effect": False,
        "risk_effect": False,
        "handoff_confirmed_event_count": 0,
        "handoff_confirmed_vehicle_count": 0,
        "handoff_confirmed_unscored_count": 0,
        "handoff_confirmed_unscored_reason_counts": {},
        "handoff_confirmed_unscored_examples": [],
        "vehicle_ledger_idempotent": 0,
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_intents": [],
    }
    resolved_lock = lock_path or vehicle_ledger_path.with_name(
        ".fast_bot_episode_truth.lock"
    )
    try:
        with _whole_run_lock(resolved_lock):
            return _run_locked(
                base=base,
                now=now,
                handoffs=handoffs,
                episode_ledger_path=episode_ledger_path,
                source_archive_dir=source_archive_dir,
                vehicle_ledger_path=vehicle_ledger_path,
                outcome_ledger_path=outcome_ledger_path,
                scorecard_path=scorecard_path,
                client_factory=client_factory,
            )
    except TruthLockBusyError:
        return {
            **base,
            "status": "LOCK_BUSY",
            "vehicle_projection_status": "FAILED",
            "handoff_confirmed_vehicle_count": 0,
            "vehicle_ledger_idempotent": 0,
            "broker_read": False,
            "vehicle_ledger_appended": 0,
            "outcome_ledger_appended": 0,
        }
    except (OSError, TypeError, ValueError) as error:
        return {
            **base,
            "status": "PRECHECK_OR_PERSISTENCE_FAILED_CLOSED",
            "vehicle_projection_status": "FAILED",
            "handoff_confirmed_vehicle_count": 0,
            "vehicle_ledger_idempotent": 0,
            "broker_read": False,
            "vehicle_ledger_appended": 0,
            "outcome_ledger_appended": 0,
            "error": f"{type(error).__name__}: {error}"[:320],
        }


def _run_locked(
    *,
    base: Mapping[str, Any],
    now: datetime,
    handoffs: Sequence[Mapping[str, Any]],
    episode_ledger_path: Path,
    source_archive_dir: Path,
    vehicle_ledger_path: Path,
    outcome_ledger_path: Path,
    scorecard_path: Path,
    client_factory: Callable[[], Any],
) -> dict[str, Any]:
    events = _load_jsonl(
        episode_ledger_path,
        max_bytes=MAX_EPISODE_LEDGER_BYTES,
        max_rows=MAX_EPISODE_LEDGER_EVENTS,
    )
    verified, verify_error = verify_episode_ledger(
        events,
        as_of_utc=now,
        source_archive_dir=source_archive_dir,
    )
    if not verified:
        return {
            **base,
            "status": "EPISODE_LEDGER_INVALID_FAIL_CLOSED",
            "vehicle_projection_status": "FAILED",
            "handoff_confirmed_vehicle_count": 0,
            "vehicle_ledger_idempotent": 0,
            "broker_read": False,
            "vehicle_ledger_appended": 0,
            "outcome_ledger_appended": 0,
            "error": str(verify_error or "EPISODE_LEDGER_INVALID"),
        }
    events_by_sha = {
        str(event.get("event_sha256") or ""): event for event in events
    }
    vehicles = _load_vehicles(vehicle_ledger_path, events_by_sha=events_by_sha)
    projected = _project_handoffs(
        events=events,
        handoffs=handoffs,
        existing_vehicles=vehicles,
    )
    projection_summary = _projection_summary(projected)
    if projected["conflicts"]:
        return {
            **base,
            "status": "VEHICLE_IDENTITY_CONFLICT",
            "vehicle_projection_status": "FAILED",
            **projection_summary,
            "broker_read": False,
            "vehicle_ledger_appended": 0,
            "outcome_ledger_appended": 0,
            "vehicle_identity_conflicts": projected["conflicts"],
        }
    new_vehicles = projected["new"]
    if new_vehicles:
        vehicles = [*vehicles, *new_vehicles]
        _write_jsonl_atomic(
            vehicle_ledger_path,
            vehicles,
            max_bytes=MAX_VEHICLE_LEDGER_BYTES,
            max_rows=MAX_LEDGER_ROWS,
        )

    try:
        outcomes = _load_outcomes(outcome_ledger_path, vehicles=vehicles)
    except (OSError, TypeError, ValueError) as error:
        return {
            **base,
            "status": "OUTCOME_LEDGER_INVALID_FAIL_CLOSED",
            "vehicle_projection_status": "VERIFIED",
            **projection_summary,
            "broker_read": False,
            "vehicle_ledger_appended": len(new_vehicles),
            "outcome_ledger_appended": 0,
            "error": f"{type(error).__name__}: {error}"[:320],
        }
    vehicle_by_sha = {
        str(vehicle["contract_sha256"]): vehicle for vehicle in vehicles
    }
    current_rows: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for outcome in outcomes:
        identity = _outcome_identity(outcome)
        current_rows.setdefault(identity, []).append(outcome)
    conflicts: list[dict[str, Any]] = []
    resolved_vehicle_sha: set[str] = set()
    for identity, rows in current_rows.items():
        vehicle = vehicle_by_sha.get(identity[0])
        valid = [
            row
            for row in rows
            if vehicle is not None and _episode_outcome_valid(row, vehicle)
        ]
        if len(rows) == 1 and len(valid) == 1:
            resolved_vehicle_sha.add(identity[0])
            continue
        conflicts.append(
            {
                "vehicle_contract_sha256": identity[0],
                "scoring_policy": identity[1],
                "row_count": len(rows),
                "valid_row_count": len(valid),
                "reason": "CURRENT_POLICY_OUTCOME_IDENTITY_CONFLICT",
            }
        )
    if conflicts:
        return {
            **base,
            "status": "OUTCOME_IDENTITY_CONFLICT",
            "vehicle_projection_status": "VERIFIED",
            **projection_summary,
            "broker_read": False,
            "vehicle_ledger_appended": len(new_vehicles),
            "outcome_ledger_appended": 0,
            "outcome_identity_conflicts": conflicts,
        }

    due = sorted(
        (
            (_parse_utc(vehicle["maturity_at_utc"]), vehicle)
            for vehicle in vehicles
            if str(vehicle["contract_sha256"]) not in resolved_vehicle_sha
            and _parse_utc(vehicle["maturity_at_utc"]) <= now
        ),
        key=lambda item: (item[0], str(item[1]["vehicle_id"])),
    )
    selected = [vehicle for _, vehicle in due[:MAX_DUE_PER_RUN]]
    if not selected:
        scorecard = build_fast_bot_episode_scorecard(
            vehicles,
            outcomes,
            as_of_utc=now,
        )
        _write_json_atomic(scorecard_path, scorecard)
        return {
            **base,
            "status": "PROJECTED_NO_DUE" if new_vehicles else "NO_DUE_VEHICLES",
            "vehicle_projection_status": "VERIFIED",
            **projection_summary,
            "broker_read": False,
            "vehicle_ledger_appended": len(new_vehicles),
            "outcome_ledger_appended": 0,
            "vehicle_count": len(vehicles),
            "due_count": len(due),
            "scorecard_status": scorecard["status"],
        }

    resolved: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    try:
        client = client_factory()
    except Exception as error:  # pragma: no cover - credential/client boundary
        scorecard = build_fast_bot_episode_scorecard(
            vehicles,
            outcomes,
            as_of_utc=now,
        )
        _write_json_atomic(scorecard_path, scorecard)
        return {
            **base,
            "status": "RESOLVED_WITH_ERRORS",
            "vehicle_projection_status": "VERIFIED",
            **projection_summary,
            "broker_read": False,
            "vehicle_ledger_appended": len(new_vehicles),
            "outcome_ledger_appended": 0,
            "vehicle_count": len(vehicles),
            "due_count": len(due),
            "selected_due_count": len(selected),
            "errors": [
                {
                    "vehicle_id": "",
                    "pair": "",
                    "error": f"{type(error).__name__}: {error}"[:320],
                }
            ],
            "scorecard_status": scorecard["status"],
        }
    for vehicle in selected:
        try:
            seat = vehicle["learning_seat"]
            generated = _parse_utc(seat["generated_at_utc"])
            maturity = _seat_maturity(seat)
            candles, hashes = fetch_frozen_s5_truth(
                client,
                pair=str(vehicle["pair"]),
                time_from=generated,
                time_to=maturity,
                chunk_candle_limit=TRUTH_CHUNK_CANDLE_LIMIT,
            )
            resolved.append(
                resolve_fast_bot_episode_vehicle(
                    vehicle,
                    candles,
                    resolved_at_utc=now,
                    truth_chunk_sha256=hashes,
                )
            )
        except Exception as error:  # pragma: no cover - broker boundary
            if len(errors) < MAX_ERROR_COUNT:
                errors.append(
                    {
                        "vehicle_id": str(vehicle.get("vehicle_id") or ""),
                        "pair": str(vehicle.get("pair") or ""),
                        "error": f"{type(error).__name__}: {error}"[:320],
                    }
                )
    if resolved:
        outcomes = [*outcomes, *resolved]
        _write_jsonl_atomic(
            outcome_ledger_path,
            outcomes,
            max_bytes=MAX_OUTCOME_LEDGER_BYTES,
            max_rows=MAX_LEDGER_ROWS,
        )
    scorecard = build_fast_bot_episode_scorecard(
        vehicles,
        outcomes,
        as_of_utc=now,
    )
    _write_json_atomic(scorecard_path, scorecard)
    return {
        **base,
        "status": "RESOLVED_WITH_ERRORS" if errors else "RESOLVED",
        "vehicle_projection_status": "VERIFIED",
        **projection_summary,
        "broker_read": True,
        "vehicle_ledger_appended": len(new_vehicles),
        "outcome_ledger_appended": len(resolved),
        "vehicle_count": len(vehicles),
        "due_count": len(due),
        "selected_due_count": len(selected),
        "errors": errors,
        "scorecard_status": scorecard["status"],
    }


def _project_handoffs(
    *,
    events: Sequence[Mapping[str, Any]],
    handoffs: Sequence[Mapping[str, Any]],
    existing_vehicles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    existing_by_id = {
        str(vehicle["vehicle_id"]): vehicle for vehicle in existing_vehicles
    }
    new: list[dict[str, Any]] = []
    conflicts: list[dict[str, str]] = []
    batch_by_id: dict[str, dict[str, Any]] = {}
    confirmed_event_count = 0
    matched_count = 0
    idempotent_count = 0
    unscored_no_frozen_seat_count = 0
    unscored_no_confirmation_candle_seat_count = 0
    unscored_examples: list[dict[str, str]] = []
    for handoff in handoffs:
        _validate_v2_handoff(handoff)
        cycle = _parse_utc(handoff["cycle_generated_at_utc"])
        regime_sha = str(handoff["regime_contract_sha256"])
        matched = [
            event
            for event in events
            if event.get("state") == "CONFIRMED"
            and _parse_utc(event.get("generated_at_utc")) == cycle
            and isinstance(event.get("observation"), Mapping)
            and event["observation"].get("regime_contract_sha256") == regime_sha
        ]
        shadow = handoff["prospective_vehicle_shadow"]
        seats_by_pair = {
            str(seat.get("pair") or ""): seat
            for seat in shadow.get("seats", [])
            if isinstance(seat, Mapping)
        }
        for event in matched:
            confirmed_event_count += 1
            pair = str(event["pair"])
            seat = seats_by_pair.get(pair)
            if seat is None:
                # The prospective learning cohort is intentionally narrower
                # than the episode detector whenever exact quote/ATR/input
                # requirements cannot freeze a six-cell seat.  A later
                # confirmed episode must not fabricate or backfill that
                # missing vehicle.  It is explicit unscored evidence, while
                # the remaining seat-backed events stay projectable.
                unscored_no_frozen_seat_count += 1
                if len(unscored_examples) < MAX_UNSCORED_EXAMPLES:
                    unscored_examples.append(
                        {
                            "pair": pair,
                            "confirmed_event_sha256": str(
                                event["event_sha256"]
                            ),
                            "handoff_contract_sha256": str(
                                handoff["contract_sha256"]
                            ),
                            "prospective_shadow_status": str(
                                shadow.get("status") or "UNKNOWN"
                            ),
                            "reason": (
                                "CONFIRMED_EVENT_NO_FROZEN_VEHICLE_SEAT"
                            ),
                        }
                    )
                continue
            if _late_confirmation_has_no_same_candle_seat(
                event=event,
                handoff=handoff,
                seat=seat,
            ):
                # A delayed detector can legitimately confirm the oldest
                # unseen M1 candle while the same-cycle learning shadow has
                # already frozen a newer M1 seat.  That newer seat is not the
                # vehicle that existed at confirmation time.  Retain the
                # episode as explicit unscored evidence; never backfill or
                # relabel the newer seat as the older confirmation candle.
                unscored_no_confirmation_candle_seat_count += 1
                if len(unscored_examples) < MAX_UNSCORED_EXAMPLES:
                    unscored_examples.append(
                        {
                            "pair": pair,
                            "confirmed_event_sha256": str(
                                event["event_sha256"]
                            ),
                            "handoff_contract_sha256": str(
                                handoff["contract_sha256"]
                            ),
                            "confirmation_candle_close_utc": str(
                                event["observation"]["candle_close_utc"]
                            ),
                            "frozen_seat_m1_closed_candle_utc": str(
                                seat["m1_closed_candle_utc"]
                            ),
                            "reason": (
                                "CONFIRMED_EVENT_NO_FROZEN_SEAT_AT_"
                                "CONFIRMATION_CANDLE"
                            ),
                        }
                    )
                continue
            matched_count += 1
            vehicle = _build_vehicle(event=event, handoff=handoff, seat=seat)
            vehicle_id = str(vehicle["vehicle_id"])
            prior = existing_by_id.get(vehicle_id) or batch_by_id.get(vehicle_id)
            if prior is not None:
                if dict(prior) != vehicle:
                    conflicts.append(
                        {
                            "vehicle_id": vehicle_id,
                            "confirmed_event_sha256": str(event["event_sha256"]),
                            "reason": "SAME_IDENTITY_DIFFERENT_FROZEN_VEHICLE",
                        }
                    )
                else:
                    idempotent_count += 1
                continue
            batch_by_id[vehicle_id] = vehicle
            new.append(vehicle)
    return {
        "new": new,
        "conflicts": conflicts,
        "confirmed_event_count": confirmed_event_count,
        "matched_count": matched_count,
        "idempotent_count": idempotent_count,
        "unscored_no_frozen_seat_count": unscored_no_frozen_seat_count,
        "unscored_no_confirmation_candle_seat_count": (
            unscored_no_confirmation_candle_seat_count
        ),
        "unscored_examples": unscored_examples,
    }


def _projection_summary(projected: Mapping[str, Any]) -> dict[str, Any]:
    no_frozen_seat = int(projected["unscored_no_frozen_seat_count"])
    no_confirmation_candle_seat = int(
        projected["unscored_no_confirmation_candle_seat_count"]
    )
    unscored = no_frozen_seat + no_confirmation_candle_seat
    reason_counts = {}
    if no_frozen_seat:
        reason_counts["CONFIRMED_EVENT_NO_FROZEN_VEHICLE_SEAT"] = (
            no_frozen_seat
        )
    if no_confirmation_candle_seat:
        reason_counts[
            "CONFIRMED_EVENT_NO_FROZEN_SEAT_AT_CONFIRMATION_CANDLE"
        ] = no_confirmation_candle_seat
    return {
        "handoff_confirmed_event_count": int(
            projected["confirmed_event_count"]
        ),
        "handoff_confirmed_vehicle_count": int(projected["matched_count"]),
        "handoff_confirmed_unscored_count": unscored,
        "handoff_confirmed_unscored_reason_counts": reason_counts,
        "handoff_confirmed_unscored_examples": list(
            projected["unscored_examples"]
        ),
        "vehicle_ledger_idempotent": int(projected["idempotent_count"]),
    }


def _late_confirmation_has_no_same_candle_seat(
    *,
    event: Mapping[str, Any],
    handoff: Mapping[str, Any],
    seat: Mapping[str, Any],
) -> bool:
    """Recognize only the valid delayed-confirmation clock gap.

    Every non-clock source binding is checked here.  A broker, regime, pair,
    generation, or event-source mismatch therefore still reaches
    ``_build_vehicle`` and fails closed instead of being mislabeled unscored.
    """

    observation = event.get("observation")
    if not isinstance(observation, Mapping):
        return False
    try:
        confirmation_close = _parse_utc(observation["candle_close_utc"])
        seat_m1_close = _parse_utc(seat["m1_closed_candle_utc"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        event.get("state") == "CONFIRMED"
        and event.get("late_detected") is True
        and seat.get("pair") == event.get("pair")
        and seat.get("generated_at_utc")
        == handoff.get("cycle_generated_at_utc")
        and seat.get("regime_contract_sha256")
        == handoff.get("regime_contract_sha256")
        and seat.get("broker_snapshot_sha256")
        == handoff.get("broker_snapshot_sha256")
        and observation.get("regime_contract_sha256")
        == handoff.get("regime_contract_sha256")
        and seat_m1_close > confirmation_close
    )


def _build_vehicle(
    *,
    event: Mapping[str, Any],
    handoff: Mapping[str, Any],
    seat: Mapping[str, Any],
) -> dict[str, Any]:
    if not _learning_seat_deep_valid(seat):
        raise ValueError("prospective episode seat is invalid")
    cells = {
        (str(candidate.get("side") or ""), str(candidate.get("method") or ""))
        for candidate in seat.get("candidates", [])
        if isinstance(candidate, Mapping)
    }
    if cells != set(CELL_ORDER):
        raise ValueError("prospective episode seat must freeze all six side-method cells")
    if any(not candidate.get("arms") for candidate in seat["candidates"]):
        raise ValueError("prospective episode candidate has no frozen arms")
    route = event.get("route")
    observation = event.get("observation")
    anchor = event.get("anchor")
    if not all(isinstance(item, Mapping) for item in (route, observation, anchor)):
        raise ValueError("confirmed episode route binding is invalid")
    pair = str(event["pair"])
    cycle = str(handoff["cycle_generated_at_utc"])
    if (
        event.get("state") != "CONFIRMED"
        or seat.get("pair") != pair
        or seat.get("generated_at_utc") != cycle
        or seat.get("m1_closed_candle_utc") != observation.get("candle_close_utc")
        or seat.get("regime_contract_sha256")
        != handoff.get("regime_contract_sha256")
        or seat.get("broker_snapshot_sha256")
        != handoff.get("broker_snapshot_sha256")
        or observation.get("regime_contract_sha256")
        != handoff.get("regime_contract_sha256")
    ):
        raise ValueError("confirmed episode vehicle source binding is invalid")
    trade_side = str(route.get("trade_side") or "")
    route_methods = [str(item) for item in route.get("candidate_methods", [])]
    if trade_side not in {"LONG", "SHORT"} or not route_methods:
        raise ValueError("confirmed episode has no routable side/method")
    candidate_roles = [
        {
            "candidate_id": str(candidate["candidate_id"]),
            "candidate_sha256": str(candidate["candidate_sha256"]),
            "side": str(candidate["side"]),
            "method": str(candidate["method"]),
            "episode_role": _episode_role(
                side=str(candidate["side"]),
                method=str(candidate["method"]),
                trade_side=trade_side,
                route_methods=route_methods,
            ),
        }
        for candidate in seat["candidates"]
    ]
    technical_feature_snapshot = _build_technical_feature_snapshot(
        handoff,
        pair=pair,
    )
    technical_feature_snapshot_sha = _canonical_sha(technical_feature_snapshot)
    technical_hypothesis_shadow = build_fast_bot_technical_hypotheses(
        technical_feature_snapshot,
        attempt_direction=str(anchor["attempt_direction"]),
        branch_outcome=str(route["branch_outcome"]),
        route_family=str(route["route_family"]),
        spread_pips=float(seat["executable_spread_pips"]),
        m5_atr_pips=float(seat["m5_atr_pips"]),
        spread_to_m5_atr=float(seat["spread_to_m5_atr"]),
    )
    if technical_hypothesis_shadow.get("status") != "EMITTED":
        raise ValueError("confirmed episode technical hypothesis shadow is invalid")
    technical_hypothesis_shadow_sha = str(
        technical_hypothesis_shadow["contract_sha256"]
    )
    technical_hypothesis_catalog_sha = str(
        technical_hypothesis_shadow["catalog_contract_sha256"]
    )
    technical_cost_state_sha = str(
        technical_hypothesis_shadow["cost_state_sha256"]
    )
    technical_hypothesis_evaluator_policy = str(
        technical_hypothesis_shadow["evaluator_policy"]
    )
    technical_hypothesis_evaluator_policy_sha = _canonical_sha(
        {"evaluator_policy": technical_hypothesis_evaluator_policy}
    )
    event_sha = str(event["event_sha256"])
    identity = {
        "vehicle_policy": VEHICLE_POLICY,
        "confirmed_event_sha256": event_sha,
    }
    source_binding = {
        "confirmed_event_sha256": event_sha,
        "handoff_contract_sha256": str(handoff["contract_sha256"]),
        "broker_snapshot_sha256": str(handoff["broker_snapshot_sha256"]),
        "regime_contract_sha256": str(handoff["regime_contract_sha256"]),
        "prospective_vehicle_shadow_sha256": str(
            handoff["prospective_vehicle_shadow_sha256"]
        ),
        "learning_seat_contract_sha256": str(seat["contract_sha256"]),
        "technical_feature_snapshot_sha256": technical_feature_snapshot_sha,
        "technical_hypothesis_catalog_sha256": technical_hypothesis_catalog_sha,
        "technical_hypothesis_shadow_sha256": technical_hypothesis_shadow_sha,
        "technical_cost_state_sha256": technical_cost_state_sha,
        "technical_hypothesis_evaluator_policy_sha256": (
            technical_hypothesis_evaluator_policy_sha
        ),
    }
    input_proof_eligible = (
        seat.get("causal_input_proof_eligible") is True
        if seat.get("selection_policy") == LEARNING_SELECTION_POLICY
        else True
    )
    scorecard_ineligibility_reasons = []
    if event.get("late_detected") is True:
        scorecard_ineligibility_reasons.append("LATE_DETECTED_EPISODE")
    if not input_proof_eligible:
        scorecard_ineligibility_reasons.append(
            "INPUT_BLOCKED_SHADOW_DIAGNOSTIC_ONLY"
        )
    body = {
        "contract": VEHICLE_CONTRACT,
        "schema_version": 1,
        "vehicle_policy": VEHICLE_POLICY,
        "vehicle_id": _canonical_sha(identity)[:24],
        "episode_id": str(event["episode_id"]),
        "confirmed_event_id": str(event["event_id"]),
        "confirmed_event_sha256": event_sha,
        "confirmed_at_utc": str(observation["candle_close_utc"]),
        "handoff_cycle_generated_at_utc": cycle,
        "pair": pair,
        "attempt_direction": str(anchor["attempt_direction"]),
        "branch_outcome": str(route["branch_outcome"]),
        "trade_side": trade_side,
        "route_family": str(route["route_family"]),
        "route_candidate_methods": route_methods,
        "late_detected": event.get("late_detected") is True,
        "causal_input_proof_eligible": input_proof_eligible,
        "scorecard_eligible": not scorecard_ineligibility_reasons,
        "scorecard_ineligibility_reasons": scorecard_ineligibility_reasons,
        "source_binding": source_binding,
        "source_binding_sha256": _canonical_sha(source_binding),
        "learning_seat_id": str(seat["seat_id"]),
        "learning_seat_contract_sha256": str(seat["contract_sha256"]),
        "learning_arm_policy": str(seat.get("arm_policy") or ""),
        "candidate_count": len(seat["candidates"]),
        "arm_count": sum(len(candidate["arms"]) for candidate in seat["candidates"]),
        "candidate_roles": candidate_roles,
        "learning_seat": dict(seat),
        "technical_feature_snapshot": technical_feature_snapshot,
        "technical_feature_snapshot_sha256": technical_feature_snapshot_sha,
        "technical_feature_hypothesis_families": list(HYPOTHESIS_FAMILIES),
        "technical_hypothesis_shadow": technical_hypothesis_shadow,
        "technical_hypothesis_shadow_sha256": technical_hypothesis_shadow_sha,
        "technical_hypothesis_catalog_sha256": technical_hypothesis_catalog_sha,
        "technical_cost_state_sha256": technical_cost_state_sha,
        "technical_hypothesis_evaluator_policy": (
            technical_hypothesis_evaluator_policy
        ),
        "technical_hypothesis_evaluator_policy_sha256": (
            technical_hypothesis_evaluator_policy_sha
        ),
        "maturity_at_utc": _seat_maturity(seat).isoformat(),
        "learning_scoring_policy": LEARNING_SCORING_POLICY,
        "diagnostic_only": True,
        "order_authority": "NONE",
        "automatic_promotion_allowed": False,
        "promotion_allowed": False,
        "primary_effect": False,
        "risk_effect": False,
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    return _seal(body)


def resolve_fast_bot_episode_vehicle(
    vehicle: Mapping[str, Any],
    candles: Sequence[S5BidAskCandle],
    *,
    resolved_at_utc: datetime,
    truth_chunk_sha256: Sequence[str],
) -> dict[str, Any]:
    """Resolve all frozen arms and add a separate executable horizon mark."""

    if not _vehicle_valid(vehicle):
        raise ValueError("invalid episode vehicle")
    resolved = _aware_utc(resolved_at_utc)
    seat = vehicle["learning_seat"]
    learning_outcome = resolve_fast_bot_learning_seat(
        seat,
        candles,
        resolved_at_utc=resolved,
        truth_chunk_sha256=truth_chunk_sha256,
    )
    roles = {
        str(row["candidate_sha256"]): row
        for row in vehicle["candidate_roles"]
    }
    ordered = sorted(candles, key=lambda item: item.timestamp_utc)
    truth_path_candles = [_truth_candle_receipt(item) for item in ordered]
    truth_path_candles_sha = _canonical_sha(truth_path_candles)
    observations: list[dict[str, Any]] = []
    for candidate in learning_outcome["candidates"]:
        role = roles[str(candidate["candidate_sha256"])]
        side = str(candidate["side"])
        method = str(candidate["method"])
        for arm in candidate["arms"]:
            observations.append(
                _arm_observation(
                    arm,
                    candidate=candidate,
                    side=side,
                    method=method,
                    episode_role=str(role["episode_role"]),
                    pair=str(vehicle["pair"]),
                    candles=ordered,
                )
            )
    body = {
        "contract": OUTCOME_CONTRACT,
        "schema_version": 1,
        "scoring_policy": SCORING_POLICY,
        "vehicle_id": str(vehicle["vehicle_id"]),
        "vehicle_contract_sha256": str(vehicle["contract_sha256"]),
        "episode_id": str(vehicle["episode_id"]),
        "confirmed_event_sha256": str(vehicle["confirmed_event_sha256"]),
        "pair": str(vehicle["pair"]),
        "branch_outcome": str(vehicle["branch_outcome"]),
        "attempt_direction": str(vehicle["attempt_direction"]),
        "route_family": str(vehicle["route_family"]),
        "late_detected": vehicle.get("late_detected") is True,
        "scorecard_eligible": vehicle.get("scorecard_eligible") is True,
        "resolved_at_utc": resolved.isoformat(),
        "maturity_at_utc": str(vehicle["maturity_at_utc"]),
        "learning_outcome_contract_sha256": str(
            learning_outcome["contract_sha256"]
        ),
        "learning_outcome": learning_outcome,
        "truth_source": "OANDA_S5_BID_ASK",
        "truth_request_from_utc": str(learning_outcome["truth_request_from_utc"]),
        "truth_request_to_utc": str(learning_outcome["truth_request_to_utc"]),
        "truth_path_sha256": str(learning_outcome["truth_path_sha256"]),
        "truth_chunk_sha256": list(learning_outcome["truth_chunk_sha256"]),
        "truth_path_candles": truth_path_candles,
        "truth_path_candles_sha256": truth_path_candles_sha,
        "truth_path_candles_persisted_for_membership_validation": True,
        "candidate_count": len(learning_outcome["candidates"]),
        "arm_observation_count": len(observations),
        "arm_observations": observations,
        "proof_score_semantics": (
            "LEARNING_CONSERVATIVE_FULL_SL_IF_OPEN_AT_HORIZON"
        ),
        "observed_mark_semantics": (
            "SEPARATE_LAST_EXECUTABLE_BID_ASK_CLOSE_NOT_PROOF_SCORE"
        ),
        "proof_and_observed_mark_must_not_be_mixed": True,
        "diagnostic_only": True,
        "order_authority": "NONE",
        "automatic_promotion_allowed": False,
        "promotion_allowed": False,
        "primary_effect": False,
        "risk_effect": False,
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    outcome = _seal(body)
    if not _episode_outcome_valid(outcome, vehicle):
        raise ValueError("episode outcome failed its sealed validator")
    return outcome


def _arm_observation(
    arm: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    side: str,
    method: str,
    episode_role: str,
    pair: str,
    candles: Sequence[S5BidAskCandle],
) -> dict[str, Any]:
    reason = str(arm["exit_reason"])
    fill_at = _parse_optional_utc(arm.get("fill_at_utc"))
    exit_at = _parse_optional_utc(arm.get("exit_at_utc"))
    fill_interval = _s5_interval(fill_at)
    exit_interval: dict[str, str] | None = None
    exit_interval_semantics: str | None = None
    mark_price: float | None = None
    mark_pips: float | None = None
    mark_side: str | None = None
    mark_interval: dict[str, str] | None = None
    observed_state = "NO_FILL" if fill_at is None else "CLOSED_BEFORE_HORIZON"
    if reason == "HORIZON_FULL_STOP_LOSS":
        if fill_at is None or exit_at is None:
            raise ValueError("horizon full-stop outcome has no fill/exit clock")
        eligible = [
            candle
            for candle in candles
            if fill_at <= candle.timestamp_utc < exit_at
        ]
        if not eligible:
            raise ValueError("open-at-horizon arm has no executable close mark")
        mark_candle = eligible[-1]
        mark_price = float(mark_candle.bid_c if side == "LONG" else mark_candle.ask_c)
        mark_side = "BID" if side == "LONG" else "ASK"
        entry = float(arm["entry"])
        pip_factor = float(instrument_pip_factor(pair))
        mark_pips = _round(
            (mark_price - entry) * pip_factor
            if side == "LONG"
            else (entry - mark_price) * pip_factor
        )
        mark_interval = _s5_interval(mark_candle.timestamp_utc)
        exit_interval = mark_interval
        exit_interval_semantics = "LAST_EXECUTABLE_MARK_ONLY_NOT_PROOF_EXIT"
        observed_state = "OPEN_AT_HORIZON"
    elif exit_at is not None:
        exit_interval = _s5_interval(exit_at)
        exit_interval_semantics = "EXECUTABLE_ATTACHED_EXIT_TOUCH"
    body = {
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_sha256": str(candidate["candidate_sha256"]),
        "side": side,
        "method": method,
        "episode_role": episode_role,
        "arm_id": str(arm["arm_id"]),
        "arm_outcome_sha256": str(arm["arm_outcome_sha256"]),
        "filled": arm.get("filled") is True,
        "fill_s5_interval_utc": fill_interval,
        "proof_exit_reason": reason,
        "proof_exit_at_utc": arm.get("exit_at_utc"),
        "proof_post_cost_realized_pips": float(
            arm["post_cost_realized_pips"]
        ),
        "exit_s5_interval_utc": exit_interval,
        "exit_s5_interval_semantics": exit_interval_semantics,
        "observed_position_state": observed_state,
        "observed_horizon_mark_price_side": mark_side,
        "observed_horizon_mark_price": mark_price,
        "observed_horizon_mark_post_cost_pips": mark_pips,
        "observed_horizon_mark_s5_interval_utc": mark_interval,
        "proof_and_observed_mark_must_not_be_mixed": True,
    }
    return {**body, "observation_sha256": _canonical_sha(body)}


def _truth_candle_receipt(candle: S5BidAskCandle) -> dict[str, Any]:
    return {
        "timestamp_utc": candle.timestamp_utc.isoformat(),
        "bid": [candle.bid_o, candle.bid_h, candle.bid_l, candle.bid_c],
        "ask": [candle.ask_o, candle.ask_h, candle.ask_l, candle.ask_c],
    }


def _truth_candles_from_receipts(value: Any) -> list[S5BidAskCandle] | None:
    if not isinstance(value, list):
        return None
    candles: list[S5BidAskCandle] = []
    try:
        for row in value:
            if not isinstance(row, Mapping) or set(row) != {
                "timestamp_utc",
                "bid",
                "ask",
            }:
                return None
            bid = row["bid"]
            ask = row["ask"]
            if not isinstance(bid, list) or not isinstance(ask, list):
                return None
            if len(bid) != 4 or len(ask) != 4:
                return None
            candles.append(
                S5BidAskCandle(
                    timestamp_utc=_parse_utc(row["timestamp_utc"]),
                    bid_o=float(bid[0]),
                    bid_h=float(bid[1]),
                    bid_l=float(bid[2]),
                    bid_c=float(bid[3]),
                    ask_o=float(ask[0]),
                    ask_h=float(ask[1]),
                    ask_l=float(ask[2]),
                    ask_c=float(ask[3]),
                )
            )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return candles


def build_fast_bot_episode_scorecard(
    vehicles: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    as_of_utc: datetime,
) -> dict[str, Any]:
    """Aggregate route-vs-inverse pairs with one vote per episode/cluster."""

    as_of = _aware_utc(as_of_utc)
    vehicle_by_sha = {
        str(vehicle["contract_sha256"]): vehicle
        for vehicle in vehicles
        if _vehicle_valid(vehicle)
    }
    resolved: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    seen_episode_ids: set[str] = set()
    for outcome in outcomes:
        vehicle = vehicle_by_sha.get(str(outcome.get("vehicle_contract_sha256") or ""))
        if vehicle is None or not _episode_outcome_valid(outcome, vehicle):
            continue
        episode_id = str(vehicle["episode_id"])
        if episode_id in seen_episode_ids:
            raise ValueError("multiple current outcomes for one episode")
        seen_episode_ids.add(episode_id)
        resolved.append((vehicle, outcome))

    aggregates: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for vehicle, outcome in resolved:
        observations = {
            (
                str(row["episode_role"]),
                str(row["method"]),
                str(row["arm_id"]),
            ): row
            for row in outcome["arm_observations"]
        }
        for method in vehicle["route_candidate_methods"]:
            arm_ids = sorted(
                {
                    str(row["arm_id"])
                    for row in outcome["arm_observations"]
                    if row["episode_role"] == ROUTE_ALIGNED
                    and row["method"] == method
                }
            )
            for arm_id in arm_ids:
                aligned = observations.get((ROUTE_ALIGNED, method, arm_id))
                inverse = observations.get((INVERSE_DIRECTION, method, arm_id))
                if aligned is None or inverse is None:
                    raise ValueError("route/inverse episode pair is incomplete")
                key = (
                    str(vehicle["branch_outcome"]),
                    str(vehicle["attempt_direction"]),
                    str(vehicle["route_family"]),
                    str(method),
                    arm_id,
                )
                row = aggregates.setdefault(
                    key,
                    {
                        "all_episode_ids": set(),
                        "late_episode_ids": set(),
                        "eligible_episode_ids": set(),
                        "proof_route": [],
                        "proof_inverse": [],
                        "proof_diff": [],
                        "mark_route": [],
                        "mark_inverse": [],
                        "mark_diff": [],
                    },
                )
                episode_id = str(vehicle["episode_id"])
                if episode_id in row["all_episode_ids"]:
                    raise ValueError("episode would inflate one scorecard cluster")
                row["all_episode_ids"].add(episode_id)
                if vehicle.get("scorecard_eligible") is not True:
                    row["late_episode_ids"].add(episode_id)
                    continue
                row["eligible_episode_ids"].add(episode_id)
                route_proof = float(aligned["proof_post_cost_realized_pips"])
                inverse_proof = float(inverse["proof_post_cost_realized_pips"])
                row["proof_route"].append(route_proof)
                row["proof_inverse"].append(inverse_proof)
                row["proof_diff"].append(_round(route_proof - inverse_proof))
                route_mark = aligned.get("observed_horizon_mark_post_cost_pips")
                inverse_mark = inverse.get("observed_horizon_mark_post_cost_pips")
                if route_mark is not None and inverse_mark is not None:
                    route_mark_number = float(route_mark)
                    inverse_mark_number = float(inverse_mark)
                    row["mark_route"].append(route_mark_number)
                    row["mark_inverse"].append(inverse_mark_number)
                    row["mark_diff"].append(
                        _round(route_mark_number - inverse_mark_number)
                    )

    clusters = []
    for key in sorted(aggregates):
        row = aggregates[key]
        clusters.append(
            {
                "branch_outcome": key[0],
                "attempt_direction": key[1],
                "route_family": key[2],
                "route_method": key[3],
                "arm_id": key[4],
                "episode_count": len(row["all_episode_ids"]),
                "scorecard_eligible_episode_count": len(
                    row["eligible_episode_ids"]
                ),
                "diagnostic_late_episode_count": len(row["late_episode_ids"]),
                "proof_score": {
                    "basis": "CONSERVATIVE_LEARNING_SCORE",
                    "paired_episode_count": len(row["proof_diff"]),
                    "route_mean_post_cost_pips": _mean(row["proof_route"]),
                    "inverse_mean_post_cost_pips": _mean(row["proof_inverse"]),
                    "route_minus_inverse_mean_pips": _mean(row["proof_diff"]),
                    "route_better_rate": _positive_rate(row["proof_diff"]),
                },
                "observed_open_horizon_mark": {
                    "basis": "LAST_EXECUTABLE_BID_ASK_CLOSE",
                    "paired_episode_count": len(row["mark_diff"]),
                    "route_mean_post_cost_pips": _mean(row["mark_route"]),
                    "inverse_mean_post_cost_pips": _mean(row["mark_inverse"]),
                    "route_minus_inverse_mean_pips": _mean(row["mark_diff"]),
                    "route_better_rate": _positive_rate(row["mark_diff"]),
                },
                "proof_and_observed_mark_must_not_be_mixed": True,
            }
        )
    (
        technical_hypothesis_clusters,
        technical_hypothesis_unscored,
        no_trade_control,
    ) = (
        _build_technical_hypothesis_scorecard(resolved)
    )
    body = {
        "contract": SCORECARD_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": as_of.isoformat(),
        "status": "DIAGNOSTIC_ONLY" if resolved else "NO_RESOLVED_EPISODES",
        "vehicle_count": len(vehicle_by_sha),
        "resolved_episode_count": len(seen_episode_ids),
        "scorecard_eligible_episode_count": len(
            {
                str(vehicle["episode_id"])
                for vehicle, _ in resolved
                if vehicle.get("scorecard_eligible") is True
            }
        ),
        "diagnostic_late_episode_count": len(
            {
                str(vehicle["episode_id"])
                for vehicle, _ in resolved
                if vehicle.get("scorecard_eligible") is not True
            }
        ),
        "cluster_count": len(clusters),
        "cluster_identity": (
            "BRANCH_ATTEMPT_ROUTE_FAMILY_METHOD_ARM_EPISODE_UNIQUE_V1"
        ),
        "clusters": clusters,
        "technical_hypothesis_cluster_count": len(
            technical_hypothesis_clusters
        ),
        "technical_hypothesis_clusters": technical_hypothesis_clusters,
        "technical_hypothesis_unscored_count": len(
            technical_hypothesis_unscored
        ),
        "technical_hypothesis_unscored": technical_hypothesis_unscored,
        "no_trade_control": no_trade_control,
        "technical_forecast_values_status": (
            "PENDING_PROSPECTIVE_FORWARD_CALIBRATION"
        ),
        "technical_cluster_statistics_are_forecast_probabilities": False,
        "proof_score_semantics": (
            "LEARNING_CONSERVATIVE_FULL_SL_IF_OPEN_AT_HORIZON"
        ),
        "observed_mark_semantics": (
            "SEPARATE_LAST_EXECUTABLE_BID_ASK_CLOSE_NOT_PROOF_SCORE"
        ),
        "proof_and_observed_mark_must_not_be_mixed": True,
        "late_episodes_are_diagnostic_only": True,
        "diagnostic_only": True,
        "order_authority": "NONE",
        "automatic_promotion_allowed": False,
        "promotion_allowed": False,
        "primary_effect": False,
        "risk_effect": False,
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_intents": [],
    }
    return _seal(body)


def _build_technical_hypothesis_scorecard(
    resolved: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Join only compatible proxy vehicles; retain all other rows as unscored."""

    aggregates: dict[str, dict[str, Any]] = {}
    unscored: dict[str, dict[str, Any]] = {}
    controls: dict[str, dict[str, Any]] = {}

    def add_episode(
        row: dict[str, Any], *, episode_id: str, eligible: bool, label: str
    ) -> None:
        if episode_id in row["all_episode_ids"]:
            raise ValueError(f"episode would inflate {label}")
        row["all_episode_ids"].add(episode_id)
        row[
            "eligible_episode_ids" if eligible else "late_episode_ids"
        ].add(episode_id)

    for vehicle, outcome in resolved:
        shadow = vehicle.get("technical_hypothesis_shadow")
        if not isinstance(shadow, Mapping):
            raise ValueError("resolved vehicle has no technical hypothesis shadow")
        hypotheses = shadow.get("hypotheses")
        if not isinstance(hypotheses, list):
            raise ValueError("technical hypothesis shadow has no rows")
        observations = [
            row
            for row in outcome["arm_observations"]
            if isinstance(row, Mapping)
        ]
        episode_id = str(vehicle["episode_id"])
        eligible = vehicle.get("scorecard_eligible") is True
        catalog_sha = str(shadow.get("catalog_contract_sha256") or "")
        evaluator_policy = str(shadow.get("evaluator_policy") or "")
        arm_policy = str(vehicle.get("learning_arm_policy") or "")
        for hypothesis in hypotheses:
            if not isinstance(hypothesis, Mapping):
                raise ValueError("technical hypothesis row is invalid")
            hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
            hypothesis_status = str(hypothesis.get("status") or "")
            family = str(hypothesis.get("family") or "")
            side = str(hypothesis.get("predicted_side") or "")
            method = str(hypothesis.get("execution_method") or "")
            entry_vehicle = str(hypothesis.get("entry_vehicle") or "")
            join_policy = str(hypothesis.get("arm_truth_join_policy") or "")
            if hypothesis_id == "H08":
                identity = {
                    "catalog_contract_sha256": catalog_sha,
                    "evaluator_policy": evaluator_policy,
                    "hypothesis_status": hypothesis_status,
                }
                identity_key = json.dumps(identity, sort_keys=True)
                state = controls.setdefault(
                    identity_key,
                    {
                        "identity": identity,
                        "all_episode_ids": set(),
                        "eligible_episode_ids": set(),
                        "late_episode_ids": set(),
                    },
                )
                add_episode(
                    state,
                    episode_id=episode_id,
                    eligible=eligible,
                    label="no-trade control",
                )
                continue

            unscored_reason: str | None = None
            if join_policy != "EXISTING_PASSIVE_QUOTE_PROXY":
                unscored_reason = "HYPOTHESIS_SPECIFIC_ENTRY_VEHICLE_NOT_IMPLEMENTED"
            elif side not in {"LONG", "SHORT"} or not method:
                unscored_reason = "PREDICTED_SIDE_OR_METHOD_UNAVAILABLE"
            matched = (
                [
                    observation
                    for observation in observations
                    if observation.get("side") == side
                    and observation.get("method") == method
                ]
                if unscored_reason is None
                else []
            )
            if unscored_reason is None and not matched:
                unscored_reason = "NO_MATCHING_PASSIVE_ARM_CELL"
            if unscored_reason is not None:
                identity = {
                    "catalog_contract_sha256": catalog_sha,
                    "evaluator_policy": evaluator_policy,
                    "learning_arm_policy": arm_policy,
                    "hypothesis_id": hypothesis_id,
                    "hypothesis_status": hypothesis_status,
                    "family": family,
                    "predicted_side": side or None,
                    "execution_method": method or None,
                    "entry_vehicle": entry_vehicle,
                    "arm_truth_join_policy": join_policy,
                    "unscored_reason": unscored_reason,
                }
                identity_key = json.dumps(identity, sort_keys=True)
                state = unscored.setdefault(
                    identity_key,
                    {
                        "identity": identity,
                        "all_episode_ids": set(),
                        "eligible_episode_ids": set(),
                        "late_episode_ids": set(),
                    },
                )
                add_episode(
                    state,
                    episode_id=episode_id,
                    eligible=eligible,
                    label="one unscored technical hypothesis cluster",
                )
                continue
            for observation in matched:
                identity = {
                    "catalog_contract_sha256": catalog_sha,
                    "evaluator_policy": evaluator_policy,
                    "learning_arm_policy": arm_policy,
                    "hypothesis_id": hypothesis_id,
                    "hypothesis_status": hypothesis_status,
                    "family": family,
                    "predicted_side": side,
                    "execution_method": method,
                    "entry_vehicle": entry_vehicle,
                    "arm_truth_join_policy": join_policy,
                    "episode_role": str(observation["episode_role"]),
                    "arm_id": str(observation["arm_id"]),
                }
                identity_key = json.dumps(identity, sort_keys=True)
                row = aggregates.setdefault(
                    identity_key,
                    {
                        "identity": identity,
                        "all_episode_ids": set(),
                        "eligible_episode_ids": set(),
                        "late_episode_ids": set(),
                        "proof_pips": [],
                        "observed_mark_pips": [],
                        "filled": [],
                        "tp_given_fill": [],
                    },
                )
                add_episode(
                    row,
                    episode_id=episode_id,
                    eligible=eligible,
                    label="one technical hypothesis cluster",
                )
                if not eligible:
                    continue
                proof_pips = float(observation["proof_post_cost_realized_pips"])
                row["proof_pips"].append(proof_pips)
                filled = observation.get("filled") is True
                row["filled"].append(1.0 if filled else 0.0)
                if filled:
                    row["tp_given_fill"].append(
                        1.0
                        if observation.get("proof_exit_reason") == "TAKE_PROFIT"
                        else 0.0
                    )
                mark = observation.get("observed_horizon_mark_post_cost_pips")
                if mark is not None:
                    row["observed_mark_pips"].append(float(mark))

    clusters: list[dict[str, Any]] = []
    for identity_key in sorted(aggregates):
        row = aggregates[identity_key]
        clusters.append(
            {
                **row["identity"],
                "episode_count": len(row["all_episode_ids"]),
                "scorecard_eligible_episode_count": len(
                    row["eligible_episode_ids"]
                ),
                "diagnostic_late_episode_count": len(row["late_episode_ids"]),
                "observed_forward_statistics": {
                    "basis": (
                        "EXISTING_PASSIVE_QUOTE_ARM_PROXY_EXACT_S5_BID_ASK"
                    ),
                    "sample_count": len(row["proof_pips"]),
                    "fill_rate": _mean(row["filled"]),
                    "tp_before_sl_rate_given_fill": _mean(
                        row["tp_given_fill"]
                    ),
                    "mean_post_cost_pips": _mean(row["proof_pips"]),
                    "net_positive_rate": _positive_rate(row["proof_pips"]),
                    "open_horizon_mark_sample_count": len(
                        row["observed_mark_pips"]
                    ),
                    "open_horizon_mark_mean_post_cost_pips": _mean(
                        row["observed_mark_pips"]
                    ),
                },
                "strategy_entry_vehicle_exact_match": False,
                "statistics_scope": "DIRECTION_METHOD_PROXY_NOT_FULL_STRATEGY",
                "observed_statistics_are_forecast_probabilities": False,
                "forecast_values_status": (
                    "PENDING_PROSPECTIVE_FORWARD_CALIBRATION"
                ),
            }
        )
    unscored_rows = [
        {
            **row["identity"],
            "episode_count": len(row["all_episode_ids"]),
            "scorecard_eligible_episode_count": len(
                row["eligible_episode_ids"]
            ),
            "diagnostic_late_episode_count": len(row["late_episode_ids"]),
            "pnl_joined": False,
        }
        for _, row in sorted(unscored.items())
    ]
    control_rows = [
        {
            **row["identity"],
            "episode_count": len(row["all_episode_ids"]),
            "scorecard_eligible_episode_count": len(
                row["eligible_episode_ids"]
            ),
            "diagnostic_late_episode_count": len(row["late_episode_ids"]),
        }
        for _, row in sorted(controls.items())
    ]
    control = {
        "hypothesis_id": "H08",
        "basis": "ZERO_PNL_NO_TRADE_CONTROL",
        "rows": control_rows,
        "pnl_joined": False,
    }
    return clusters, unscored_rows, control


def _build_technical_feature_snapshot(
    handoff: Mapping[str, Any], *, pair: str
) -> dict[str, Any]:
    cycle = _parse_utc(handoff["cycle_generated_at_utc"])
    views: dict[str, Mapping[str, Any]] = {}
    # Match the episode merge order: the current/fast packet owns a timeframe
    # when both packets contain it.
    for payload_name in ("slow_pair_charts", "fast_pair_charts"):
        payload = handoff.get(payload_name)
        if not isinstance(payload, Mapping):
            raise ValueError("episode handoff chart packet is invalid")
        charts = payload.get("charts")
        if not isinstance(charts, list):
            raise ValueError("episode handoff chart list is invalid")
        pair_charts = [
            chart
            for chart in charts
            if isinstance(chart, Mapping) and chart.get("pair") == pair
        ]
        if len(pair_charts) > 1:
            raise ValueError("episode handoff has duplicate pair charts")
        if not pair_charts:
            continue
        raw_views = pair_charts[0].get("views")
        if not isinstance(raw_views, list):
            raise ValueError("episode handoff pair views are invalid")
        for raw_view in raw_views:
            if not isinstance(raw_view, Mapping):
                continue
            timeframe = str(raw_view.get("granularity") or "").upper()
            if timeframe in TIMEFRAMES:
                views[timeframe] = raw_view
    if set(views) != set(TIMEFRAMES):
        raise ValueError("episode feature snapshot requires all seven timeframes")

    rows: list[dict[str, Any]] = []
    for timeframe in TIMEFRAMES:
        view = views[timeframe]
        candles = view.get("recent_candles")
        if not isinstance(candles, list):
            raise ValueError("episode feature view has no candle sequence")
        complete_starts = [
            _parse_utc(candle.get("t"))
            for candle in candles
            if isinstance(candle, Mapping) and candle.get("complete") is True
        ]
        if not complete_starts:
            raise ValueError("episode feature view has no complete candle")
        candle_close = max(complete_starts) + timedelta(
            seconds=TIMEFRAME_SECONDS[timeframe]
        )
        if candle_close > cycle:
            raise ValueError("episode feature candle clock exceeds the handoff cycle")
        raw_market = view.get("market_state")
        market = raw_market if isinstance(raw_market, Mapping) else {}
        market_features: dict[str, Any] = {}
        for key, allowed in MARKET_STATE_ENUM_FEATURES.items():
            raw = market.get(key)
            if raw is None:
                continue
            if not isinstance(raw, str) or raw not in allowed:
                raise ValueError(f"invalid episode market-state feature: {key}")
            market_features[key] = raw
        if "confidence" in market:
            confidence = _finite_feature_number(market.get("confidence"))
            if confidence is None or not 0.0 <= float(confidence) <= 1.0:
                raise ValueError("invalid episode market-state confidence")
            market_features["confidence"] = confidence
        if "evidence_complete" in market:
            evidence_complete = market.get("evidence_complete")
            if not isinstance(evidence_complete, bool):
                raise ValueError("invalid episode market-state evidence flag")
            market_features["evidence_complete"] = evidence_complete

        raw_indicators = view.get("indicators")
        indicators = raw_indicators if isinstance(raw_indicators, Mapping) else {}
        indicator_features: dict[str, Any] = {}
        for key in sorted(NUMERIC_INDICATOR_FEATURES):
            if key not in indicators:
                continue
            parsed = _finite_feature_number(indicators.get(key))
            if parsed is None:
                raise ValueError(f"invalid episode numeric feature: {key}")
            indicator_features[key] = parsed
        for key in sorted(ENUM_INDICATOR_FEATURES):
            if key not in indicators:
                continue
            parsed = _bounded_enum(indicators.get(key))
            if parsed is None:
                raise ValueError(f"invalid episode enum feature: {key}")
            indicator_features[key] = parsed

        raw_series = view.get("indicator_series")
        if raw_series is None:
            series_source: Mapping[str, Any] = {}
        elif isinstance(raw_series, Mapping):
            series_source = raw_series
        else:
            raise ValueError("invalid episode indicator-series packet")
        series_features: dict[str, list[int | float]] = {}
        for key in sorted(INDICATOR_SERIES_FEATURES):
            if key not in series_source:
                continue
            raw_values = series_source.get(key)
            if not isinstance(raw_values, (list, tuple)):
                raise ValueError(f"invalid episode indicator series: {key}")
            values: list[int | float] = []
            for raw_value in raw_values[-MAX_INDICATOR_SERIES_VALUES:]:
                parsed = _finite_feature_number(raw_value)
                if parsed is None:
                    raise ValueError(f"invalid episode indicator series: {key}")
                values.append(parsed)
            series_features[key] = values
        row_body = {
            "timeframe": timeframe,
            "complete_candle_close_utc": candle_close.isoformat(),
            "market_state": market_features,
            "indicators": indicator_features,
            "indicator_series": series_features,
        }
        rows.append({**row_body, "feature_sha256": _canonical_sha(row_body)})
    body = {
        "contract": "QR_FAST_BOT_EPISODE_TECHNICAL_FEATURE_SNAPSHOT_V1",
        "schema_version": 1,
        "pair": pair,
        "handoff_cycle_generated_at_utc": cycle.isoformat(),
        "feature_allowlist_version": 1,
        "timeframes": rows,
        "hypothesis_families": list(HYPOTHESIS_FAMILIES),
        "raw_chart_packet_embedded": False,
        "diagnostic_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    snapshot = _seal(body)
    if len(_canonical_json_bytes(snapshot)) > MAX_TECHNICAL_FEATURE_SNAPSHOT_BYTES:
        raise ValueError("episode technical feature snapshot exceeds its byte cap")
    return snapshot


def _technical_feature_snapshot_valid(
    value: Any, *, pair: str, cycle: str
) -> bool:
    if not isinstance(value, Mapping) or not _sealed_valid(
        value, "QR_FAST_BOT_EPISODE_TECHNICAL_FEATURE_SNAPSHOT_V1"
    ):
        return False
    rows = value.get("timeframes")
    if not bool(
        value.get("schema_version") == 1
        and value.get("pair") == pair
        and value.get("handoff_cycle_generated_at_utc") == cycle
        and value.get("feature_allowlist_version") == 1
        and value.get("hypothesis_families") == list(HYPOTHESIS_FAMILIES)
        and value.get("raw_chart_packet_embedded") is False
        and value.get("diagnostic_only") is True
        and value.get("order_authority") == "NONE"
        and value.get("live_permission") is False
        and value.get("broker_mutation_allowed") is False
        and isinstance(rows, list)
        and len(rows) == len(TIMEFRAMES)
        and len(_canonical_json_bytes(value)) <= MAX_TECHNICAL_FEATURE_SNAPSHOT_BYTES
    ):
        return False
    seen: set[str] = set()
    try:
        cycle_clock = _parse_utc(cycle)
        for row in rows:
            if not isinstance(row, Mapping):
                return False
            timeframe = str(row.get("timeframe") or "")
            close = _parse_utc(row["complete_candle_close_utc"])
            market = row.get("market_state")
            indicators = row.get("indicators")
            indicator_series = row.get("indicator_series")
            body = {
                key: item for key, item in row.items() if key != "feature_sha256"
            }
            if not bool(
                timeframe in TIMEFRAMES
                and timeframe not in seen
                and close <= cycle_clock
                and row.get("feature_sha256") == _canonical_sha(body)
                and isinstance(market, Mapping)
                and set(market).issubset(MARKET_STATE_FEATURES)
                and isinstance(indicators, Mapping)
                and set(indicators).issubset(
                    NUMERIC_INDICATOR_FEATURES | ENUM_INDICATOR_FEATURES
                )
                and isinstance(indicator_series, Mapping)
                and set(indicator_series).issubset(INDICATOR_SERIES_FEATURES)
            ):
                return False
            for key, item in market.items():
                if key in MARKET_STATE_ENUM_FEATURES:
                    if not isinstance(item, str) or item not in MARKET_STATE_ENUM_FEATURES[key]:
                        return False
                elif key == "confidence":
                    parsed = _finite_feature_number(item)
                    if parsed is None or not 0.0 <= float(parsed) <= 1.0:
                        return False
                elif key == "evidence_complete":
                    if not isinstance(item, bool):
                        return False
                else:
                    return False
            for key, item in indicators.items():
                if key in NUMERIC_INDICATOR_FEATURES:
                    if _finite_feature_number(item) != item:
                        return False
                elif _bounded_enum(item) != item:
                    return False
            for values in indicator_series.values():
                if not isinstance(values, list) or len(values) > MAX_INDICATOR_SERIES_VALUES:
                    return False
                if any(_finite_feature_number(item) != item for item in values):
                    return False
            seen.add(timeframe)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return seen == set(TIMEFRAMES)


def _finite_feature_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or abs(parsed) > 1_000_000_000_000.0:
        return None
    return int(value) if isinstance(value, int) else parsed


def _bounded_enum(value: Any) -> str | None:
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    normalized = value.upper()
    if value != normalized or len(value) > 64:
        return None
    return value


def _validate_v2_handoff(value: Mapping[str, Any]) -> None:
    shadow = value.get("prospective_vehicle_shadow")
    snapshot = value.get("broker_snapshot")
    regime = value.get("regime_contract")
    fast_charts = value.get("fast_pair_charts")
    slow_charts = value.get("slow_pair_charts")
    if not bool(
        isinstance(value, Mapping)
        and value.get("contract") == EPISODE_HANDOFF_V2_CONTRACT
        and value.get("schema_version").__class__ is int
        and value.get("schema_version") == 2
        and _sealed_valid(value, EPISODE_HANDOFF_V2_CONTRACT)
        and isinstance(shadow, Mapping)
        and _sealed_valid(shadow, LEARNING_SHADOW_CONTRACT)
        and isinstance(snapshot, Mapping)
        and isinstance(regime, Mapping)
        and _sealed_valid(regime, "QR_HIERARCHICAL_BOT_REGIME_V1")
        and isinstance(fast_charts, Mapping)
        and isinstance(slow_charts, Mapping)
        and value.get("broker_snapshot_sha256") == _canonical_sha(snapshot)
        and value.get("regime_contract_sha256") == regime.get("contract_sha256")
        and value.get("fast_pair_charts_sha256") == _canonical_sha(fast_charts)
        and value.get("slow_pair_charts_sha256") == _canonical_sha(slow_charts)
        and isinstance(regime.get("sources"), Mapping)
        and regime["sources"].get("broker_snapshot_sha256")
        == value.get("broker_snapshot_sha256")
        and regime["sources"].get("fast_pair_charts_sha256")
        == value.get("fast_pair_charts_sha256")
        and regime["sources"].get("slow_pair_charts_sha256")
        == value.get("slow_pair_charts_sha256")
        and value.get("prospective_vehicle_shadow_sha256")
        == shadow.get("contract_sha256")
        and shadow.get("generated_at_utc") == value.get("cycle_generated_at_utc")
        and shadow.get("regime_contract_sha256")
        == value.get("regime_contract_sha256")
        and shadow.get("broker_snapshot_sha256")
        == value.get("broker_snapshot_sha256")
        and value.get("diagnostic_only") is True
        and value.get("order_authority") == "NONE"
        and value.get("shadow_only") is True
        and value.get("live_permission") is False
        and value.get("broker_mutation_allowed") is False
        and shadow.get("order_authority") == "NONE"
        and shadow.get("live_permission") is False
        and shadow.get("broker_mutation_allowed") is False
    ):
        raise ValueError("invalid V2 episode vehicle handoff")
    if _parse_utc(value["cycle_generated_at_utc"]) != _parse_utc(
        regime["generated_at_utc"]
    ):
        raise ValueError("V2 episode handoff cycle/regime binding is invalid")
    seats = shadow.get("seats")
    if not isinstance(seats, list) or any(
        not isinstance(seat, Mapping) or not _learning_seat_deep_valid(seat)
        for seat in seats
    ):
        raise ValueError("V2 episode handoff contains an invalid prospective seat")
    if (
        shadow.get("seat_count") != len(seats)
        or shadow.get("candidate_count")
        != sum(len(seat["candidates"]) for seat in seats)
        or len({str(seat["pair"]) for seat in seats}) != len(seats)
    ):
        raise ValueError("V2 episode handoff prospective seat index is invalid")


def _vehicle_valid(value: Mapping[str, Any]) -> bool:
    try:
        if not _sealed_valid(value, VEHICLE_CONTRACT):
            return False
        seat = value["learning_seat"]
        technical_features = value["technical_feature_snapshot"]
        technical_hypotheses = value["technical_hypothesis_shadow"]
        roles = value["candidate_roles"]
        route_methods = value["route_candidate_methods"]
        source_binding = value["source_binding"]
        maturity = _parse_utc(value["maturity_at_utc"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    identity = {
        "vehicle_policy": VEHICLE_POLICY,
        "confirmed_event_sha256": value.get("confirmed_event_sha256"),
    }
    input_proof_eligible = (
        seat.get("causal_input_proof_eligible") is True
        if seat.get("selection_policy") == LEARNING_SELECTION_POLICY
        else True
    )
    expected_ineligibility_reasons = []
    if value.get("late_detected") is True:
        expected_ineligibility_reasons.append("LATE_DETECTED_EPISODE")
    if not input_proof_eligible:
        expected_ineligibility_reasons.append(
            "INPUT_BLOCKED_SHADOW_DIAGNOSTIC_ONLY"
        )
    legacy_eligibility_contract = all(
        key not in value
        for key in (
            "causal_input_proof_eligible",
            "scorecard_ineligibility_reasons",
        )
    )
    eligibility_contract_valid = bool(
        (
            legacy_eligibility_contract
            and input_proof_eligible
            and value.get("scorecard_eligible")
            is (value.get("late_detected") is not True)
        )
        or (
            not legacy_eligibility_contract
            and value.get("causal_input_proof_eligible")
            is input_proof_eligible
            and value.get("scorecard_eligible")
            is (not expected_ineligibility_reasons)
            and value.get("scorecard_ineligibility_reasons")
            == expected_ineligibility_reasons
        )
    )
    if not bool(
        value.get("schema_version") == 1
        and value.get("vehicle_policy") == VEHICLE_POLICY
        and value.get("vehicle_id") == _canonical_sha(identity)[:24]
        and _sha_text(value.get("confirmed_event_sha256"))
        and isinstance(seat, Mapping)
        and _learning_seat_deep_valid(seat)
        and value.get("learning_seat_contract_sha256")
        == seat.get("contract_sha256")
        and value.get("learning_seat_id") == seat.get("seat_id")
        and _technical_feature_snapshot_valid(
            technical_features,
            pair=str(value.get("pair") or ""),
            cycle=str(value.get("handoff_cycle_generated_at_utc") or ""),
        )
        and value.get("technical_feature_snapshot_sha256")
        == _canonical_sha(technical_features)
        and source_binding.get("technical_feature_snapshot_sha256")
        == value.get("technical_feature_snapshot_sha256")
        and technical_hypothesis_shadow_valid(
            technical_hypotheses,
            feature_snapshot=technical_features,
            attempt_direction=str(value.get("attempt_direction") or ""),
            branch_outcome=str(value.get("branch_outcome") or ""),
            route_family=str(value.get("route_family") or ""),
            spread_pips=float(seat["executable_spread_pips"]),
            m5_atr_pips=float(seat["m5_atr_pips"]),
            spread_to_m5_atr=float(seat["spread_to_m5_atr"]),
        )
        and value.get("technical_hypothesis_shadow_sha256")
        == technical_hypotheses.get("contract_sha256")
        and value.get("technical_hypothesis_catalog_sha256")
        == technical_hypotheses.get("catalog_contract_sha256")
        and value.get("technical_cost_state_sha256")
        == technical_hypotheses.get("cost_state_sha256")
        and value.get("technical_hypothesis_evaluator_policy")
        == technical_hypotheses.get("evaluator_policy")
        and value.get("technical_hypothesis_evaluator_policy_sha256")
        == _canonical_sha(
            {
                "evaluator_policy": technical_hypotheses.get(
                    "evaluator_policy"
                )
            }
        )
        and source_binding.get("technical_hypothesis_shadow_sha256")
        == value.get("technical_hypothesis_shadow_sha256")
        and source_binding.get("technical_hypothesis_catalog_sha256")
        == value.get("technical_hypothesis_catalog_sha256")
        and source_binding.get("technical_cost_state_sha256")
        == value.get("technical_cost_state_sha256")
        and source_binding.get(
            "technical_hypothesis_evaluator_policy_sha256"
        )
        == value.get("technical_hypothesis_evaluator_policy_sha256")
        and value.get("technical_feature_hypothesis_families")
        == list(HYPOTHESIS_FAMILIES)
        and value.get("learning_arm_policy") == seat.get("arm_policy")
        and value.get("pair") == seat.get("pair")
        and value.get("candidate_count") == len(seat["candidates"])
        and value.get("arm_count")
        == sum(len(candidate["arms"]) for candidate in seat["candidates"])
        and isinstance(roles, list)
        and len(roles) == len(seat["candidates"])
        and isinstance(route_methods, list)
        and route_methods
        and isinstance(source_binding, Mapping)
        and set(source_binding)
        == {
            "confirmed_event_sha256",
            "handoff_contract_sha256",
            "broker_snapshot_sha256",
            "regime_contract_sha256",
            "prospective_vehicle_shadow_sha256",
            "learning_seat_contract_sha256",
            "technical_feature_snapshot_sha256",
            "technical_hypothesis_catalog_sha256",
            "technical_hypothesis_shadow_sha256",
            "technical_cost_state_sha256",
            "technical_hypothesis_evaluator_policy_sha256",
        }
        and value.get("source_binding_sha256") == _canonical_sha(source_binding)
        and source_binding.get("confirmed_event_sha256")
        == value.get("confirmed_event_sha256")
        and source_binding.get("learning_seat_contract_sha256")
        == seat.get("contract_sha256")
        and source_binding.get("broker_snapshot_sha256")
        == seat.get("broker_snapshot_sha256")
        and source_binding.get("regime_contract_sha256")
        == seat.get("regime_contract_sha256")
        and _sha_text(source_binding.get("handoff_contract_sha256"))
        and _sha_text(source_binding.get("prospective_vehicle_shadow_sha256"))
        and maturity == _seat_maturity(seat)
        and value.get("learning_scoring_policy") == LEARNING_SCORING_POLICY
        and eligibility_contract_valid
        and _authority_is_zero(value)
    ):
        return False
    expected_candidates = {
        str(candidate["candidate_sha256"]): candidate
        for candidate in seat["candidates"]
    }
    seen: set[str] = set()
    for role in roles:
        if not isinstance(role, Mapping):
            return False
        candidate_sha = str(role.get("candidate_sha256") or "")
        candidate = expected_candidates.get(candidate_sha)
        if candidate is None or candidate_sha in seen:
            return False
        expected_role = _episode_role(
            side=str(candidate["side"]),
            method=str(candidate["method"]),
            trade_side=str(value["trade_side"]),
            route_methods=[str(item) for item in route_methods],
        )
        if dict(role) != {
            "candidate_id": str(candidate["candidate_id"]),
            "candidate_sha256": candidate_sha,
            "side": str(candidate["side"]),
            "method": str(candidate["method"]),
            "episode_role": expected_role,
        }:
            return False
        seen.add(candidate_sha)
    return len(seen) == len(expected_candidates)


def _vehicle_matches_event(
    vehicle: Mapping[str, Any], event: Mapping[str, Any]
) -> bool:
    route = event.get("route")
    anchor = event.get("anchor")
    observation = event.get("observation")
    return bool(
        event.get("state") == "CONFIRMED"
        and isinstance(route, Mapping)
        and isinstance(anchor, Mapping)
        and isinstance(observation, Mapping)
        and vehicle.get("episode_id") == event.get("episode_id")
        and vehicle.get("confirmed_event_id") == event.get("event_id")
        and vehicle.get("confirmed_event_sha256") == event.get("event_sha256")
        and vehicle.get("pair") == event.get("pair")
        and vehicle.get("confirmed_at_utc") == observation.get("candle_close_utc")
        and vehicle.get("attempt_direction") == anchor.get("attempt_direction")
        and vehicle.get("branch_outcome") == route.get("branch_outcome")
        and vehicle.get("trade_side") == route.get("trade_side")
        and vehicle.get("route_family") == route.get("route_family")
        and vehicle.get("route_candidate_methods") == route.get("candidate_methods")
        and vehicle.get("late_detected") is (event.get("late_detected") is True)
        and vehicle["source_binding"].get("regime_contract_sha256")
        == observation.get("regime_contract_sha256")
    )


def _episode_outcome_valid(
    value: Mapping[str, Any], vehicle: Mapping[str, Any]
) -> bool:
    try:
        if not _sealed_valid(value, OUTCOME_CONTRACT):
            return False
        learning_outcome = value["learning_outcome"]
        observations = value["arm_observations"]
        truth_path_receipts = value["truth_path_candles"]
        truth_path_candles = _truth_candles_from_receipts(
            truth_path_receipts
        )
        if truth_path_candles is None:
            return False
        if not _outcome_valid_for_seat(learning_outcome, vehicle["learning_seat"]):
            return False
        generated = _parse_utc(vehicle["learning_seat"]["generated_at_utc"])
        maturity = _seat_maturity(vehicle["learning_seat"])
        _validate_truth_path(
            truth_path_candles,
            generated=generated,
            maturity=maturity,
        )
        expected_truth_path_sha = _truth_path_sha(
            pair=str(vehicle["pair"]),
            generated=generated,
            maturity=maturity,
            candles=truth_path_candles,
            chunk_hashes=[str(item) for item in value["truth_chunk_sha256"]],
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if not bool(
        value.get("schema_version") == 1
        and value.get("scoring_policy") == SCORING_POLICY
        and value.get("vehicle_id") == vehicle.get("vehicle_id")
        and value.get("vehicle_contract_sha256")
        == vehicle.get("contract_sha256")
        and value.get("episode_id") == vehicle.get("episode_id")
        and value.get("confirmed_event_sha256")
        == vehicle.get("confirmed_event_sha256")
        and value.get("pair") == vehicle.get("pair")
        and value.get("branch_outcome") == vehicle.get("branch_outcome")
        and value.get("attempt_direction") == vehicle.get("attempt_direction")
        and value.get("route_family") == vehicle.get("route_family")
        and value.get("late_detected") is (vehicle.get("late_detected") is True)
        and value.get("scorecard_eligible")
        is (vehicle.get("scorecard_eligible") is True)
        and _parse_utc(value["resolved_at_utc"])
        >= _parse_utc(vehicle["maturity_at_utc"])
        and value.get("maturity_at_utc") == vehicle.get("maturity_at_utc")
        and value.get("learning_outcome_contract_sha256")
        == learning_outcome.get("contract_sha256")
        and value.get("truth_request_from_utc")
        == learning_outcome.get("truth_request_from_utc")
        and value.get("truth_request_to_utc")
        == learning_outcome.get("truth_request_to_utc")
        and value.get("truth_path_sha256")
        == learning_outcome.get("truth_path_sha256")
        and value.get("truth_chunk_sha256")
        == learning_outcome.get("truth_chunk_sha256")
        and value.get("truth_path_candles_sha256")
        == _canonical_sha(truth_path_receipts)
        and value.get("truth_path_candles_persisted_for_membership_validation")
        is True
        and len(truth_path_candles)
        == int(learning_outcome.get("truth_candle_count") or -1)
        and expected_truth_path_sha == value.get("truth_path_sha256")
        and isinstance(observations, list)
        and value.get("candidate_count") == len(learning_outcome["candidates"])
        and value.get("arm_observation_count") == len(observations)
        and value.get("proof_and_observed_mark_must_not_be_mixed") is True
        and _authority_is_zero(value)
    ):
        return False
    expected_arms = {
        (
            str(candidate["candidate_sha256"]),
            str(arm["arm_id"]),
        ): (candidate, arm)
        for candidate in learning_outcome["candidates"]
        for arm in candidate["arms"]
    }
    roles = {
        str(role["candidate_sha256"]): str(role["episode_role"])
        for role in vehicle["candidate_roles"]
    }
    seen: set[tuple[str, str]] = set()
    truth_by_timestamp = {
        candle.timestamp_utc: candle for candle in truth_path_candles
    }
    for observation in observations:
        if not isinstance(observation, Mapping):
            return False
        body = {
            key: item
            for key, item in observation.items()
            if key != "observation_sha256"
        }
        key = (
            str(observation.get("candidate_sha256") or ""),
            str(observation.get("arm_id") or ""),
        )
        pair = expected_arms.get(key)
        if (
            pair is None
            or key in seen
            or observation.get("observation_sha256") != _canonical_sha(body)
        ):
            return False
        candidate, arm = pair
        if not bool(
            observation.get("candidate_id") == candidate.get("candidate_id")
            and observation.get("side") == candidate.get("side")
            and observation.get("method") == candidate.get("method")
            and observation.get("episode_role") == roles.get(key[0])
            and observation.get("arm_outcome_sha256")
            == arm.get("arm_outcome_sha256")
            and observation.get("filled") is (arm.get("filled") is True)
            and observation.get("proof_exit_reason") == arm.get("exit_reason")
            and observation.get("proof_exit_at_utc") == arm.get("exit_at_utc")
            and math.isclose(
                float(observation["proof_post_cost_realized_pips"]),
                float(arm["post_cost_realized_pips"]),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and observation.get("fill_s5_interval_utc")
            == _s5_interval(_parse_optional_utc(arm.get("fill_at_utc")))
            and observation.get("proof_and_observed_mark_must_not_be_mixed")
            is True
        ):
            return False
        reason = str(arm["exit_reason"])
        mark = observation.get("observed_horizon_mark_post_cost_pips")
        if reason == "HORIZON_FULL_STOP_LOSS":
            try:
                price = float(observation["observed_horizon_mark_price"])
                mark_interval = observation[
                    "observed_horizon_mark_s5_interval_utc"
                ]
                if not isinstance(mark_interval, Mapping):
                    return False
                mark_clock = _parse_utc(mark_interval["from_utc"])
                mark_candle = truth_by_timestamp[mark_clock]
                fill_clock = _parse_utc(arm["fill_at_utc"])
                exit_clock = _parse_utc(arm["exit_at_utc"])
                eligible_marks = [
                    candle
                    for candle in truth_path_candles
                    if fill_clock <= candle.timestamp_utc < exit_clock
                ]
                if not eligible_marks:
                    return False
                last_executable_mark = max(
                    eligible_marks,
                    key=lambda candle: candle.timestamp_utc,
                )
                executable_price = float(
                    mark_candle.bid_c
                    if candidate["side"] == "LONG"
                    else mark_candle.ask_c
                )
                expected_mark = _round(
                    (price - float(arm["entry"]))
                    * float(instrument_pip_factor(str(vehicle["pair"])))
                    if candidate["side"] == "LONG"
                    else (float(arm["entry"]) - price)
                    * float(instrument_pip_factor(str(vehicle["pair"])))
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                return False
            if not bool(
                observation.get("observed_position_state") == "OPEN_AT_HORIZON"
                and mark_candle.timestamp_utc
                == last_executable_mark.timestamp_utc
                and math.isclose(
                    price,
                    executable_price,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and observation.get("observed_horizon_mark_price_side")
                == ("BID" if candidate["side"] == "LONG" else "ASK")
                and mark is not None
                and math.isclose(float(mark), expected_mark, abs_tol=1e-6)
                and observation.get("observed_horizon_mark_s5_interval_utc")
                == _s5_interval(mark_candle.timestamp_utc)
                and observation.get("exit_s5_interval_utc")
                == _s5_interval(mark_candle.timestamp_utc)
                and observation.get("exit_s5_interval_semantics")
                == "LAST_EXECUTABLE_MARK_ONLY_NOT_PROOF_EXIT"
            ):
                return False
        else:
            expected_exit_interval = _s5_interval(
                _parse_optional_utc(arm.get("exit_at_utc"))
            )
            expected_state = (
                "NO_FILL" if arm.get("fill_at_utc") is None else "CLOSED_BEFORE_HORIZON"
            )
            if not bool(
                mark is None
                and observation.get("observed_horizon_mark_price") is None
                and observation.get("observed_horizon_mark_price_side") is None
                and observation.get("observed_horizon_mark_s5_interval_utc") is None
                and observation.get("observed_position_state") == expected_state
                and observation.get("exit_s5_interval_utc")
                == expected_exit_interval
                and observation.get("exit_s5_interval_semantics")
                == (
                    "EXECUTABLE_ATTACHED_EXIT_TOUCH"
                    if expected_exit_interval is not None
                    else None
                )
            ):
                return False
        seen.add(key)
    return len(seen) == len(expected_arms)


def _load_vehicles(
    path: Path, *, events_by_sha: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = _load_jsonl(
        path,
        max_bytes=MAX_VEHICLE_LEDGER_BYTES,
        max_rows=MAX_LEDGER_ROWS,
    )
    seen_ids: set[str] = set()
    seen_events: set[str] = set()
    for index, row in enumerate(rows, start=1):
        event_sha = str(row.get("confirmed_event_sha256") or "")
        event = events_by_sha.get(event_sha)
        vehicle_id = str(row.get("vehicle_id") or "")
        if (
            not _vehicle_valid(row)
            or event is None
            or not _vehicle_matches_event(row, event)
            or vehicle_id in seen_ids
            or event_sha in seen_events
        ):
            raise ValueError(f"invalid episode vehicle ledger row {index}")
        seen_ids.add(vehicle_id)
        seen_events.add(event_sha)
    return rows


def _load_outcomes(
    path: Path, *, vehicles: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = _load_jsonl(
        path,
        max_bytes=MAX_OUTCOME_LEDGER_BYTES,
        max_rows=MAX_LEDGER_ROWS,
    )
    vehicle_by_sha = {
        str(vehicle["contract_sha256"]): vehicle for vehicle in vehicles
    }
    for index, row in enumerate(rows, start=1):
        vehicle = vehicle_by_sha.get(str(row.get("vehicle_contract_sha256") or ""))
        if vehicle is None or not _episode_outcome_valid(row, vehicle):
            raise ValueError(f"invalid episode outcome ledger row {index}")
    return rows


def _outcome_identity(value: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(value.get("vehicle_contract_sha256") or ""),
        str(value.get("scoring_policy") or ""),
    )


def _episode_role(
    *, side: str, method: str, trade_side: str, route_methods: Sequence[str]
) -> str:
    same_side = side == trade_side
    route_method = method in set(route_methods)
    if same_side and route_method:
        return ROUTE_ALIGNED
    if not same_side and route_method:
        return INVERSE_DIRECTION
    if same_side:
        return SAME_SIDE_OTHER_METHOD
    return INVERSE_SIDE_OTHER_METHOD


def _authority_is_zero(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("diagnostic_only") is True
        and value.get("order_authority") == "NONE"
        and value.get("automatic_promotion_allowed") is False
        and value.get("promotion_allowed") is False
        and value.get("primary_effect") is False
        and value.get("risk_effect") is False
        and value.get("shadow_only") is True
        and value.get("live_permission") is False
        and value.get("broker_mutation_allowed") is False
    )


@contextmanager
def _whole_run_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("episode truth lock directory is invalid")
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("episode truth lock must be regular")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise TruthLockBusyError("episode truth worker already active") from error
        yield
    finally:
        os.close(descriptor)


def _load_jsonl(path: Path, *, max_bytes: int, max_rows: int) -> list[dict[str, Any]]:
    try:
        initial = path.lstat()
    except FileNotFoundError:
        return []
    if not stat.S_ISREG(initial.st_mode) or not 0 <= initial.st_size <= max_bytes:
        raise ValueError(f"invalid bounded JSONL file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != initial.st_size:
            raise ValueError(f"JSONL changed before read: {path}")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        len(raw) > max_bytes
        or len(raw) != before.st_size
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ValueError(f"JSONL changed during read: {path}")
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"torn JSONL tail: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise ValueError(f"blank JSONL row {line_number}: {path}")
        try:
            value = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid JSONL row {line_number}: {path}") from error
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row {line_number}: {path}")
        rows.append(value)
        if len(rows) > max_rows:
            raise ValueError(f"JSONL row cap exceeded: {path}")
    return rows


def _write_jsonl_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_bytes: int,
    max_rows: int,
) -> None:
    if len(rows) > max_rows:
        raise ValueError("episode truth JSONL row cap reached")
    raw = b"".join(_canonical_json_bytes(dict(row)) + b"\n" for row in rows)
    if len(raw) > max_bytes:
        raise ValueError("episode truth JSONL byte cap reached")
    _write_bytes_atomic(path, raw)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    raw = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_bytes_atomic(path, raw)


def _write_bytes_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir() or path.is_symlink():
        raise ValueError("episode truth destination is invalid")
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = os.open(
        temp,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _s5_interval(value: datetime | None) -> dict[str, str] | None:
    if value is None:
        return None
    start = _aware_utc(value)
    return {
        "from_utc": start.isoformat(),
        "to_utc": (start + timedelta(seconds=5)).isoformat(),
    }


def _mean(values: Sequence[float]) -> float | None:
    return _round(sum(values) / len(values)) if values else None


def _positive_rate(values: Sequence[float]) -> float | None:
    return _round(sum(value > 0.0 for value in values) / len(values)) if values else None


def _round(value: float) -> float:
    return round(float(value), ROUND_DIGITS)


def _parse_optional_utc(value: Any) -> datetime | None:
    return None if value is None else _parse_utc(value)


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("timestamp must be an aware ISO string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamp must be an aware ISO string") from error
    return _aware_utc(parsed)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {str(key): item for key, item in value.items() if key != "contract_sha256"}
    return {**body, "contract_sha256": _canonical_sha(body)}


def _sealed_valid(value: Mapping[str, Any], contract: str) -> bool:
    if not isinstance(value, Mapping) or value.get("contract") != contract:
        return False
    stored = value.get("contract_sha256")
    return bool(
        _sha_text(stored)
        and stored
        == _canonical_sha(
            {str(key): item for key, item in value.items() if key != "contract_sha256"}
        )
    )


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha_text(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


__all__ = [
    "EPISODE_ROLES",
    "INVERSE_DIRECTION",
    "INVERSE_SIDE_OTHER_METHOD",
    "OUTCOME_CONTRACT",
    "ROUTE_ALIGNED",
    "SAME_SIDE_OTHER_METHOD",
    "SCORECARD_CONTRACT",
    "SCORING_POLICY",
    "TRUTH_ADAPTER_CONTRACT",
    "VEHICLE_CONTRACT",
    "VEHICLE_POLICY",
    "build_fast_bot_episode_scorecard",
    "resolve_fast_bot_episode_vehicle",
    "run_fast_bot_episode_truth_cycle",
]
