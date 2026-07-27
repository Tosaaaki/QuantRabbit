"""January-development v2 for the immutable r13 AI inventory experiment.

This module intentionally reads the v1 prepared derivative instead of reading
or replaying the immutable r13 bot job again.  It adds:

* a point-in-time, closed-bar D1/H4/H1/M5 regime cache;
* family-level BASE/STRESS calibration (one policy per family and arm);
* a fresh Worker envelope that contains only the current causal packet;
* AI-cost-adjusted gates and January-development status vocabulary; and
* sealed walk-forward and mechanism-to-sibling-factory contracts.

It imports no broker, live gateway, automation, or deployment code.
"""

from __future__ import annotations

import bisect
import gzip
import json
import math
import os
import stat
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from quant_rabbit.dojo_portfolio_replay_reducer import canonical_portfolio_sha256
from quant_rabbit.dojo_r13_ai_inventory_oos import (
    A_BOT_ONLY,
    B_INVENTORY_ONLY,
    C_FORECAST_INVENTORY,
    DojoR13AIInventoryError,
    RESPONSE_CONTRACT,
    _AUTHORITY,
    _atomic_json,
    _default_narrative,
    _file_sha256,
    load_prepared_coordinate,
    load_prepared_study,
    score_forecasts_posthoc,
    simulate_partition,
    validate_worker_response,
)


V2_CONTRACT: Final = "QR_DOJO_R13_AI_INVENTORY_OOS_STUDY_V2"
REGIME_CACHE_CONTRACT: Final = "QR_DOJO_POINT_IN_TIME_MTF_REGIME_CACHE_V2"
REGIME_MATRIX_CONTRACT: Final = "QR_DOJO_STRATEGY_REGIME_MATRIX_V2"
CALIBRATION_CONTRACT: Final = "QR_DOJO_R13_AI_INVENTORY_CALIBRATION_V2"
WORKER_ENVELOPE_CONTRACT: Final = "QR_DOJO_R13_AI_WORKER_ENVELOPE_V2"
WORKER_SESSION_CONTRACT: Final = "QR_DOJO_R13_AI_WORKER_SESSION_V2"
OOS_RESULT_CONTRACT: Final = "QR_DOJO_R13_AI_INVENTORY_OOS_RESULT_V2"
WALK_FORWARD_CONTRACT: Final = "QR_DOJO_AI_SUPERVISOR_WALK_FORWARD_SEAL_V2"
FACTORY_CONTRACT: Final = "QR_DOJO_MECHANISM_SIBLING_FACTORY_V2"
SCHEMA_VERSION: Final = 2

FEATURE_VERSION: Final = "QR_MTF_CLOSED_BAR_FEATURES_V2"
PROMPT_VERSION: Final = "QR_INVENTORY_NARRATIVE_WORKER_V2"
AI_COST_JPY_PER_USD: Final = 160.0
AI_INPUT_USD_PER_M: Final = 5.0
AI_OUTPUT_USD_PER_M: Final = 15.0

TIMEFRAMES: Final = {
    "M5": 300,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
}

# A small, preregistered family-level screen.  The v2 runner first ranks these
# on calibration only, then seals one policy/cadence per family and arm.
POLICIES: Final = {
    "GUARD_ONLY_V2": {
        "loss_partial": 0.75,
        "loss_close": 0.95,
        "giveback_partial": 0.65,
        "drawdown_pause": 0.20,
        "mtf_conflict": False,
    },
    "INVENTORY_BALANCED_V2": {
        "loss_partial": 0.40,
        "loss_close": 0.70,
        "giveback_partial": 0.35,
        "drawdown_pause": 0.025,
        "mtf_conflict": False,
    },
    "INVENTORY_PROTECTIVE_V2": {
        "loss_partial": 0.25,
        "loss_close": 0.55,
        "giveback_partial": 0.25,
        "drawdown_pause": 0.015,
        "mtf_conflict": False,
    },
    "MTF_CONFLICT_GUARD_V2": {
        "loss_partial": 0.35,
        "loss_close": 0.65,
        "giveback_partial": 0.30,
        "drawdown_pause": 0.020,
        "mtf_conflict": True,
    },
}

CADENCES: Final = ("FIXED_15M", "FIXED_60M", "ADAPTIVE")
MIN_CALIBRATION_TRADES: Final = 12
MIN_REGIME_TRADES: Final = 8
MAX_PHASE_B_CALLS: Final = 4

STRATEGY_HORIZONS: Final = {
    "burst": {"primary_tf": "M5", "horizon_min": 30},
    "mean_revert_24h": {"primary_tf": "H1", "horizon_min": 120},
    "prev_day_extreme_fade": {"primary_tf": "H1", "horizon_min": 120},
    "pullback_limit": {"primary_tf": "H1", "horizon_min": 60},
    "round_number_fade": {"primary_tf": "M5", "horizon_min": 60},
    "spike_fade": {"primary_tf": "M5", "horizon_min": 30},
}

_ZERO_SHA: Final = "0" * 64
WORKER_RESPONSE_SCHEMA: Final = {
    "contract": RESPONSE_CONTRACT,
    "schema_version": 1,
    "top_level_exact_keys": [
        "contract",
        "schema_version",
        "packet_sha256",
        "observed_at",
        "narrative_state",
        "forecast",
        "inventory_diagnosis",
        "action",
        "rationale",
        "next_trigger",
        "authority",
    ],
    "narrative_state_exact_keys": [
        "version",
        "micro_concrete",
        "micro_abstract",
        "macro_concrete",
        "macro_abstract",
        "global_story",
        "strategy_story",
        "prior_hypothesis",
        "current_observation",
        "what_matched",
        "what_failed",
        "why",
        "next_hypothesis",
        "evidence_refs",
    ],
    "forecast": {
        "exact_keys": [
            "direction",
            "confidence",
            "horizon_min",
            "invalidation",
            "evidence_refs",
        ],
        "direction": ["UP", "DOWN", "RANGE", "UNCERTAIN"],
        "horizon_min": [30, 60, 120],
    },
    "inventory_diagnosis": {
        "exact_keys": [
            "risk_level",
            "strategy_regime_fit",
            "inventory_story_mismatch",
            "tp_profit_retention_risk",
            "loss_giveback_risk",
        ],
        "risk_level": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        "strategy_regime_fit": ["FIT", "MIXED", "MISMATCH", "UNKNOWN"],
    },
    "action": {
        "exact_keys": ["type", "fraction", "direction_restriction"],
        "type": [
            "HOLD",
            "PAUSE_NEW_ENTRIES",
            "RESUME",
            "REDUCE_LONG",
            "REDUCE_SHORT",
            "PARTIAL_CLOSE",
            "CLOSE_RISKY",
            "CLOSE_ALL",
        ],
        "direction_restriction": [
            "NONE",
            "LONG_ONLY",
            "SHORT_ONLY",
            "NO_NEW_LONGS",
            "NO_NEW_SHORTS",
        ],
        "fraction": "0.1..0.9 for REDUCE/PARTIAL_CLOSE; otherwise null",
    },
    "next_trigger": "string",
    "authority_exact": dict(_AUTHORITY),
}


def _strict_json(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise DojoR13AIInventoryError("value is not strict JSON") from exc


def _iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _read_gzip_json(path: Path, maximum_bytes: int = 512 * 1024 * 1024) -> Any:
    target = path.resolve(strict=True)
    before = target.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
        raise DojoR13AIInventoryError(f"invalid gzip input: {target}")
    descriptor = os.open(
        target,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        compressed = b""
        while chunk := os.read(descriptor, 1024 * 1024):
            compressed += chunk
            if len(compressed) > maximum_bytes:
                raise DojoR13AIInventoryError("gzip input exceeded bound")
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = target.stat(follow_symlinks=False)
    identities = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        for item in (before, opened, after)
    }
    if len(identities) != 1:
        raise DojoR13AIInventoryError("gzip input changed while reading")
    try:
        return json.loads(gzip.decompress(compressed))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoR13AIInventoryError("invalid gzip JSON") from exc


def _atomic_gzip_json(path: Path, value: Any) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = gzip.compress(
        json.dumps(
            _strict_json(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8"),
        mtime=0,
    )
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if target.exists():
        temporary.unlink()
        raise FileExistsError(f"immutable output already exists: {target}")
    os.replace(temporary, target)


def _series_from_frames(
    frames: Sequence[Mapping[str, Any]],
) -> tuple[list[int], dict[str, list[tuple[int, float, float]]]]:
    epochs: list[int] = []
    by_pair: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for frame in frames:
        if frame["phase"] != "C":
            continue
        epoch = int(frame["epoch"])
        epochs.append(epoch)
        for quote in frame["quotes"]:
            bid = float(quote["bid"])
            ask = float(quote["ask"])
            if not math.isfinite(bid) or not math.isfinite(ask) or ask < bid:
                raise DojoR13AIInventoryError("invalid observed quote")
            by_pair[str(quote["pair"])].append(
                (epoch, (bid + ask) / 2.0, ask - bid)
            )
    if not epochs or not by_pair:
        raise DojoR13AIInventoryError("no close-coordinate quotes")
    return sorted(set(epochs)), dict(by_pair)


def _closed_tf_series(
    source: Sequence[tuple[int, float, float]], seconds: int
) -> tuple[list[int], list[float], list[float]]:
    if seconds == TIMEFRAMES["M5"]:
        return (
            [row[0] for row in source],
            [row[1] for row in source],
            [row[2] for row in source],
        )
    buckets: dict[int, tuple[int, float, float]] = {}
    for epoch, mid, spread in source:
        bucket_end = (epoch // seconds + 1) * seconds
        buckets[bucket_end] = (epoch, mid, spread)
    ends = sorted(buckets)
    return (
        ends,
        [buckets[end][1] for end in ends],
        [buckets[end][2] for end in ends],
    )


def _percentile_rank(values: Sequence[float], current: float) -> float:
    if not values:
        return 0.5
    return sum(value <= current for value in values) / len(values)


def _tf_feature(
    *,
    decision_epoch: int,
    ends: Sequence[int],
    closes: Sequence[float],
    spreads: Sequence[float],
    timeframe: str,
) -> dict[str, Any]:
    # For H1/H4/D1 only buckets whose end is at or before the decision exist.
    # M5 close coordinates are already completed observations in the source.
    count = bisect.bisect_right(ends, decision_epoch)
    if count <= 1:
        return {
            "direction": "UNKNOWN",
            "strength": 0.0,
            "vol_percentile": None,
            "structure": "INSUFFICIENT",
            "confidence": 0.0,
            "age_seconds": None,
            "closed_bar_count": count,
        }
    values = list(closes[max(0, count - 160) : count])
    spread_values = list(spreads[max(0, count - 160) : count])
    recent_returns = [
        math.log(later / earlier)
        for earlier, later in zip(values, values[1:])
        if earlier > 0 and later > 0
    ]
    volatility = (
        math.sqrt(
            sum(value * value for value in recent_returns[-24:])
            / max(1, len(recent_returns[-24:]))
        )
        if recent_returns
        else 0.0
    )
    # A causal, bounded volatility proxy.  Using the empirical distribution of
    # already observed absolute returns avoids rebuilding a nested rolling
    # window for every timestamp while preserving the immutable cutoff.
    absolute_returns = [abs(value) for value in recent_returns[-101:]]
    current_abs_return = absolute_returns[-1] if absolute_returns else 0.0
    vol_percentile = _percentile_rank(
        absolute_returns[:-1], current_abs_return
    )
    fast_count = min(4, len(values))
    slow_count = min(13, len(values))
    fast = sum(values[-fast_count:]) / fast_count
    slow = sum(values[-slow_count:]) / slow_count
    trend_gap = fast / slow - 1.0 if slow > 0 else 0.0
    scale = max(volatility, 1e-8)
    normalized = trend_gap / scale
    if normalized >= 0.60:
        direction = "UP"
    elif normalized <= -0.60:
        direction = "DOWN"
    else:
        direction = "RANGE"
    strength = min(1.0, abs(normalized) / 2.5)
    structure_values = values[-min(25, len(values)) :]
    low = min(structure_values)
    high = max(structure_values)
    position = (
        (values[-1] - low) / (high - low) if high > low else 0.5
    )
    if position >= 0.90:
        structure = "UPPER_BOUNDARY"
    elif position <= 0.10:
        structure = "LOWER_BOUNDARY"
    elif 0.35 <= position <= 0.65:
        structure = "MID_RANGE"
    else:
        structure = "INNER_RANGE"
    spread_now = spread_values[-1] if spread_values else 0.0
    spread_percentile = _percentile_rank(spread_values[:-1], spread_now)
    confidence = min(
        0.95,
        max(
            0.10,
            0.30
            + 0.45 * strength
            + 0.15 * min(1.0, len(values) / 40.0)
            - 0.15 * max(0.0, spread_percentile - 0.8),
        ),
    )
    return {
        "direction": direction,
        "strength": strength,
        "vol_percentile": vol_percentile,
        "structure": structure,
        "confidence": confidence,
        "age_seconds": max(0, decision_epoch - int(ends[count - 1])),
        "closed_bar_count": count,
        "spread_percentile": spread_percentile,
        "range_position": position,
        "timeframe": timeframe,
    }


def _session(epoch: int) -> str:
    hour = datetime.fromtimestamp(epoch, timezone.utc).hour
    if 0 <= hour < 7:
        return "ASIA"
    if 7 <= hour < 13:
        return "LONDON"
    if 13 <= hour < 21:
        return "NEW_YORK"
    return "ROLLOVER"


def _combine_regime(
    *,
    epoch: int,
    pair: str,
    features: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    macro_rows = [features["D1"], features["H4"]]
    macro_directions = [
        row["direction"] for row in macro_rows if row["direction"] != "UNKNOWN"
    ]
    if macro_directions and len(set(macro_directions)) == 1:
        macro_direction = macro_directions[0]
    elif "UP" in macro_directions and "DOWN" in macro_directions:
        macro_direction = "CONFLICT"
    else:
        macro_direction = "RANGE"
    directions = [
        features[name]["direction"]
        for name in ("D1", "H4", "H1", "M5")
        if features[name]["direction"] != "UNKNOWN"
    ]
    directional = [value for value in directions if value in {"UP", "DOWN"}]
    agreement = (
        len(directional) >= 2 and len(set(directional)) == 1
    )
    conflict = "UP" in directional and "DOWN" in directional
    vol_values = [
        float(row["vol_percentile"])
        for row in features.values()
        if row["vol_percentile"] is not None
    ]
    vol_score = sum(vol_values) / len(vol_values) if vol_values else 0.5
    if vol_score >= 0.75:
        vol_regime = "HIGH"
    elif vol_score <= 0.25:
        vol_regime = "LOW"
    else:
        vol_regime = "MID"
    body = {
        "observed_at_epoch": epoch,
        "observed_at": _iso(epoch),
        "pair": pair,
        "feature_version": FEATURE_VERSION,
        "macro": {
            "direction": macro_direction,
            "d1": features["D1"],
            "h4": features["H4"],
        },
        "meso": {"direction": features["H1"]["direction"], "h1": features["H1"]},
        "micro": {"direction": features["M5"]["direction"], "m5": features["M5"]},
        "tf_agreement": agreement,
        "tf_conflict": conflict,
        "vol_regime": vol_regime,
        "session": _session(epoch),
    }
    return {**body, "regime_state_sha256": canonical_portfolio_sha256(body)}


def build_regime_cache(
    *, source_root: Path, output_root: Path
) -> dict[str, Any]:
    """Compute the causal MTF cache once from the v1 immutable derivative."""

    started = time.monotonic()
    study, frames = load_prepared_study(source_root)
    source_path = source_root.resolve(strict=True) / study["market_frames_file"]
    source_bytes, source_sha = _file_sha256(source_path)
    epochs, by_pair = _series_from_frames(frames)
    tf_series = {
        pair: {
            name: _closed_tf_series(rows, seconds)
            for name, seconds in TIMEFRAMES.items()
        }
        for pair, rows in by_pair.items()
    }
    rows: list[dict[str, Any]] = []
    for epoch in epochs:
        for pair in sorted(by_pair):
            features = {
                name: _tf_feature(
                    decision_epoch=epoch,
                    ends=tf_series[pair][name][0],
                    closes=tf_series[pair][name][1],
                    spreads=tf_series[pair][name][2],
                    timeframe=name,
                )
                for name in ("D1", "H4", "H1", "M5")
            }
            rows.append(
                _combine_regime(epoch=epoch, pair=pair, features=features)
            )
    cache_path = output_root / "regime-cache.json.gz"
    _atomic_gzip_json(cache_path, rows)
    cache_bytes, cache_file_sha = _file_sha256(cache_path)
    body = {
        "contract": REGIME_CACHE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "study_sha256": study["study_sha256"],
        "source_market_frames_file_sha256": source_sha,
        "source_market_frames_file_bytes": source_bytes,
        "cache_file": cache_path.name,
        "cache_file_sha256": cache_file_sha,
        "cache_file_bytes": cache_bytes,
        "decision_timestamp_count": len(epochs),
        "pair_count": len(by_pair),
        "row_count": len(rows),
        "closed_bar_only": True,
        "forming_higher_timeframe_bar_included": False,
        "future_quote_included": False,
    }
    result = {
        **body,
        "cache_manifest_sha256": canonical_portfolio_sha256(body),
        "build_elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output_root / "regime-cache-manifest.json", result)
    return result


def load_regime_cache(
    *, output_root: Path, study_sha256: str
) -> tuple[dict[str, Any], dict[tuple[int, str], dict[str, Any]]]:
    manifest_path = output_root.resolve(strict=True) / "regime-cache-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed = manifest.pop("cache_manifest_sha256", None)
    elapsed = manifest.pop("build_elapsed_seconds", None)
    if (
        manifest.get("contract") != REGIME_CACHE_CONTRACT
        or manifest.get("study_sha256") != study_sha256
        or claimed != canonical_portfolio_sha256(manifest)
    ):
        raise DojoR13AIInventoryError("regime cache manifest changed")
    manifest["cache_manifest_sha256"] = claimed
    manifest["build_elapsed_seconds"] = elapsed
    cache_path = output_root / manifest["cache_file"]
    size, digest = _file_sha256(cache_path)
    if (
        size != manifest["cache_file_bytes"]
        or digest != manifest["cache_file_sha256"]
    ):
        raise DojoR13AIInventoryError("regime cache changed")
    rows = _read_gzip_json(cache_path)
    index = {
        (int(row["observed_at_epoch"]), str(row["pair"])): row for row in rows
    }
    if len(index) != manifest["row_count"]:
        raise DojoR13AIInventoryError("regime cache denominator mismatch")
    return manifest, index


def _mean_confidence_interval(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "lower_95": None, "upper_95": None}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "lower_95": None, "upper_95": None}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half = 1.96 * math.sqrt(variance / len(values))
    return {"mean": mean, "lower_95": mean - half, "upper_95": mean + half}


def _regime_key(state: Mapping[str, Any]) -> str:
    return "|".join(
        (
            str(state["macro"]["direction"]),
            str(state["meso"]["direction"]),
            str(state["micro"]["direction"]),
            str(state["vol_regime"]),
            str(state["session"]),
            "CONFLICT" if state["tf_conflict"] else "NO_CONFLICT",
        )
    )


def build_strategy_regime_matrix(
    *,
    source_root: Path,
    output_root: Path,
    partition: str,
) -> dict[str, Any]:
    """Single-pass sufficient statistics from the immutable trade schedule."""

    if partition not in {"CALIBRATION", "OOS"}:
        raise DojoR13AIInventoryError("invalid regime matrix partition")
    started = time.monotonic()
    study, _ = load_prepared_study(source_root)
    cache_manifest, cache = load_regime_cache(
        output_root=output_root, study_sha256=study["study_sha256"]
    )
    window = (
        study["calibration_window"]
        if partition == "CALIBRATION"
        else study["oos_window"]
    )
    start_epoch = int(window["start_epoch"])
    end_epoch = int(window["end_epoch"])
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    overall: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    missing = 0
    for coordinate_ref in study["coordinates"]:
        coordinate = load_prepared_coordinate(
            source_root, study, coordinate_ref["coordinate_id"]
        )
        for trade in coordinate["trades"]:
            opened = int(trade["opened_epoch"])
            if not start_epoch <= opened < end_epoch:
                continue
            state = cache.get((opened, str(trade["pair"])))
            if state is None:
                missing += 1
                continue
            row = {
                "position_id": trade["position_id"],
                "net_pnl_jpy": float(trade["baseline_net_pnl_jpy"]),
                "price_pnl_jpy": float(trade["baseline_price_pnl_jpy"]),
                "financing_jpy": float(trade["baseline_financing_jpy"]),
                "units": float(trade["units"]),
                "side": trade["side"],
                "pair": trade["pair"],
                "opened_epoch": opened,
                "regime_key": _regime_key(state),
            }
            key = (
                str(coordinate["family_id"]),
                str(coordinate["cost_scenario"]),
                row["regime_key"],
            )
            groups[key].append(row)
            overall[(key[0], key[1])].append(row)
    matrix = []
    for (family, cost, regime_key), trades in sorted(groups.items()):
        pnls = [row["net_pnl_jpy"] for row in trades]
        gross_profit = sum(max(0.0, value) for value in pnls)
        gross_loss = -sum(min(0.0, value) for value in pnls)
        matrix.append(
            {
                "family_id": family,
                "cost_scenario": cost,
                "regime_key": regime_key,
                "trade_count": len(trades),
                "net_after_execution_costs_jpy": sum(pnls),
                "cost_jpy": sum(row["financing_jpy"] for row in trades),
                "profit_factor": (
                    gross_profit / gross_loss if gross_loss > 0 else None
                ),
                "expectancy": _mean_confidence_interval(pnls),
                "negative_pnl_contribution_jpy": sum(
                    min(0.0, value) for value in pnls
                ),
                "gross_units": sum(abs(row["units"]) for row in trades),
                "eligible_event_count": len(trades),
                "sample_gate": (
                    "ENOUGH_FOR_SCREEN"
                    if len(trades) >= MIN_REGIME_TRADES
                    else "DORMANT_OR_INSUFFICIENT"
                ),
            }
        )
    overall_rows = []
    for (family, cost), trades in sorted(overall.items()):
        pnls = [row["net_pnl_jpy"] for row in trades]
        overall_rows.append(
            {
                "family_id": family,
                "cost_scenario": cost,
                "trade_count": len(trades),
                "net_after_execution_costs_jpy": sum(pnls),
                "expectancy": _mean_confidence_interval(pnls),
            }
        )
    body = {
        "contract": REGIME_MATRIX_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "study_sha256": study["study_sha256"],
        "regime_cache_manifest_sha256": cache_manifest[
            "cache_manifest_sha256"
        ],
        "partition": partition,
        "single_pass_baseline_transcript_aggregation": True,
        "future_quote_included": False,
        "missing_regime_state_trade_count": missing,
        "overall": overall_rows,
        "matrix": matrix,
    }
    result = {
        **body,
        "matrix_sha256": canonical_portfolio_sha256(body),
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(
        output_root / f"strategy-regime-matrix-{partition.lower()}.json",
        result,
    )
    return result


def _regime_for_packet(
    packet: Mapping[str, Any],
    cache: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any] | None:
    positions = packet["inventory"]["positions"]
    if not positions:
        return None
    pair = str(positions[0]["pair"])
    row = cache.get((int(packet["cutoff_epoch"]), pair))
    return _strict_json(row) if row is not None else None


def deterministic_v2_worker_response(
    packet: Mapping[str, Any],
    *,
    policy_id: str,
    regime_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Phase-A policy; Phase-B actual Worker responses use the same schema."""

    if policy_id not in POLICIES:
        raise DojoR13AIInventoryError("unknown v2 policy")
    profile = POLICIES[policy_id]
    positions = list(packet["inventory"]["positions"])
    loss = max((float(row["loss_progress"]) for row in positions), default=0.0)
    giveback = max(
        (
            float(row["giveback_jpy"]) / float(row["mfe_jpy"])
            if float(row["mfe_jpy"]) > 0
            else 0.0
            for row in positions
        ),
        default=0.0,
    )
    drawdown = float(packet["inventory"]["drawdown_fraction"])
    action_type = "HOLD"
    fraction: float | None = None
    restriction = "NONE"
    risk = "LOW"
    fit = "UNKNOWN" if regime_state is None else "FIT"
    conflict = bool(regime_state and regime_state["tf_conflict"])
    meso_direction = (
        str(regime_state["meso"]["direction"])
        if regime_state is not None
        else "UNKNOWN"
    )
    if not positions and packet["prior_narrative_state"] is not None:
        action_type = "RESUME"
    elif loss >= float(profile["loss_close"]):
        action_type = "CLOSE_RISKY"
        risk = "CRITICAL"
    elif loss >= float(profile["loss_partial"]):
        action_type = "PARTIAL_CLOSE"
        fraction = 0.5
        risk = "HIGH"
    elif giveback >= float(profile["giveback_partial"]):
        action_type = "PARTIAL_CLOSE"
        fraction = 0.25
        risk = "HIGH"
    elif drawdown >= float(profile["drawdown_pause"]):
        action_type = "PAUSE_NEW_ENTRIES"
        risk = "HIGH"
    elif bool(profile["mtf_conflict"]) and conflict and positions:
        action_type = "PAUSE_NEW_ENTRIES"
        fit = "MISMATCH"
        risk = "MEDIUM"
        sides = {str(row["side"]) for row in positions}
        if meso_direction == "UP" and sides == {"SHORT"}:
            restriction = "NO_NEW_SHORTS"
        elif meso_direction == "DOWN" and sides == {"LONG"}:
            restriction = "NO_NEW_LONGS"
    version = (
        int(packet["prior_narrative_state"]["version"]) + 1
        if packet["prior_narrative_state"] is not None
        else 1
    )
    evidence = ["inventory", "observed_market.technical_by_position_pair"]
    narrative = _default_narrative(version, evidence)
    if regime_state is not None:
        narrative.update(
            {
                "micro_concrete": (
                    f"M5={regime_state['micro']['direction']}; "
                    f"vol={regime_state['vol_regime']}."
                ),
                "micro_abstract": (
                    "Micro timing is evaluated only from closed M5 observations."
                ),
                "macro_concrete": (
                    f"D1/H4 macro={regime_state['macro']['direction']}; "
                    f"H1 meso={regime_state['meso']['direction']}."
                ),
                "macro_abstract": (
                    "Higher-timeframe evidence is closed-bar and causal."
                ),
                "strategy_story": (
                    f"{packet['family_id']} primary horizon is "
                    f"{STRATEGY_HORIZONS[packet['family_id']]['primary_tf']}."
                ),
                "current_observation": (
                    f"loss={loss:.4f}; giveback={giveback:.4f}; "
                    f"drawdown={drawdown:.4f}; tf_conflict={conflict}."
                ),
                "what_failed": (
                    "Timeframes conflict." if conflict else "No strong conflict."
                ),
                "why": (
                    "Inventory risk and timeframe fit, not directional confidence, "
                    "control intervention size."
                ),
                "next_hypothesis": (
                    "Reassess on the next registered cadence or invalidation event."
                ),
            }
        )
    forecast = None
    if packet["arm"] == C_FORECAST_INVENTORY:
        directions = []
        if regime_state is not None:
            directions = [
                regime_state["macro"]["direction"],
                regime_state["meso"]["direction"],
                regime_state["micro"]["direction"],
            ]
        directional = [item for item in directions if item in {"UP", "DOWN"}]
        if len(directional) >= 2 and len(set(directional)) == 1:
            direction = directional[0]
            confidence = 0.60
        elif conflict:
            direction = "UNCERTAIN"
            confidence = 0.30
        else:
            direction = "RANGE"
            confidence = 0.45
        forecast = {
            "direction": direction,
            "confidence": confidence,
            "horizon_min": STRATEGY_HORIZONS[
                str(packet["family_id"])
            ]["horizon_min"],
            "invalidation": (
                "Invalidate when the closed H1 direction changes or the "
                "inventory risk bucket changes."
            ),
            "evidence_refs": [
                "observed_market.technical_by_position_pair",
                "inventory",
            ],
        }
    response = {
        "contract": RESPONSE_CONTRACT,
        "schema_version": 1,
        "packet_sha256": packet["packet_sha256"],
        "observed_at": packet["observed_at"],
        "narrative_state": narrative,
        "forecast": forecast,
        "inventory_diagnosis": {
            "risk_level": risk,
            "strategy_regime_fit": fit,
            "inventory_story_mismatch": (
                "Strong timeframe conflict." if conflict else "No strong mismatch."
            ),
            "tp_profit_retention_risk": f"giveback_fraction={giveback:.6f}",
            "loss_giveback_risk": (
                f"loss_progress={loss:.6f};drawdown={drawdown:.6f}"
            ),
        },
        "action": {
            "type": action_type,
            "fraction": fraction,
            "direction_restriction": restriction,
        },
        "rationale": (
            f"Preregistered v2 inventory-first policy {policy_id}; "
            "forecast confidence never increases risk."
        ),
        "next_trigger": (
            "Next registered cadence, risk-bucket change, profit giveback, "
            "or timeframe-conflict transition."
        ),
        "authority": dict(_AUTHORITY),
    }
    validate_worker_response(packet=packet, response=response)
    return response


def _metric_projection(
    cell: Mapping[str, Any], *, include_ai_cost: bool = True
) -> dict[str, Any]:
    metrics = dict(cell["metrics"])
    cost_usd = (
        float(metrics["ai_notional_cost_usd"]) if include_ai_cost else 0.0
    )
    cost_jpy = cost_usd * AI_COST_JPY_PER_USD
    net = float(metrics["net_after_all_costs_jpy"])
    return {
        "net_after_execution_costs_jpy": net,
        "ai_cost_jpy": cost_jpy,
        "net_after_all_costs_including_ai_jpy": net - cost_jpy,
        "ending_equity_after_ai_jpy": float(metrics["ending_equity_jpy"]) - cost_jpy,
        "monthly_equity_multiple_after_ai": (
            (float(metrics["ending_equity_jpy"]) - cost_jpy)
            / float(cell["initial_capital_jpy"])
        ),
        "profit_factor_after_execution_costs": metrics["profit_factor"],
        "win_rate": metrics["win_rate"],
        "expectancy_jpy": metrics["expectancy_jpy"],
        "max_drawdown_fraction": metrics["max_drawdown_fraction"],
        "max_margin_utilization_fraction": metrics[
            "max_margin_utilization_fraction"
        ],
        "margin_call_count": metrics["margin_call_count"],
        "ruin_event_count": metrics["ruin_event_count"],
        "tp_profit_retained_fraction": metrics[
            "tp_profit_retained_fraction"
        ],
        "loss_avoided_jpy": metrics["loss_avoided_jpy"],
        "missed_upside_jpy": metrics["missed_upside_jpy"],
        "turnover_jpy": metrics["turnover_jpy"],
        "scheduled_trade_count": metrics["scheduled_trade_count"],
        "trade_count": metrics["trade_count"],
        "skipped_trade_count": metrics["skipped_trade_count"],
        "ai_decision_count": metrics["ai_decision_count"],
        "ai_call_count": metrics["ai_call_count"],
        "ai_fallback_count": metrics["ai_fallback_count"],
        "ai_estimated_input_tokens": metrics["ai_estimated_input_tokens"],
        "ai_estimated_output_tokens": metrics["ai_estimated_output_tokens"],
        "ai_notional_cost_usd": cost_usd,
    }


def _risk_gate(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> bool:
    return (
        candidate["max_drawdown_fraction"]
        <= baseline["max_drawdown_fraction"] + 1e-12
        and candidate["margin_call_count"] <= baseline["margin_call_count"]
        and candidate["ruin_event_count"] <= baseline["ruin_event_count"]
    )


def calibrate_v2(
    *, source_root: Path, output_root: Path
) -> dict[str, Any]:
    """Successive-halving screen and family-level BASE/STRESS selection."""

    started = time.monotonic()
    study, frames = load_prepared_study(source_root)
    cache_manifest, cache = load_regime_cache(
        output_root=output_root, study_sha256=study["study_sha256"]
    )
    matrix_path = output_root / "strategy-regime-matrix-calibration.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix_sha = matrix.pop("matrix_sha256", None)
    matrix_elapsed = matrix.pop("elapsed_seconds", None)
    if matrix_sha != canonical_portfolio_sha256(matrix):
        raise DojoR13AIInventoryError("calibration regime matrix changed")
    matrix["matrix_sha256"] = matrix_sha
    matrix["elapsed_seconds"] = matrix_elapsed
    trade_counts = {
        (row["family_id"], row["cost_scenario"]): row["trade_count"]
        for row in matrix["overall"]
    }
    coordinates = {}
    baselines = {}
    for ref in study["coordinates"]:
        coordinate = load_prepared_coordinate(
            source_root, study, ref["coordinate_id"]
        )
        coordinates[(coordinate["family_id"], coordinate["cost_scenario"])] = (
            coordinate
        )
        cell = simulate_partition(
            study=study,
            coordinate=coordinate,
            frames=frames,
            partition="CALIBRATION",
            arm=A_BOT_ONLY,
            cadence_id=None,
            policy_version="IMMUTABLE_R13",
            prompt_version="NONE",
            worker=None,
            capture_full_audit=False,
        )
        baselines[(coordinate["family_id"], coordinate["cost_scenario"])] = (
            _metric_projection(cell, include_ai_cost=False)
        )

    # Phase A0: sample-aware deterministic pre-screen.  DORMANT is not REJECT.
    family_screen = {}
    for family in sorted(STRATEGY_HORIZONS):
        minimum = min(
            trade_counts.get((family, cost), 0) for cost in ("BASE", "STRESS")
        )
        family_screen[family] = (
            "DORMANT_OR_INSUFFICIENT"
            if minimum < MIN_CALIBRATION_TRADES
            else "ELIGIBLE_FOR_SUCCESSIVE_HALVING"
        )

    candidate_rows = []
    selected_rows = []
    cartesian_count = (
        len(STRATEGY_HORIZONS) * 2 * len(POLICIES) * len(CADENCES) * 2
    )
    for family in sorted(STRATEGY_HORIZONS):
        for arm in (B_INVENTORY_ONLY, C_FORECAST_INVENTORY):
            if family_screen[family] == "DORMANT_OR_INSUFFICIENT":
                selected_rows.append(
                    {
                        "family_id": family,
                        "arm": arm,
                        "policy_version": "GUARD_ONLY_V2",
                        "cadence_id": "ADAPTIVE",
                        "selection_status": "DORMANT_OR_INSUFFICIENT",
                        "selection_rule": "NO_PARAMETER_RANK_WITHOUT_SAMPLE",
                    }
                )
                continue
            # Round 1 is the two risk-first policies at ADAPTIVE cadence.
            round_one = ("GUARD_ONLY_V2", "INVENTORY_PROTECTIVE_V2")
            scores = []
            for policy_id in round_one:
                per_cost = []
                for cost in ("BASE", "STRESS"):
                    coordinate = coordinates[(family, cost)]
                    cell = simulate_partition(
                        study=study,
                        coordinate=coordinate,
                        frames=frames,
                        partition="CALIBRATION",
                        arm=arm,
                        cadence_id="ADAPTIVE",
                        policy_version=policy_id,
                        prompt_version="PHASE_A_DETERMINISTIC_V2",
                        worker=lambda packet, policy=policy_id: (
                            deterministic_v2_worker_response(
                                packet,
                                policy_id=policy,
                                regime_state=_regime_for_packet(packet, cache),
                            )
                        ),
                        capture_full_audit=False,
                    )
                    projected = _metric_projection(
                        cell, include_ai_cost=False
                    )
                    base = baselines[(family, cost)]
                    per_cost.append((cost, projected, base, cell["cell_sha256"]))
                score = sum(
                    row[1]["net_after_all_costs_including_ai_jpy"]
                    for row in per_cost
                )
                risk_ok = all(_risk_gate(row[1], row[2]) for row in per_cost)
                scores.append((policy_id, score, risk_ok, per_cost))
            winner = max(scores, key=lambda row: (row[2], row[1], row[0]))
            # Round 2: winner and one adjacent challenger across three cadences.
            challenger = (
                "MTF_CONFLICT_GUARD_V2"
                if winner[0] == "GUARD_ONLY_V2"
                else "INVENTORY_BALANCED_V2"
            )
            finalists = (winner[0], challenger)
            final_scores = []
            for policy_id in finalists:
                for cadence_id in CADENCES:
                    per_cost = []
                    for cost in ("BASE", "STRESS"):
                        coordinate = coordinates[(family, cost)]
                        cell = simulate_partition(
                            study=study,
                            coordinate=coordinate,
                            frames=frames,
                            partition="CALIBRATION",
                            arm=arm,
                            cadence_id=cadence_id,
                            policy_version=policy_id,
                            prompt_version="PHASE_A_DETERMINISTIC_V2",
                            worker=lambda packet, policy=policy_id: (
                                deterministic_v2_worker_response(
                                    packet,
                                    policy_id=policy,
                                    regime_state=_regime_for_packet(packet, cache),
                                )
                            ),
                            capture_full_audit=False,
                        )
                        projected = _metric_projection(
                            cell, include_ai_cost=False
                        )
                        base = baselines[(family, cost)]
                        per_cost.append(
                            (cost, projected, base, cell["cell_sha256"])
                        )
                        candidate_rows.append(
                            {
                                "family_id": family,
                                "cost_scenario": cost,
                                "arm": arm,
                                "policy_version": policy_id,
                                "cadence_id": cadence_id,
                                "metrics": projected,
                                "risk_gate": _risk_gate(projected, base),
                                "cell_sha256": cell["cell_sha256"],
                            }
                        )
                    net_score = sum(
                        row[1]["net_after_all_costs_including_ai_jpy"]
                        for row in per_cost
                    )
                    risk_ok = all(
                        _risk_gate(row[1], row[2]) for row in per_cost
                    )
                    final_scores.append(
                        (policy_id, cadence_id, net_score, risk_ok, per_cost)
                    )
            selected = max(
                final_scores,
                key=lambda row: (
                    row[3],
                    row[2],
                    -sum(item[1]["ai_call_count"] for item in row[4]),
                    row[0],
                    row[1],
                ),
            )
            selected_rows.append(
                {
                    "family_id": family,
                    "arm": arm,
                    "policy_version": selected[0],
                    "cadence_id": selected[1],
                    "selection_status": (
                        "CALIBRATION_RISK_GATE_PASS"
                        if selected[3]
                        else "CALIBRATION_DIAGNOSTIC_ONLY"
                    ),
                    "base_stress_combined_net_after_ai_jpy": selected[2],
                    "selection_rule": (
                        "SUCCESSIVE_HALVING_FAMILY_LEVEL_BASE_STRESS_"
                        "MAX_NET_THEN_MIN_CALLS_WITH_RISK_GATE"
                    ),
                }
            )
    evaluated_cells = len(candidate_rows)
    body = {
        "contract": CALIBRATION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "study_sha256": study["study_sha256"],
        "regime_cache_manifest_sha256": cache_manifest[
            "cache_manifest_sha256"
        ],
        "regime_matrix_sha256": matrix_sha,
        "partition": "CALIBRATION",
        "january_role": (
            "MECHANISM_DISCOVERY_INTEGRATION_CALIBRATION_MONTH_"
            "NOT_FINAL_MODEL_VALIDATION"
        ),
        "future_oos_accessed_during_selection": False,
        "full_cartesian_bruteforce_used": False,
        "screen_method": "CACHED_REGIME_MATRIX_PLUS_SUCCESSIVE_HALVING",
        "family_screen": family_screen,
        "policies": POLICIES,
        "cadences": list(CADENCES),
        "cartesian_cell_equivalent": cartesian_count,
        "evaluated_cell_count": evaluated_cells,
        "deterministic_screen_reduction_fraction": (
            1.0 - evaluated_cells / cartesian_count
            if cartesian_count
            else 0.0
        ),
        "selections": selected_rows,
        "candidate_cells": candidate_rows,
    }
    result = {
        **body,
        "calibration_sha256": canonical_portfolio_sha256(body),
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output_root / "calibration-v2.json", result)
    return result


def load_calibration(output_root: Path, study_sha256: str) -> dict[str, Any]:
    path = output_root.resolve(strict=True) / "calibration-v2.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    claimed = row.pop("calibration_sha256", None)
    elapsed = row.pop("elapsed_seconds", None)
    if (
        row.get("contract") != CALIBRATION_CONTRACT
        or row.get("study_sha256") != study_sha256
        or claimed != canonical_portfolio_sha256(row)
        or row.get("future_oos_accessed_during_selection") is not False
    ):
        raise DojoR13AIInventoryError("v2 calibration changed")
    row["calibration_sha256"] = claimed
    row["elapsed_seconds"] = elapsed
    return row


def worker_session_v2(
    *,
    source_root: Path,
    output_root: Path,
    coordinate_id: str,
    arm: str,
    session_output: Path,
    response_provider: Callable[[dict[str, Any]], Mapping[str, Any]],
    worker_id: str,
    worker_model: str,
    max_ai_calls: int = MAX_PHASE_B_CALLS,
) -> dict[str, Any]:
    """Run one sealed OOS cell through a fresh packet-only Worker boundary."""

    if arm not in {B_INVENTORY_ONLY, C_FORECAST_INVENTORY}:
        raise DojoR13AIInventoryError("v2 Worker arm must be B or C")
    study, frames = load_prepared_study(source_root)
    coordinate = load_prepared_coordinate(source_root, study, coordinate_id)
    calibration = load_calibration(output_root, study["study_sha256"])
    _, cache = load_regime_cache(
        output_root=output_root, study_sha256=study["study_sha256"]
    )
    selection = next(
        row
        for row in calibration["selections"]
        if row["family_id"] == coordinate["family_id"] and row["arm"] == arm
    )
    envelope_hashes = []

    def request(packet: dict[str, Any]) -> Mapping[str, Any]:
        regime_state = _regime_for_packet(packet, cache)
        body = {
            "contract": WORKER_ENVELOPE_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "study_sha256": study["study_sha256"],
            "calibration_sha256": calibration["calibration_sha256"],
            "packet": packet,
            "regime_state": regime_state,
            "strategy_horizon": STRATEGY_HORIZONS[
                coordinate["family_id"]
            ],
            "response_schema": WORKER_RESPONSE_SCHEMA,
            "policy_version": selection["policy_version"],
            "prompt_version": PROMPT_VERSION,
            "worker_context": (
                "FRESH_PACKET_ONLY_OR_VERSIONED_NARRATIVE_STATE_ONLY"
            ),
            "future_quote_included": False,
            "terminal_result_included": False,
            "other_arm_result_included": False,
            "append_wall_clock_included": False,
            "profit_conditioned_retry_allowed": False,
            "authority": dict(_AUTHORITY),
        }
        envelope = {
            **body,
            "envelope_sha256": canonical_portfolio_sha256(body),
        }
        envelope_hashes.append(envelope["envelope_sha256"])
        return response_provider(envelope)

    cell = simulate_partition(
        study=study,
        coordinate=coordinate,
        frames=frames,
        partition="OOS",
        arm=arm,
        cadence_id=selection["cadence_id"],
        policy_version=selection["policy_version"],
        prompt_version=PROMPT_VERSION,
        worker=request,
        max_ai_calls=max_ai_calls,
        capture_full_audit=True,
    )
    body = {
        "contract": WORKER_SESSION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "study_sha256": study["study_sha256"],
        "calibration_sha256": calibration["calibration_sha256"],
        "coordinate_id": coordinate_id,
        "family_id": coordinate["family_id"],
        "cost_scenario": coordinate["cost_scenario"],
        "arm": arm,
        "policy_version": selection["policy_version"],
        "cadence_id": selection["cadence_id"],
        "worker_id": worker_id,
        "worker_model": worker_model,
        "max_actual_ai_calls_preregistered": max_ai_calls,
        "profit_conditioned_retry_allowed": False,
        "envelope_hashes": envelope_hashes,
        "cell": cell,
    }
    result = {**body, "session_sha256": canonical_portfolio_sha256(body)}
    _atomic_json(session_output, result)
    return result


def deterministic_oos_session_v2(
    *,
    source_root: Path,
    output_root: Path,
    coordinate_id: str,
    arm: str,
    session_output: Path,
    max_ai_calls: int = MAX_PHASE_B_CALLS,
) -> dict[str, Any]:
    """Deterministic integration path; never labelled as an actual model call."""

    def provider(envelope: dict[str, Any]) -> Mapping[str, Any]:
        return deterministic_v2_worker_response(
            envelope["packet"],
            policy_id=envelope["policy_version"],
            regime_state=envelope["regime_state"],
        )

    return worker_session_v2(
        source_root=source_root,
        output_root=output_root,
        coordinate_id=coordinate_id,
        arm=arm,
        session_output=session_output,
        response_provider=provider,
        worker_id="deterministic-integration-worker",
        worker_model="RULE_BOUNDARY_NOT_ACTUAL_MODEL",
        max_ai_calls=max_ai_calls,
    )


def _load_session(path: Path) -> dict[str, Any]:
    row = json.loads(path.read_text(encoding="utf-8"))
    claimed = row.pop("session_sha256", None)
    if (
        row.get("contract") != WORKER_SESSION_CONTRACT
        or claimed != canonical_portfolio_sha256(row)
    ):
        raise DojoR13AIInventoryError(f"v2 session changed: {path}")
    row["session_sha256"] = claimed
    cell = row["cell"]
    cell_claimed = cell["cell_sha256"]
    if cell_claimed != canonical_portfolio_sha256(
        {key: value for key, value in cell.items() if key != "cell_sha256"}
    ):
        raise DojoR13AIInventoryError(f"v2 cell changed: {path}")
    return row


def _family_status(
    *,
    rows: Sequence[Mapping[str, Any]],
    sample_status: str,
) -> str:
    if sample_status == "DORMANT_OR_INSUFFICIENT":
        return "DORMANT_OR_INSUFFICIENT"
    if all(bool(row["positive_hard_gate"]) for row in rows):
        # January is a short development gate and cannot establish ACTIVE.
        return "CONDITIONAL_JANUARY_SHORT_GATE_PASS"
    return "JANUARY_OBSERVED_FAILURE_NOT_REJECT"


def aggregate_oos_v2(
    *,
    source_root: Path,
    output_root: Path,
    sessions_root: Path,
) -> dict[str, Any]:
    """Aggregate the one-time January OOS after calibration is sealed."""

    started = time.monotonic()
    study, frames = load_prepared_study(source_root)
    calibration = load_calibration(output_root, study["study_sha256"])
    rows = []
    comparisons = []
    actual_candidate_cells = []
    baseline_cells = {}
    for ref in study["coordinates"]:
        coordinate = load_prepared_coordinate(
            source_root, study, ref["coordinate_id"]
        )
        baseline = simulate_partition(
            study=study,
            coordinate=coordinate,
            frames=frames,
            partition="OOS",
            arm=A_BOT_ONLY,
            cadence_id=None,
            policy_version="IMMUTABLE_R13",
            prompt_version="NONE",
            worker=None,
            capture_full_audit=False,
        )
        baseline_cells[coordinate["coordinate_id"]] = baseline
        rows.append(
            {
                "coordinate_id": coordinate["coordinate_id"],
                "family_id": coordinate["family_id"],
                "cost_scenario": coordinate["cost_scenario"],
                "arm": A_BOT_ONLY,
                "worker_model": None,
                "metrics": _metric_projection(baseline),
                "cell_sha256": baseline["cell_sha256"],
            }
        )
        arm_cells = {}
        arm_metrics = {}
        for arm in (B_INVENTORY_ONLY, C_FORECAST_INVENTORY):
            path = sessions_root / arm / f"{coordinate['coordinate_id']}.json"
            session = _load_session(path)
            cell = session["cell"]
            arm_cells[arm] = cell
            is_actual_model = (
                session["worker_model"] != "RULE_BOUNDARY_NOT_ACTUAL_MODEL"
            )
            if is_actual_model:
                actual_candidate_cells.append(cell)
            projected = _metric_projection(
                cell, include_ai_cost=is_actual_model
            )
            arm_metrics[arm] = projected
            rows.append(
                {
                    "coordinate_id": coordinate["coordinate_id"],
                    "family_id": coordinate["family_id"],
                    "cost_scenario": coordinate["cost_scenario"],
                    "arm": arm,
                    "worker_model": session["worker_model"],
                    "policy_version": session["policy_version"],
                    "cadence_id": session["cadence_id"],
                    "metrics": projected,
                    "forecast_evaluation_actual_worker": cell[
                        "forecast_evaluation_actual_worker"
                    ],
                    "session_sha256": session["session_sha256"],
                    "cell_sha256": cell["cell_sha256"],
                }
            )
        a = _metric_projection(baseline)
        b = arm_metrics[B_INVENTORY_ONLY]
        c = arm_metrics[C_FORECAST_INVENTORY]
        coordinate_rows = []
        for name, candidate in (("B", b), ("C", c)):
            positive_gate = (
                candidate["net_after_all_costs_including_ai_jpy"] > 0
                and candidate["profit_factor_after_execution_costs"] is not None
                and candidate["profit_factor_after_execution_costs"] > 1
                and _risk_gate(candidate, a)
            )
            coordinate_rows.append(
                {
                    "arm": name,
                    "net_delta_vs_a_after_ai_jpy": (
                        candidate["net_after_all_costs_including_ai_jpy"]
                        - a["net_after_all_costs_including_ai_jpy"]
                    ),
                    "risk_gate": _risk_gate(candidate, a),
                    "positive_hard_gate": positive_gate,
                }
            )
        comparisons.append(
            {
                "coordinate_id": coordinate["coordinate_id"],
                "family_id": coordinate["family_id"],
                "cost_scenario": coordinate["cost_scenario"],
                "a_metrics": a,
                "b_metrics": b,
                "c_metrics": c,
                "arm_gates": coordinate_rows,
            }
        )
    family_decisions = []
    for family in sorted(STRATEGY_HORIZONS):
        family_rows = [
            row for row in comparisons if row["family_id"] == family
        ]
        c_rows = [
            {
                **next(item for item in row["arm_gates"] if item["arm"] == "C"),
                "cost_scenario": row["cost_scenario"],
            }
            for row in family_rows
        ]
        sample_status = calibration["family_screen"][family]
        family_decisions.append(
            {
                "family_id": family,
                "status": _family_status(
                    rows=c_rows, sample_status=sample_status
                ),
                "base_stress_both_positive_hard_gate": (
                    {row["cost_scenario"] for row in c_rows}
                    == {"BASE", "STRESS"}
                    and all(row["positive_hard_gate"] for row in c_rows)
                ),
                "mechanism_improvement_confirmed": (
                    {row["cost_scenario"] for row in c_rows}
                    == {"BASE", "STRESS"}
                    and all(
                        row["positive_hard_gate"]
                        and row["net_delta_vs_a_after_ai_jpy"] > 0
                        for row in c_rows
                    )
                ),
                "january_loss_is_not_sufficient_reject_evidence": True,
                "registry_retention": "CHAMPION_CHALLENGER_RETAIN",
            }
        )
    actual_model_cells = [
        row
        for row in rows
        if row["arm"] != A_BOT_ONLY
        and row["worker_model"] != "RULE_BOUNDARY_NOT_ACTUAL_MODEL"
    ]
    c_forecasts = [
        forecast
        for cell in actual_candidate_cells
        if cell["arm"] == C_FORECAST_INVENTORY
        for forecast in cell["forecast_rows"]
        if not forecast["fallback"]
    ]
    total_actual_calls = sum(
        row["metrics"]["ai_call_count"] for row in actual_model_cells
    )
    full_schedule_calls = (
        len(study["coordinates"])
        * 2
        * MAX_PHASE_B_CALLS
    )
    body = {
        "contract": OOS_RESULT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "study_sha256": study["study_sha256"],
        "calibration_sha256": calibration["calibration_sha256"],
        "partition": "OOS",
        "january_role": (
            "DEVELOPMENT_MONTH_SHORT_GATE_NOT_FINAL_MODEL_VALIDATION"
        ),
        "policy_adjustment_after_oos_allowed": False,
        "source_quote_coverage_proved": False,
        "classification": (
            "EXPERIMENTAL_SAME_INCOMPLETE_SOURCE_PAIRED_DIFFERENCE"
        ),
        "ai_cost_jpy_per_usd": AI_COST_JPY_PER_USD,
        "stretch_target_monthly_equity_multiple": 3.0,
        "lexicographic_objective": [
            "DATA_INTEGRITY_NO_LEAKAGE_NO_LIVE_AUTHORITY",
            "MARGIN_CALL_RUIN_HARD_GATE",
            "OOS_NET_POSITIVE_PF_GT_1_DD_NONWORSE",
            "MAXIMIZE_GEOMETRIC_RETURN_WITHIN_GATES",
            "MEASURE_3X_REPRODUCIBILITY",
        ],
        "cell_count": len(rows),
        "actual_model_cell_count": len(actual_model_cells),
        "actual_model_call_count": total_actual_calls,
        "full_schedule_call_equivalent": full_schedule_calls,
        "actual_ai_call_reduction_fraction": (
            1.0 - total_actual_calls / full_schedule_calls
            if full_schedule_calls
            else 0.0
        ),
        "forecast_summary_actual_worker": score_forecasts_posthoc(
            forecast_rows=c_forecasts,
            frames=frames,
        ),
        "cells": rows,
        "comparisons": comparisons,
        "family_decisions": family_decisions,
        "oracle_d_used_for_acting_or_claims": False,
        "next_month_bot_only_claimed": False,
    }
    result = {
        **body,
        "result_sha256": canonical_portfolio_sha256(body),
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output_root / "oos-result-v2.json", result)
    return result


def seal_walk_forward_contract(
    *, output_root: Path, oos_result: Mapping[str, Any]
) -> dict[str, Any]:
    """Seal the post-January one-way evaluation contract."""

    body = {
        "contract": WALK_FORWARD_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "january_result_sha256": oos_result["result_sha256"],
        "january_role": (
            "MECHANISM_DISCOVERY_INTEGRATION_CALIBRATION_AND_SHORT_GATE"
        ),
        "january_is_final_model_validation": False,
        "return_to_january_for_same_version_adjustment_allowed": False,
        "minimum_non_overlapping_oos_blocks": 8,
        "preferred_month_count": 12,
        "direction": "ONE_WAY_FORWARD_ONLY",
        "validation_method": "PURGED_EMBARGOED_WALK_FORWARD_NO_RANDOM_CV",
        "regime_balance": [
            "TREND_UP",
            "TREND_DOWN",
            "RANGE",
            "HIGH_VOL",
            "LOW_VOL",
            "SPREAD_EXPANSION",
            "MONTH_START_END",
            "MAJOR_SESSIONS",
            "EVENT_PROXIMITY",
        ],
        "monthly_metrics": [
            "MEDIAN",
            "WORST",
            "POSITIVE_MONTH_HIT_RATE",
            "PROFIT_FACTOR",
            "MAX_DRAWDOWN",
            "MARGIN_UTILIZATION",
            "RUIN_MARGIN_CALL",
            "EQUITY_MULTIPLE",
            "3X_HIT_RATE",
            "LOSS_STREAK",
            "RECOVERY_PERIOD",
        ],
        "policy_change_creates_new_version": True,
        "new_version_uses_only_later_untouched_periods": True,
        "source_quote_coverage_proved": False,
        "multi_month_classification_until_coverage_repair": "EXPERIMENTAL",
        "coverage_repair_requires_final_revalidation": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "order_authority": "NONE",
    }
    result = {**body, "walk_forward_sha256": canonical_portfolio_sha256(body)}
    _atomic_json(output_root / "walk-forward-seal-v2.json", result)
    return result


def build_factory_contract(
    *, output_root: Path, oos_result: Mapping[str, Any]
) -> dict[str, Any]:
    """Design the post-v2 mechanism factory without starting strategy search."""

    passed = [
        row["family_id"]
        for row in oos_result["family_decisions"]
        if row["mechanism_improvement_confirmed"]
    ]
    status = (
        "READY_FOR_CALIBRATION_EVIDENCE_ONLY_DESIGN"
        if passed
        else "NOT_STARTED_NO_JANUARY_BASE_STRESS_CHAMPION"
    )
    body = {
        "contract": FACTORY_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "january_result_sha256": oos_result["result_sha256"],
        "status": status,
        "eligible_parent_families": passed,
        "worker_input_boundary": (
            "TRAINING_AND_CALIBRATION_EVIDENCE_ONLY_HELD_OUT_OOS_EXCLUDED"
        ),
        "candidate_schema": [
            "hypothesis",
            "regime_filter",
            "entry",
            "exit",
            "inventory_guard",
            "invalidation",
            "expected_failure_mode",
            "one_factor_difference",
            "complexity",
            "cost",
        ],
        "spike_regime_axes": [
            "ATR_PERCENTILE_VOL_EXPANSION",
            "RANGE_VS_TREND_ALIGNED_VS_TREND_REVERSE_SPIKE",
            "FOLLOW_THROUGH_VS_MEAN_REVERSION",
            "DIRECTION_SESSION_SPREAD_LIQUIDITY",
            "DISTANCE_TO_PREV_DAY_EXTREME_ROUND_NUMBER_STRUCTURE",
            "INVENTORY_BIAS_AND_LONG_SHORT_CONCENTRATION",
        ],
        "mean_revert_mechanism_decomposition": [
            "REGIME_MISMATCH",
            "INVENTORY_CONCENTRATION",
            "PROFIT_GIVEBACK",
        ],
        "siblings_per_parent_max": 3,
        "one_factor_difference_required": True,
        "multiple_testing_budget_per_parent": 3,
        "evaluation_order": [
            "CALIBRATION",
            "SEALED_OOS",
            "SEPARATE_MONTH_OR_OLHC",
        ],
        "champion_challenger_registry_retains_all_12_strategies": True,
        "portfolio_evaluation": {
            "reconstruct_combined_equity_curve": True,
            "individual_profit_sum_is_sufficient": False,
            "measure": [
                "SYNCHRONOUS_PNL_COVARIANCE_CORRELATION",
                "COMMON_CURRENCY_DIRECTION_SESSION_REGIME_EXPOSURE",
                "SIMULTANEOUS_LOSS_AND_INVENTORY_OVERLAP",
                "MARGINAL_NET_AND_MARGINAL_DRAWDOWN",
                "DIVERSIFICATION_BENEFIT",
                "TURNOVER_COST_AND_MARGIN_COMPETITION",
            ],
            "addition_gate": (
                "SEALED_OOS_MARGINAL_NET_POSITIVE_AND_PF_OR_RISK_ADJUSTED_"
                "IMPROVEMENT_AND_DD_RUIN_NONWORSE"
            ),
            "correlated_name_only_duplicates_rejected": True,
        },
        "live_permission": False,
        "broker_mutation_allowed": False,
        "automation_created": False,
    }
    result = {**body, "factory_sha256": canonical_portfolio_sha256(body)}
    _atomic_json(output_root / "mechanism-sibling-factory-v2.json", result)
    return result


__all__ = [
    "AI_COST_JPY_PER_USD",
    "CALIBRATION_CONTRACT",
    "FACTORY_CONTRACT",
    "MAX_PHASE_B_CALLS",
    "OOS_RESULT_CONTRACT",
    "POLICIES",
    "REGIME_CACHE_CONTRACT",
    "REGIME_MATRIX_CONTRACT",
    "WALK_FORWARD_CONTRACT",
    "WORKER_RESPONSE_SCHEMA",
    "aggregate_oos_v2",
    "build_factory_contract",
    "build_regime_cache",
    "build_strategy_regime_matrix",
    "calibrate_v2",
    "deterministic_oos_session_v2",
    "deterministic_v2_worker_response",
    "load_regime_cache",
    "seal_walk_forward_contract",
    "worker_session_v2",
]
