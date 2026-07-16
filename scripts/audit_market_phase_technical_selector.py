#!/usr/bin/env python3
"""Select technical combinations and thresholds separately by market phase.

All features are computed from complete M5 candles available at forecast time.
Rolling phase boundaries use only earlier rows. A chronological validation block
locks one rule/orientation/strength threshold per phase; the final holdout is
untouched until that choice is fixed. Exact entry-relative BID/ASK opens include
the paid spread.

This is direction evidence only. A surviving phase/rule still needs an exact S5
TP/SL replay and the ordinary verifier/risk/gateway chain before live use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for item in (SRC, SCRIPT_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import oanda_history_replay_validate as replay  # noqa: E402
from train_causal_technical_forecaster import (  # noqa: E402
    _label_frame,
    _pair_feature_frame,
)

from quant_rabbit.technical_forecast_evaluation import (  # noqa: E402
    directional_metrics,
)


CONTRACT = "QR_MARKET_PHASE_TECHNICAL_SELECTOR_V1"
PHASES = ("PRE_RANGE", "RANGE", "PRE_TREND", "TREND")
SESSION_BUCKETS = (
    "UTC_00_08",
    "UTC_08_13",
    "UTC_13_17",
    "UTC_17_22",
    "UTC_22_24",
)
RULES = (
    "breakout_fast",
    "trend_fast",
    "trend_slow",
    "pullback_in_trend",
    "mean_revert_fast",
    "mean_revert_slow",
)
THRESHOLD_QUANTILES = (0.50, 0.70, 0.85, 0.95)


def main() -> int:
    args = _parse_args()
    np, pd = _dependencies()
    history_dirs = replay._history_dirs(
        args.history_dir,
        granularity="M5",
        auto_min_days=0.0,
    )
    candles, candle_stats = replay._load_candles(history_dirs, granularity="M5")
    pair_codes = {pair: index for index, pair in enumerate(sorted(candles))}
    frames = []
    for pair, pair_candles in sorted(candles.items()):
        frame = _pair_feature_frame(
            pair,
            pair_candles,
            pair_code=pair_codes[pair],
            np=np,
            pd=pd,
        )
        frames.append(_phase_and_rule_frame(frame, np=np, pd=pd))
    dataset = pd.concat(
        [_label_frame(frame, horizon_min=args.horizon_min, np=np, pd=pd) for frame in frames]
    ).sort_index()
    # The gateway can observe and reject the executable next-open quote.  A
    # forecast-candle spread is not the cost actually paid and previously let
    # rollover-widened entries leak through the cap.
    dataset = dataset.loc[
        dataset["entry_spread_pips"] <= args.spread_cap_pips
    ].copy()
    timestamps = np.asarray(sorted(dataset.index.unique()))
    validation_at = timestamps[int(len(timestamps) * args.train_fraction)]
    holdout_at = timestamps[
        int(len(timestamps) * (args.train_fraction + args.validation_fraction))
    ]
    validation = dataset.loc[
        (dataset.index >= validation_at)
        & (dataset.index < holdout_at)
        & (dataset["future_timestamp_utc"] < holdout_at)
    ]
    holdout = dataset.loc[dataset.index >= holdout_at]

    by_phase: dict[str, Any] = {}
    validation_selected: list[dict[str, Any]] = []
    holdout_selected: list[dict[str, Any]] = []
    runtime_selector: dict[str, Any] = {}
    by_pair_phase: dict[str, Any] = {}
    pair_phase_runtime_selector: dict[str, dict[str, Any]] = {}
    pair_validation_selected: list[dict[str, Any]] = []
    pair_holdout_selected: list[dict[str, Any]] = []
    by_pair_phase_session: dict[str, Any] = {}
    pair_phase_session_research_selector: dict[str, dict[str, Any]] = {}
    pair_phase_session_holdout_selector: dict[str, dict[str, Any]] = {}
    session_validation_selected: list[dict[str, Any]] = []
    session_holdout_selected: list[dict[str, Any]] = []
    strict_session_validation_selected: list[dict[str, Any]] = []
    strict_session_holdout_selected: list[dict[str, Any]] = []
    for phase in PHASES:
        validation_phase = validation.loc[validation["market_phase"] == phase]
        holdout_phase = holdout.loc[holdout["market_phase"] == phase]
        selection = _select_on_validation(
            validation_phase,
            horizon_min=args.horizon_min,
            minimum_trades=args.minimum_validation_trades,
            minimum_days=args.minimum_validation_days,
            np=np,
        )
        locked = selection.get("selected")
        validation_rows: list[dict[str, Any]] = []
        phase_holdout_rows: list[dict[str, Any]] = []
        if isinstance(locked, Mapping):
            validation_rows = _selected_rows(
                validation_phase,
                rule=str(locked["rule"]),
                orientation=int(locked["orientation"]),
                threshold=float(locked["threshold"]),
                horizon_min=args.horizon_min,
            )
            phase_holdout_rows = _selected_rows(
                holdout_phase,
                rule=str(locked["rule"]),
                orientation=int(locked["orientation"]),
                threshold=float(locked["threshold"]),
                horizon_min=args.horizon_min,
            )
        validation_metrics = directional_metrics(validation_rows)
        holdout_metrics = directional_metrics(phase_holdout_rows)
        candidate = _phase_candidate(
            validation_metrics,
            holdout_metrics,
            minimum_validation_trades=args.minimum_validation_trades,
            minimum_holdout_trades=args.minimum_holdout_trades,
            minimum_days=args.minimum_holdout_days,
        )
        if candidate and isinstance(locked, Mapping):
            runtime_selector[phase] = {
                "rule": locked["rule"],
                "orientation": "DIRECT" if int(locked["orientation"]) == 1 else "INVERSE",
                "minimum_absolute_score": locked["threshold"],
            }
            validation_selected.extend(validation_rows)
            holdout_selected.extend(phase_holdout_rows)
        by_phase[phase] = {
            "validation_rows": len(validation_phase),
            "holdout_rows": len(holdout_phase),
            "locked_selection": locked,
            "validation_metrics": validation_metrics,
            "holdout_metrics": holdout_metrics,
            "direction_candidate": candidate,
            "selection_candidates": selection.get("candidates", []),
        }

        pairs = sorted(
            set(map(str, validation_phase["pair"].unique()))
            & set(map(str, holdout_phase["pair"].unique()))
        )
        for pair in pairs:
            validation_pair = validation_phase.loc[validation_phase["pair"] == pair]
            holdout_pair = holdout_phase.loc[holdout_phase["pair"] == pair]
            pair_selection = _select_on_validation(
                validation_pair,
                horizon_min=args.horizon_min,
                minimum_trades=args.minimum_pair_validation_trades,
                minimum_days=args.minimum_pair_validation_days,
                np=np,
            )
            pair_locked = pair_selection.get("selected")
            pair_validation_rows: list[dict[str, Any]] = []
            pair_holdout_rows: list[dict[str, Any]] = []
            if isinstance(pair_locked, Mapping):
                pair_validation_rows = _selected_rows(
                    validation_pair,
                    rule=str(pair_locked["rule"]),
                    orientation=int(pair_locked["orientation"]),
                    threshold=float(pair_locked["threshold"]),
                    horizon_min=args.horizon_min,
                )
                pair_holdout_rows = _selected_rows(
                    holdout_pair,
                    rule=str(pair_locked["rule"]),
                    orientation=int(pair_locked["orientation"]),
                    threshold=float(pair_locked["threshold"]),
                    horizon_min=args.horizon_min,
                )
            pair_validation_metrics = directional_metrics(pair_validation_rows)
            pair_holdout_metrics = directional_metrics(pair_holdout_rows)
            pair_candidate = _phase_candidate(
                pair_validation_metrics,
                pair_holdout_metrics,
                minimum_validation_trades=args.minimum_pair_validation_trades,
                minimum_holdout_trades=args.minimum_pair_holdout_trades,
                minimum_days=args.minimum_pair_holdout_days,
            )
            key = f"{pair}:{phase}"
            by_pair_phase[key] = {
                "pair": pair,
                "phase": phase,
                "locked_selection": pair_locked,
                "validation_metrics": pair_validation_metrics,
                "holdout_metrics": pair_holdout_metrics,
                "direction_candidate": pair_candidate,
            }
            if pair_candidate and isinstance(pair_locked, Mapping):
                pair_phase_runtime_selector.setdefault(pair, {})[phase] = {
                    "rule": pair_locked["rule"],
                    "orientation": (
                        "DIRECT"
                        if int(pair_locked["orientation"]) == 1
                        else "INVERSE"
                    ),
                    "minimum_absolute_score": pair_locked["threshold"],
                }
                pair_validation_selected.extend(pair_validation_rows)
                pair_holdout_selected.extend(pair_holdout_rows)

            sessions = sorted(
                set(map(str, validation_pair["utc_session_bucket"].unique()))
                & set(map(str, holdout_pair["utc_session_bucket"].unique()))
            )
            for session in sessions:
                validation_cell = validation_pair.loc[
                    validation_pair["utc_session_bucket"] == session
                ]
                holdout_cell = holdout_pair.loc[
                    holdout_pair["utc_session_bucket"] == session
                ]
                cell_selection = _select_on_validation(
                    validation_cell,
                    horizon_min=args.horizon_min,
                    minimum_trades=args.minimum_session_validation_trades,
                    minimum_days=args.minimum_session_validation_days,
                    np=np,
                )
                cell_locked = cell_selection.get("selected")
                cell_validation_rows: list[dict[str, Any]] = []
                cell_holdout_rows: list[dict[str, Any]] = []
                if isinstance(cell_locked, Mapping):
                    cell_validation_rows = _selected_rows(
                        validation_cell,
                        rule=str(cell_locked["rule"]),
                        orientation=int(cell_locked["orientation"]),
                        threshold=float(cell_locked["threshold"]),
                        horizon_min=args.horizon_min,
                    )
                    cell_holdout_rows = _selected_rows(
                        holdout_cell,
                        rule=str(cell_locked["rule"]),
                        orientation=int(cell_locked["orientation"]),
                        threshold=float(cell_locked["threshold"]),
                        horizon_min=args.horizon_min,
                    )
                cell_validation_metrics = directional_metrics(
                    cell_validation_rows
                )
                cell_holdout_metrics = directional_metrics(cell_holdout_rows)
                research_survivor = _validation_survivor(
                    cell_validation_metrics,
                    minimum_trades=args.minimum_session_validation_trades,
                    minimum_days=args.minimum_session_validation_days,
                )
                cell_candidate = _phase_candidate(
                    cell_validation_metrics,
                    cell_holdout_metrics,
                    minimum_validation_trades=(
                        args.minimum_session_validation_trades
                    ),
                    minimum_holdout_trades=args.minimum_session_holdout_trades,
                    minimum_days=args.minimum_session_holdout_days,
                )
                cell_key = f"{pair}:{phase}:{session}"
                by_pair_phase_session[cell_key] = {
                    "pair": pair,
                    "phase": phase,
                    "utc_session_bucket": session,
                    "locked_selection": cell_locked,
                    "validation_metrics": cell_validation_metrics,
                    "validation_research_survivor": research_survivor,
                    "holdout_metrics": cell_holdout_metrics,
                    "direction_candidate": cell_candidate,
                }
                if research_survivor and isinstance(cell_locked, Mapping):
                    pair_phase_session_research_selector.setdefault(
                        pair, {}
                    ).setdefault(phase, {})[session] = _selector_view(cell_locked)
                    session_validation_selected.extend(cell_validation_rows)
                    session_holdout_selected.extend(cell_holdout_rows)
                if cell_candidate and isinstance(cell_locked, Mapping):
                    pair_phase_session_holdout_selector.setdefault(
                        pair, {}
                    ).setdefault(phase, {})[session] = _selector_view(cell_locked)
                    strict_session_validation_selected.extend(
                        cell_validation_rows
                    )
                    strict_session_holdout_selected.extend(cell_holdout_rows)

    aggregate_validation = directional_metrics(validation_selected)
    aggregate_holdout = directional_metrics(holdout_selected)
    pair_validation_selected = _merge_non_overlapping_rows(
        pair_validation_selected,
        horizon_min=args.horizon_min,
    )
    pair_holdout_selected = _merge_non_overlapping_rows(
        pair_holdout_selected,
        horizon_min=args.horizon_min,
    )
    pair_aggregate_validation = directional_metrics(pair_validation_selected)
    pair_aggregate_holdout = directional_metrics(pair_holdout_selected)
    session_validation_selected = _merge_non_overlapping_rows(
        session_validation_selected,
        horizon_min=args.horizon_min,
    )
    session_holdout_selected = _merge_non_overlapping_rows(
        session_holdout_selected,
        horizon_min=args.horizon_min,
    )
    strict_session_validation_selected = _merge_non_overlapping_rows(
        strict_session_validation_selected,
        horizon_min=args.horizon_min,
    )
    strict_session_holdout_selected = _merge_non_overlapping_rows(
        strict_session_holdout_selected,
        horizon_min=args.horizon_min,
    )
    session_validation_metrics = directional_metrics(session_validation_selected)
    session_holdout_metrics = directional_metrics(session_holdout_selected)
    strict_session_validation_metrics = directional_metrics(
        strict_session_validation_selected
    )
    strict_session_holdout_metrics = directional_metrics(
        strict_session_holdout_selected
    )
    args.prediction_output_dir.mkdir(parents=True, exist_ok=True)
    prediction_stem = f"market_phase_session_{args.horizon_min}m"
    validation_predictions_path = (
        args.prediction_output_dir / f"{prediction_stem}_validation.jsonl"
    )
    holdout_predictions_path = (
        args.prediction_output_dir / f"{prediction_stem}_holdout.jsonl"
    )
    _write_jsonl(validation_predictions_path, session_validation_selected)
    _write_jsonl(holdout_predictions_path, session_holdout_selected)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": CONTRACT,
        "history_dirs": [str(path.resolve()) for path in history_dirs],
        "history_candles_sha256": replay._truth_candles_digest(candles),
        "history_granularity": "M5",
        "history_price_component": "BID_ASK",
        "horizon_min": args.horizon_min,
        "spread_cap_pips": args.spread_cap_pips,
        "validation_from_utc": validation_at.isoformat(),
        "holdout_from_utc": holdout_at.isoformat(),
        "phase_semantics": _phase_semantics(),
        "session_semantics": _session_semantics(),
        "rule_semantics": _rule_semantics(),
        "runtime_selector": runtime_selector,
        "selected_phase_count": len(runtime_selector),
        "aggregate_validation_metrics": aggregate_validation,
        "aggregate_holdout_metrics": aggregate_holdout,
        "direction_candidate": bool(runtime_selector)
        and _aggregate_candidate(aggregate_validation, aggregate_holdout),
        "pair_phase_runtime_selector": pair_phase_runtime_selector,
        "selected_pair_phase_count": sum(
            len(phases) for phases in pair_phase_runtime_selector.values()
        ),
        "pair_phase_aggregate_validation_metrics": pair_aggregate_validation,
        "pair_phase_aggregate_holdout_metrics": pair_aggregate_holdout,
        "pair_phase_direction_candidate": bool(pair_phase_runtime_selector)
        and _aggregate_candidate(
            pair_aggregate_validation,
            pair_aggregate_holdout,
        ),
        "pair_phase_session_research_selector": (
            pair_phase_session_research_selector
        ),
        "selected_pair_phase_session_research_count": _nested_selector_count(
            pair_phase_session_research_selector
        ),
        "pair_phase_session_validation_metrics": session_validation_metrics,
        "pair_phase_session_holdout_metrics": session_holdout_metrics,
        "pair_phase_session_holdout_selector": (
            pair_phase_session_holdout_selector
        ),
        "selected_pair_phase_session_holdout_count": _nested_selector_count(
            pair_phase_session_holdout_selector
        ),
        "pair_phase_session_strict_validation_metrics": (
            strict_session_validation_metrics
        ),
        "pair_phase_session_strict_holdout_metrics": (
            strict_session_holdout_metrics
        ),
        "validation_predictions_path": str(
            validation_predictions_path.resolve()
        ),
        "validation_predictions_sha256": _file_sha256(
            validation_predictions_path
        ),
        "holdout_predictions_path": str(holdout_predictions_path.resolve()),
        "holdout_predictions_sha256": _file_sha256(
            holdout_predictions_path
        ),
        "promotion_allowed": False,
        "promotion_blockers": [
            "DIRECTION_EDGE_REQUIRES_LOCKED_S5_TP_SL_VEHICLE",
            "FORWARD_LIVE_SHADOW_REQUIRED",
        ],
        "by_phase": by_phase,
        "by_pair_phase": by_pair_phase,
        "by_pair_phase_session": by_pair_phase_session,
        **candle_stats,
    }
    _write_json(args.report_output, report)
    _write_json(args.selector_output, {
        "contract": CONTRACT,
        "source_report_sha256": _sha256(report),
        "horizon_min": args.horizon_min,
        "runtime_selector": runtime_selector,
        "pair_phase_runtime_selector": pair_phase_runtime_selector,
        "pair_phase_session_research_selector": (
            pair_phase_session_research_selector
        ),
        "pair_phase_session_holdout_selector": (
            pair_phase_session_holdout_selector
        ),
        "validation_predictions_sha256": _file_sha256(
            validation_predictions_path
        ),
        "holdout_predictions_sha256": _file_sha256(
            holdout_predictions_path
        ),
        "live_permission": False,
        "requires_s5_vehicle_replay": True,
    })
    print(f"wrote {args.report_output}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", type=Path, action="append", required=True)
    parser.add_argument("--horizon-min", type=int, default=60)
    parser.add_argument("--spread-cap-pips", type=float, default=2.0)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-validation-trades", type=int, default=40)
    parser.add_argument("--minimum-validation-days", type=int, default=12)
    parser.add_argument("--minimum-holdout-trades", type=int, default=40)
    parser.add_argument("--minimum-holdout-days", type=int, default=12)
    parser.add_argument("--minimum-pair-validation-trades", type=int, default=20)
    parser.add_argument("--minimum-pair-validation-days", type=int, default=8)
    parser.add_argument("--minimum-pair-holdout-trades", type=int, default=20)
    parser.add_argument("--minimum-pair-holdout-days", type=int, default=8)
    parser.add_argument("--minimum-session-validation-trades", type=int, default=12)
    parser.add_argument("--minimum-session-validation-days", type=int, default=6)
    parser.add_argument("--minimum-session-holdout-trades", type=int, default=12)
    parser.add_argument("--minimum-session-holdout-days", type=int, default=6)
    parser.add_argument(
        "--prediction-output-dir",
        type=Path,
        default=(
            ROOT
            / "logs"
            / "reports"
            / "forecast_improvement"
            / "market_phase_predictions"
        ),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "logs/reports/forecast_improvement/market_phase_selector_latest.json",
    )
    parser.add_argument(
        "--selector-output",
        type=Path,
        default=ROOT / "logs/reports/forecast_improvement/market_phase_selector_candidate.json",
    )
    return parser.parse_args()


def _dependencies():
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("phase selector audit requires numpy and pandas") from exc
    return np, pd


def _phase_and_rule_frame(frame, *, np, pd):
    result = frame.copy()
    hours = result.index.hour
    result["utc_session_bucket"] = np.select(
        [hours < 8, hours < 13, hours < 17, hours < 22],
        SESSION_BUCKETS[:-1],
        default=SESSION_BUCKETS[-1],
    )
    atr14 = result["atr_14"].replace(0.0, np.nan)
    atr48 = result["atr_48"].replace(0.0, np.nan)
    atr288 = result["atr_288"].replace(0.0, np.nan)
    result["trend_strength"] = result["ema_gap_12_48"].abs() / atr48
    result["volatility_ratio"] = atr14 / atr288
    result["trend_strength_change"] = result["trend_strength"] - result["trend_strength"].shift(12)
    alignment = (
        (np.sign(result["return_12"]) == np.sign(result["return_48"]))
        & (np.sign(result["return_48"]) == np.sign(result["ema_gap_12_48"]))
    )
    window = 30 * 288
    trend_low = result["trend_strength"].rolling(window, min_periods=7 * 288).quantile(0.25).shift(1)
    trend_high = result["trend_strength"].rolling(window, min_periods=7 * 288).quantile(0.75).shift(1)
    volatility_low = result["volatility_ratio"].rolling(window, min_periods=7 * 288).quantile(0.25).shift(1)
    trend = alignment & (result["trend_strength"] >= trend_high)
    pre_trend = (~trend) & (
        (result["volatility_ratio"] <= volatility_low)
        | (alignment & (result["trend_strength_change"] > 0.0))
    )
    range_state = (~trend) & (~pre_trend) & (result["trend_strength"] <= trend_low)
    result["market_phase"] = np.select(
        [trend, pre_trend, range_state],
        ["TREND", "PRE_TREND", "RANGE"],
        default="PRE_RANGE",
    )
    location12 = (result["range_location_12"] - 0.5) * 2.0
    location48 = (result["range_location_48"] - 0.5) * 2.0
    rsi = (result["rsi_14"] - 50.0) / 50.0
    result["breakout_fast"] = (
        result["return_3"] / atr14
        + result["body_atr_ratio"]
        + location48
    ) / 3.0
    result["trend_fast"] = (
        result["return_3"] / atr14
        + result["return_12"] / atr48
        + result["ema_gap_5_24"] / atr48
    ) / 3.0
    result["trend_slow"] = (
        result["return_12"] / atr48
        + result["return_48"] / atr288
        + result["ema_gap_12_48"] / atr48
    ) / 3.0
    result["pullback_in_trend"] = (
        result["ema_gap_12_48"] / atr48
        - result["return_3"] / atr14
        - location12
    ) / 3.0
    result["mean_revert_fast"] = (
        -result["return_3"] / atr14
        - rsi
        - location12
    ) / 3.0
    result["mean_revert_slow"] = (
        -result["return_12"] / atr48
        - rsi
        - location48
    ) / 3.0
    return result


def _select_on_validation(frame, *, horizon_min: int, minimum_trades: int, minimum_days: int, np):
    candidates: list[dict[str, Any]] = []
    for rule in RULES:
        absolute = frame[rule].abs().dropna().to_numpy()
        if not len(absolute):
            continue
        thresholds = sorted({round(float(np.quantile(absolute, q)), 9) for q in THRESHOLD_QUANTILES})
        for orientation in (1, -1):
            for threshold in thresholds:
                rows = _selected_rows(
                    frame,
                    rule=rule,
                    orientation=orientation,
                    threshold=threshold,
                    horizon_min=horizon_min,
                )
                metrics = directional_metrics(rows)
                enough = bool(
                    metrics["trades"] >= minimum_trades
                    and metrics["active_days"] >= minimum_days
                )
                candidates.append({
                    "rule": rule,
                    "orientation": orientation,
                    "threshold": threshold,
                    "minimum_evidence_met": enough,
                    "metrics": metrics,
                })
    eligible = [row for row in candidates if row["minimum_evidence_met"]]
    ranked = eligible or candidates
    ranked.sort(key=_candidate_rank, reverse=True)
    return {"selected": ranked[0] if ranked else None, "candidates": ranked[:12]}


def _candidate_rank(row: Mapping[str, Any]):
    metrics = row.get("metrics") or {}
    return (
        _rank(metrics.get("one_sided_95_daily_lower_pips")),
        _rank(metrics.get("one_sided_95_mean_lower_pips")),
        _rank(metrics.get("mean_pips")),
        int(metrics.get("trades") or 0),
    )


def _selected_rows(frame, *, rule: str, orientation: int, threshold: float, horizon_min: int):
    score = frame[rule] * orientation
    qualified_mask = score.notna() & (score.abs() >= threshold)
    qualified = frame.loc[
        qualified_mask,
        [
            "pair",
            "market_phase",
            "utc_session_bucket",
            "entry_timestamp_utc",
            "future_timestamp_utc",
            "target_mid_pips",
            "long_pips",
            "short_pips",
            "entry_spread_pips",
            "exit_spread_pips",
            "roundtrip_spread_cost_pips",
        ],
    ].copy()
    # Dataset timestamps repeat across pairs, so positional assignment is
    # required; label-based reindexing on the duplicate DatetimeIndex is
    # ambiguous and newer pandas correctly rejects it.
    qualified["technical_score"] = score.loc[qualified_mask].to_numpy()
    rows: list[dict[str, Any]] = []
    import pandas as pd

    hold = pd.Timedelta(minutes=horizon_min)
    for pair, group in qualified.groupby("pair", sort=False):
        accepted_until = None
        for item in group.itertuples():
            timestamp = item.Index
            if accepted_until is not None and timestamp < accepted_until:
                continue
            predicted = float(item.technical_score)
            direction = "UP" if predicted >= 0.0 else "DOWN"
            long_pips = float(item.long_pips)
            short_pips = float(item.short_pips)
            gross_directional_pips = (
                float(item.target_mid_pips)
                if direction == "UP"
                else -float(item.target_mid_pips)
            )
            rows.append({
                "timestamp_utc": timestamp.isoformat(),
                "entry_timestamp_utc": item.entry_timestamp_utc.isoformat(),
                "future_timestamp_utc": item.future_timestamp_utc.isoformat(),
                "pair": str(pair),
                "market_phase": str(item.market_phase),
                "utc_session_bucket": str(item.utc_session_bucket),
                "technical_rule": rule,
                "orientation": "DIRECT" if orientation == 1 else "INVERSE",
                "predicted_pips": predicted,
                "predicted_direction": direction,
                "long_pips": long_pips,
                "short_pips": short_pips,
                "entry_spread_pips": float(item.entry_spread_pips),
                "exit_spread_pips": float(item.exit_spread_pips),
                "roundtrip_spread_cost_pips": float(
                    item.roundtrip_spread_cost_pips
                ),
                "gross_directional_pips": gross_directional_pips,
                "executed_pips": long_pips if direction == "UP" else short_pips,
            })
            accepted_until = timestamp + hold
    return rows


def _merge_non_overlapping_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    horizon_min: int,
) -> list[dict[str, Any]]:
    """Reapply pair-local flatness after independently selected phases merge."""

    accepted_until: dict[str, datetime] = {}
    selected: list[dict[str, Any]] = []
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            datetime.fromisoformat(str(row["timestamp_utc"])),
            str(row["pair"]),
        ),
    )
    from datetime import timedelta

    hold = timedelta(minutes=horizon_min)
    for row in ordered:
        pair = str(row["pair"])
        timestamp = datetime.fromisoformat(str(row["timestamp_utc"]))
        if timestamp < accepted_until.get(pair, timestamp):
            continue
        selected.append(row)
        accepted_until[pair] = timestamp + hold
    return selected


def _phase_candidate(validation, holdout, *, minimum_validation_trades: int, minimum_holdout_trades: int, minimum_days: int) -> bool:
    return bool(
        validation.get("trades", 0) >= minimum_validation_trades
        and holdout.get("trades", 0) >= minimum_holdout_trades
        and holdout.get("active_days", 0) >= minimum_days
        and _positive(validation.get("mean_pips"))
        and _positive(holdout.get("one_sided_95_mean_lower_pips"))
        and _positive(holdout.get("one_sided_95_daily_lower_pips"))
        and _finite(holdout.get("profit_factor")) is not None
        and float(holdout["profit_factor"]) > 1.0
        and float(holdout.get("positive_day_rate") or 0.0) >= 0.55
    )


def _validation_survivor(
    metrics: Mapping[str, Any],
    *,
    minimum_trades: int,
    minimum_days: int,
) -> bool:
    """Admit a validation-only cell to the locked S5 vehicle search.

    This is deliberately weaker than final direction acceptance: it keeps a
    positive executable point estimate for joint TP/SL selection, while the
    untouched holdout still has to clear the strict confidence bounds.
    """

    return bool(
        int(metrics.get("trades") or 0) >= minimum_trades
        and int(metrics.get("active_days") or 0) >= minimum_days
        and _positive(metrics.get("mean_pips"))
        and float(metrics.get("profit_factor") or 0.0) > 1.0
    )


def _selector_view(locked: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rule": locked["rule"],
        "orientation": (
            "DIRECT" if int(locked["orientation"]) == 1 else "INVERSE"
        ),
        "minimum_absolute_score": locked["threshold"],
    }


def _nested_selector_count(selector: Mapping[str, Any]) -> int:
    count = 0
    for phases in selector.values():
        if not isinstance(phases, Mapping):
            continue
        for sessions in phases.values():
            if isinstance(sessions, Mapping):
                count += len(sessions)
    return count


def _aggregate_candidate(validation, holdout) -> bool:
    return bool(
        _positive(validation.get("mean_pips"))
        and _positive(holdout.get("one_sided_95_mean_lower_pips"))
        and _positive(holdout.get("one_sided_95_daily_lower_pips"))
        and float(holdout.get("profit_factor") or 0.0) > 1.0
    )


def _phase_semantics() -> dict[str, str]:
    return {
        "TREND": "aligned 12/48-bar return and EMA direction with trend strength at/above its causal rolling upper quartile",
        "PRE_TREND": "causal rolling lower-quartile volatility compression or aligned trend strength increasing over 12 bars",
        "RANGE": "trend strength at/below its causal rolling lower quartile outside PRE_TREND",
        "PRE_RANGE": "remaining transition/deceleration state",
    }


def _session_semantics() -> dict[str, str]:
    return {
        "UTC_00_08": "fixed UTC 00:00-07:59 cohort",
        "UTC_08_13": "fixed UTC 08:00-12:59 cohort",
        "UTC_13_17": "fixed UTC 13:00-16:59 cohort",
        "UTC_17_22": "fixed UTC 17:00-21:59 cohort",
        "UTC_22_24": "fixed UTC 22:00-23:59 cohort",
    }


def _rule_semantics() -> dict[str, str]:
    return {
        "breakout_fast": "3-bar return + candle body/ATR + 48-bar range location",
        "trend_fast": "3/12-bar return + fast EMA gap, ATR normalized",
        "trend_slow": "12/48-bar return + balanced EMA gap, ATR normalized",
        "pullback_in_trend": "balanced EMA trend minus fast pullback and short-range location",
        "mean_revert_fast": "opposite 3-bar return + RSI + short-range location",
        "mean_revert_slow": "opposite 12-bar return + RSI + 48-bar range location",
    }


def _rank(value: Any) -> float:
    parsed = _finite(value)
    return parsed if parsed is not None else -math.inf


def _positive(value: Any) -> bool:
    parsed = _finite(value)
    return parsed is not None and parsed > 0.0


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(
        json.dumps(
            _json_safe(dict(row)),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


if __name__ == "__main__":
    raise SystemExit(main())
