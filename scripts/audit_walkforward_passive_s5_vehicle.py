#!/usr/bin/env python3
"""Lock and test a passive TP/SL vehicle for walk-forward directions on S5."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for item in (SRC, SCRIPT_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import oanda_history_replay_validate as replay  # noqa: E402

from quant_rabbit.forecast_passive_limit_replay import (  # noqa: E402
    MARKET_BRACKET_REPLAY_CONTRACT,
    MARKET_STOP_TIME_CLOSE_REPLAY_CONTRACT,
    PASSIVE_LIMIT_REPLAY_CONTRACT,
    replay_metrics,
    simulate_market_bracket,
    simulate_market_stop_time_close,
    simulate_passive_limit,
)


AUDIT_CONTRACT = "QR_WALKFORWARD_ENTRY_S5_VEHICLE_V2"
VEHICLE_HORIZON_MIN = 1440.0
TIME_CLOSE_QUOTE_GRACE_SECONDS = 60.0
ENTRY_TTL_GRID_MIN = (5.0, 15.0, 60.0)
REWARD_GRID_PIPS = (5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0)
RISK_GRID_PIPS = (5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0)
TIME_CLOSE_RISK_GRID_PIPS = (20.0, 30.0, 50.0, 80.0, 100.0)
PROMOTION_BLOCKERS = (
    "S5_VEHICLE_REQUIRES_NEW_FORWARD_SHADOW",
    "RESEARCH_ARTIFACT_NOT_CONNECTED_TO_LIVE_INTENT_GENERATOR",
    "RISK_AND_GATEWAY_RECEIPTS_REQUIRED",
)


def main() -> int:
    args = _parse_args()
    validation_predictions = _load_predictions(args.validation_predictions)
    holdout_predictions = _load_predictions(args.holdout_predictions)
    selected_pairs = {
        item.strip().upper()
        for item in str(args.pairs or "").split(",")
        if item.strip()
    }
    if selected_pairs:
        validation_predictions = _filter_predictions(
            validation_predictions,
            pairs=selected_pairs,
        )
        holdout_predictions = _filter_predictions(
            holdout_predictions,
            pairs=selected_pairs,
        )
    all_predictions = validation_predictions + holdout_predictions
    windows = _prediction_windows(
        all_predictions,
        vehicle_horizon_min=args.vehicle_horizon_min,
        exit_grace_seconds=TIME_CLOSE_QUOTE_GRACE_SECONDS,
    )
    candles, candle_stats = replay._load_candles(
        args.history_dir,
        granularity="S5",
        windows_by_pair=windows,
    )
    candle_times = {
        pair: [candle.timestamp_utc for candle in pair_candles]
        for pair, pair_candles in candles.items()
    }

    candidates: list[dict[str, Any]] = []
    for entry_vehicle, entry_ttl, max_hold in _entry_configurations(
        vehicle_horizon_min=args.vehicle_horizon_min,
    ):
        for reward, risk in _vehicle_geometries():
            outcomes = _simulate_predictions(
                validation_predictions,
                candles,
                candle_times,
                entry_ttl_min=entry_ttl,
                max_hold_min=max_hold,
                reward_pips=reward,
                risk_pips=risk,
                entry_vehicle=entry_vehicle,
                vehicle_horizon_min=args.vehicle_horizon_min,
            )
            metrics = replay_metrics(outcomes)
            enough = int(metrics["fills"]) >= args.minimum_validation_fills
            candidates.append(
                {
                    "entry_ttl_min": entry_ttl,
                    "max_hold_min": max_hold,
                    "entry_vehicle": entry_vehicle,
                    "reward_pips": reward,
                    "risk_pips": risk,
                    "minimum_validation_fills_met": enough,
                    "validation_metrics": metrics,
                    "validation_status_counts": dict(
                        sorted(
                            collections.Counter(
                                row.get("status") for row in outcomes
                            ).items()
                        )
                    ),
                }
            )
    for risk in TIME_CLOSE_RISK_GRID_PIPS:
        outcomes = _simulate_predictions(
            validation_predictions,
            candles,
            candle_times,
            entry_ttl_min=0.0,
            max_hold_min=args.vehicle_horizon_min,
            reward_pips=None,
            risk_pips=risk,
            entry_vehicle="MARKET_STOP_TIME_CLOSE",
            vehicle_horizon_min=args.vehicle_horizon_min,
        )
        metrics = replay_metrics(outcomes)
        candidates.append(
            {
                "entry_ttl_min": 0.0,
                "max_hold_min": args.vehicle_horizon_min,
                "entry_vehicle": "MARKET_STOP_TIME_CLOSE",
                "reward_pips": None,
                "risk_pips": risk,
                "minimum_validation_fills_met": (
                    int(metrics["fills"]) >= args.minimum_validation_fills
                ),
                "validation_metrics": metrics,
                "validation_status_counts": dict(
                    sorted(
                        collections.Counter(
                            row.get("status") for row in outcomes
                        ).items()
                    )
                ),
            }
        )
    eligible = [
        candidate
        for candidate in candidates
        if candidate["minimum_validation_fills_met"]
    ]
    ranked = eligible or candidates
    ranked.sort(key=_candidate_rank, reverse=True)
    selected = ranked[0]
    candidate_leaderboard = ranked[:20]
    holdout_outcomes = _simulate_predictions(
        holdout_predictions,
        candles,
        candle_times,
        entry_ttl_min=float(selected["entry_ttl_min"]),
        max_hold_min=float(selected["max_hold_min"]),
        reward_pips=(
            None
            if selected.get("reward_pips") is None
            else float(selected["reward_pips"])
        ),
        risk_pips=(
            None
            if selected.get("risk_pips") is None
            else float(selected["risk_pips"])
        ),
        entry_vehicle=str(selected["entry_vehicle"]),
        vehicle_horizon_min=args.vehicle_horizon_min,
    )
    holdout_metrics = replay_metrics(holdout_outcomes)
    vehicle_candidate = _vehicle_candidate(
        selected["validation_metrics"],
        holdout_metrics,
        minimum_validation_fills=args.minimum_validation_fills,
        minimum_holdout_fills=args.minimum_holdout_fills,
    )
    positive_expectancy_candidate = _positive_expectancy_candidate(
        selected["validation_metrics"],
        holdout_metrics,
        minimum_validation_fills=args.minimum_validation_fills,
        minimum_holdout_fills=args.minimum_holdout_fills,
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": AUDIT_CONTRACT,
        "vehicle_contracts": {
            "MARKET": MARKET_BRACKET_REPLAY_CONTRACT,
            "MARKET_STOP_TIME_CLOSE": MARKET_STOP_TIME_CLOSE_REPLAY_CONTRACT,
            "PASSIVE_LIMIT": PASSIVE_LIMIT_REPLAY_CONTRACT,
        },
        "promotion_allowed": False,
        "promotion_blockers": list(PROMOTION_BLOCKERS),
        "vehicle_candidate": vehicle_candidate,
        "strict_statistical_vehicle_candidate": vehicle_candidate,
        "positive_expectancy_candidate": positive_expectancy_candidate,
        "selection_semantics": (
            "entry vehicle and geometry selected once from validation "
            "conservative S5 metrics; holdout vehicle is locked; "
            + (
                "MARKET_STOP_TIME_CLOSE exits at the first executable quote "
                "at or after the frozen horizon within the configured grace; "
                if str(selected["entry_vehicle"]) == "MARKET_STOP_TIME_CLOSE"
                else "fixed TP/SL exit geometry is used; "
            )
            + "unresolved fills and S5 same-bar ambiguity are charged full risk"
        ),
        "validation_predictions_path": str(args.validation_predictions.resolve()),
        "validation_predictions_sha256": _file_sha256(
            args.validation_predictions
        ),
        "holdout_predictions_path": str(args.holdout_predictions.resolve()),
        "holdout_predictions_sha256": _file_sha256(args.holdout_predictions),
        "history_dirs": [str(path.resolve()) for path in args.history_dir],
        "selected_pairs": sorted(selected_pairs),
        "truth_candles_sha256": replay._truth_candles_digest(candles),
        "entry_ttl_grid_min": list(ENTRY_TTL_GRID_MIN),
        "vehicle_horizon_min": args.vehicle_horizon_min,
        "time_close_quote_grace_seconds": TIME_CLOSE_QUOTE_GRACE_SECONDS,
        "reward_grid_pips": list(REWARD_GRID_PIPS),
        "risk_grid_pips": list(RISK_GRID_PIPS),
        "time_close_risk_grid_pips": list(TIME_CLOSE_RISK_GRID_PIPS),
        "payoff_constraint": "REWARD_PIPS_GT_RISK_PIPS",
        "minimum_validation_fills": args.minimum_validation_fills,
        "minimum_holdout_fills": args.minimum_holdout_fills,
        "grid_candidates": len(candidates),
        "candidate_leaderboard": candidate_leaderboard,
        "selected_vehicle": selected,
        "holdout_metrics": holdout_metrics,
        "holdout_status_counts": dict(
            sorted(
                collections.Counter(
                    row.get("status") for row in holdout_outcomes
                ).items()
            )
        ),
        "holdout_outcomes": holdout_outcomes,
        **candle_stats,
    }
    _write_json(args.report_output, report)
    _write_text(args.report_output.with_suffix(".md"), _markdown(report))
    print(f"wrote {args.report_output}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--holdout-predictions", type=Path, required=True)
    parser.add_argument("--history-dir", type=Path, action="append", required=True)
    parser.add_argument("--minimum-validation-fills", type=int, default=15)
    parser.add_argument("--minimum-holdout-fills", type=int, default=15)
    parser.add_argument(
        "--vehicle-horizon-min",
        type=float,
        default=VEHICLE_HORIZON_MIN,
    )
    parser.add_argument(
        "--pairs",
        default="",
        help="optional comma-separated prediction-pair filter",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT
        / "logs"
        / "reports"
        / "forecast_improvement"
        / "walkforward_passive_s5_vehicle_latest.json",
    )
    return parser.parse_args()


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"prediction row must be an object: {path}")
                rows.append(payload)
    return rows


def _vehicle_geometries() -> list[tuple[float, float]]:
    """Predeclare payoff-positive brackets; reject win-rate-only loss tails."""

    return [
        (reward, risk)
        for reward in REWARD_GRID_PIPS
        for risk in RISK_GRID_PIPS
        if reward > risk
    ]


def _entry_configurations(
    *,
    vehicle_horizon_min: float = VEHICLE_HORIZON_MIN,
) -> list[tuple[str, float, float]]:
    horizon = float(vehicle_horizon_min)
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("vehicle horizon must be finite and positive")
    passive = [
        (
            "PASSIVE_LIMIT",
            entry_ttl,
            horizon - entry_ttl,
        )
        for entry_ttl in ENTRY_TTL_GRID_MIN
        if entry_ttl < horizon
    ]
    return passive + [("MARKET", 0.0, horizon)]


def _filter_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    pairs: set[str],
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if str(row.get("pair") or "").upper() in pairs
    ]


def _prediction_windows(
    rows: Sequence[Mapping[str, Any]],
    *,
    vehicle_horizon_min: float = VEHICLE_HORIZON_MIN,
    exit_grace_seconds: float = 5.0,
) -> dict[str, list[tuple[datetime, datetime]]]:
    windows: dict[str, list[tuple[datetime, datetime]]] = {}
    for row in rows:
        pair = str(row.get("pair") or "").upper()
        entry = _parse_time(row.get("entry_timestamp_utc"))
        if not pair or entry is None:
            continue
        windows.setdefault(pair, []).append(
            (
                entry,
                entry
                + timedelta(minutes=vehicle_horizon_min)
                + timedelta(seconds=exit_grace_seconds),
            )
        )
    return {
        pair: replay._merge_windows(pair_windows)
        for pair, pair_windows in windows.items()
    }


def _simulate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    candles: Mapping[str, Sequence[Any]],
    candle_times: Mapping[str, Sequence[datetime]],
    *,
    entry_ttl_min: float,
    max_hold_min: float,
    reward_pips: float | None,
    risk_pips: float | None,
    entry_vehicle: str = "PASSIVE_LIMIT",
    vehicle_horizon_min: float = VEHICLE_HORIZON_MIN,
) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for source_index, prediction in enumerate(predictions):
        pair = str(prediction.get("pair") or "").upper()
        timestamp = _parse_time(prediction.get("entry_timestamp_utc"))
        direction = str(prediction.get("predicted_direction") or "").upper()
        row = SimpleNamespace(
            source_index=source_index,
            timestamp_utc=timestamp,
            pair=pair,
            direction=direction,
            current_price=None,
            target_price=None,
            invalidation_price=None,
        )
        if entry_vehicle == "MARKET_STOP_TIME_CLOSE":
            if risk_pips is None:
                raise ValueError("time-close vehicle requires fixed risk")
            outcome = simulate_market_stop_time_close(
                row,
                candles.get(pair, ()),
                horizon_min=max_hold_min,
                risk_pips=risk_pips,
                candle_interval=timedelta(seconds=5),
                candle_times=candle_times.get(pair, ()),
                time_close_quote_grace=timedelta(
                    seconds=TIME_CLOSE_QUOTE_GRACE_SECONDS
                ),
            )
        elif entry_vehicle == "MARKET":
            if reward_pips is None or risk_pips is None:
                raise ValueError("market bracket requires reward and risk")
            outcome = simulate_market_bracket(
                row,
                candles.get(pair, ()),
                horizon_min=max_hold_min,
                reward_pips=reward_pips,
                risk_pips=risk_pips,
                candle_interval=timedelta(seconds=5),
                candle_times=candle_times.get(pair, ()),
            )
        else:
            if reward_pips is None or risk_pips is None:
                raise ValueError("passive bracket requires reward and risk")
            outcome = simulate_passive_limit(
                row,
                candles.get(pair, ()),
                horizon_min=vehicle_horizon_min,
                entry_ttl_min=entry_ttl_min,
                max_hold_min=max_hold_min,
                reward_pips=reward_pips,
                risk_pips=risk_pips,
                candle_interval=timedelta(seconds=5),
                candle_times=candle_times.get(pair, ()),
            )
        outcomes.append(
            {
                **outcome,
                "entry_vehicle": entry_vehicle,
                "prediction_timestamp_utc": prediction.get("timestamp_utc"),
                "selected_rule": prediction.get("selected_rule"),
                "orientation": prediction.get("orientation"),
            }
        )
    return outcomes


def _candidate_rank(candidate: Mapping[str, Any]) -> tuple[float, float, int]:
    metrics = candidate.get("validation_metrics") or {}
    return (
        _rank_value(metrics.get("mean_conservative_pips")),
        _rank_value(metrics.get("one_sided_95_mean_lower_pips")),
        int(metrics.get("fills") or 0),
    )


def _vehicle_candidate(
    validation: Mapping[str, Any],
    holdout: Mapping[str, Any],
    *,
    minimum_validation_fills: int,
    minimum_holdout_fills: int,
) -> bool:
    return bool(
        int(validation.get("fills") or 0) >= minimum_validation_fills
        and int(holdout.get("fills") or 0) >= minimum_holdout_fills
        and _rank_value(validation.get("mean_conservative_pips")) > 0.0
        and _rank_value(holdout.get("mean_conservative_pips")) > 0.0
        and _rank_value(holdout.get("one_sided_95_mean_lower_pips")) > 0.0
        and _rank_value(holdout.get("conservative_profit_factor")) > 1.0
    )


def _positive_expectancy_candidate(
    validation: Mapping[str, Any],
    holdout: Mapping[str, Any],
    *,
    minimum_validation_fills: int,
    minimum_holdout_fills: int,
) -> bool:
    """Identify replay-positive probation evidence without claiming proof."""

    return bool(
        int(validation.get("fills") or 0) >= minimum_validation_fills
        and int(holdout.get("fills") or 0) >= minimum_holdout_fills
        and _rank_value(validation.get("mean_conservative_pips")) > 0.0
        and _rank_value(holdout.get("mean_conservative_pips")) > 0.0
        and _rank_value(validation.get("conservative_profit_factor")) > 1.0
        and _rank_value(holdout.get("conservative_profit_factor")) > 1.0
        and _rank_value(holdout.get("positive_day_rate")) >= 0.5
    )


def _rank_value(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return -math.inf
    return result if math.isfinite(result) else result


def _parse_time(value: Any) -> datetime | None:
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
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    _write_text(
        path,
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def _markdown(report: Mapping[str, Any]) -> str:
    selected = report.get("selected_vehicle") or {}
    validation = selected.get("validation_metrics") or {}
    holdout = report.get("holdout_metrics") or {}
    return "\n".join(
        [
            "# Walk-forward S5 entry vehicle",
            "",
            f"- vehicle candidate: `{report.get('vehicle_candidate')}`",
            f"- positive expectancy candidate: `{report.get('positive_expectancy_candidate')}`",
            f"- entry TTL: `{selected.get('entry_ttl_min')}` minutes",
            f"- entry vehicle: `{selected.get('entry_vehicle')}`",
            f"- TP / SL: `{selected.get('reward_pips')}` / `{selected.get('risk_pips')}` pips",
            f"- validation fills / mean: `{validation.get('fills')}` / `{validation.get('mean_conservative_pips')}`",
            f"- holdout fills / mean: `{holdout.get('fills')}` / `{holdout.get('mean_conservative_pips')}`",
            f"- holdout one-sided 95% lower: `{holdout.get('one_sided_95_mean_lower_pips')}`",
            "",
            "Live promotion remains disabled pending a new forward shadow and the normal risk/gateway receipts.",
        ]
    ) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
