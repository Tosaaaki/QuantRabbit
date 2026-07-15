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
    PASSIVE_LIMIT_REPLAY_CONTRACT,
    replay_metrics,
    simulate_passive_limit,
)


AUDIT_CONTRACT = "QR_WALKFORWARD_PASSIVE_S5_VEHICLE_V1"
ENTRY_TTL_GRID_MIN = (5.0, 15.0, 60.0)
REWARD_GRID_PIPS = (5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0)
RISK_GRID_PIPS = (5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0)
PROMOTION_BLOCKERS = (
    "S5_VEHICLE_REQUIRES_NEW_FORWARD_SHADOW",
    "RESEARCH_ARTIFACT_NOT_CONNECTED_TO_LIVE_INTENT_GENERATOR",
    "RISK_AND_GATEWAY_RECEIPTS_REQUIRED",
)


def main() -> int:
    args = _parse_args()
    validation_predictions = _load_predictions(args.validation_predictions)
    holdout_predictions = _load_predictions(args.holdout_predictions)
    all_predictions = validation_predictions + holdout_predictions
    windows = _prediction_windows(all_predictions)
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
    for entry_ttl in ENTRY_TTL_GRID_MIN:
        max_hold = 1440.0 - entry_ttl
        for reward in REWARD_GRID_PIPS:
            for risk in RISK_GRID_PIPS:
                outcomes = _simulate_predictions(
                    validation_predictions,
                    candles,
                    candle_times,
                    entry_ttl_min=entry_ttl,
                    max_hold_min=max_hold,
                    reward_pips=reward,
                    risk_pips=risk,
                )
                metrics = replay_metrics(outcomes)
                enough = int(metrics["fills"]) >= args.minimum_validation_fills
                candidates.append(
                    {
                        "entry_ttl_min": entry_ttl,
                        "max_hold_min": max_hold,
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
    eligible = [
        candidate
        for candidate in candidates
        if candidate["minimum_validation_fills_met"]
    ]
    ranked = eligible or candidates
    ranked.sort(key=_candidate_rank, reverse=True)
    selected = ranked[0]
    holdout_outcomes = _simulate_predictions(
        holdout_predictions,
        candles,
        candle_times,
        entry_ttl_min=float(selected["entry_ttl_min"]),
        max_hold_min=float(selected["max_hold_min"]),
        reward_pips=float(selected["reward_pips"]),
        risk_pips=float(selected["risk_pips"]),
    )
    holdout_metrics = replay_metrics(holdout_outcomes)
    vehicle_candidate = _vehicle_candidate(
        selected["validation_metrics"],
        holdout_metrics,
        minimum_validation_fills=args.minimum_validation_fills,
        minimum_holdout_fills=args.minimum_holdout_fills,
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": AUDIT_CONTRACT,
        "vehicle_contract": PASSIVE_LIMIT_REPLAY_CONTRACT,
        "promotion_allowed": False,
        "promotion_blockers": list(PROMOTION_BLOCKERS),
        "vehicle_candidate": vehicle_candidate,
        "selection_semantics": (
            "geometry selected once from validation conservative S5 metrics; "
            "holdout geometry is locked; no market entry and no time close; "
            "unresolved fills and S5 same-bar ambiguity are charged full risk"
        ),
        "validation_predictions_path": str(args.validation_predictions.resolve()),
        "validation_predictions_sha256": _file_sha256(
            args.validation_predictions
        ),
        "holdout_predictions_path": str(args.holdout_predictions.resolve()),
        "holdout_predictions_sha256": _file_sha256(args.holdout_predictions),
        "history_dirs": [str(path.resolve()) for path in args.history_dir],
        "truth_candles_sha256": replay._truth_candles_digest(candles),
        "entry_ttl_grid_min": list(ENTRY_TTL_GRID_MIN),
        "reward_grid_pips": list(REWARD_GRID_PIPS),
        "risk_grid_pips": list(RISK_GRID_PIPS),
        "minimum_validation_fills": args.minimum_validation_fills,
        "minimum_holdout_fills": args.minimum_holdout_fills,
        "grid_candidates": len(candidates),
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


def _prediction_windows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[tuple[datetime, datetime]]]:
    windows: dict[str, list[tuple[datetime, datetime]]] = {}
    for row in rows:
        pair = str(row.get("pair") or "").upper()
        entry = _parse_time(row.get("entry_timestamp_utc"))
        future = _parse_time(row.get("future_timestamp_utc"))
        if not pair or entry is None or future is None:
            continue
        windows.setdefault(pair, []).append(
            (entry, future + timedelta(seconds=5))
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
    reward_pips: float,
    risk_pips: float,
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
        outcome = simulate_passive_limit(
            row,
            candles.get(pair, ()),
            horizon_min=1440.0,
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
                "prediction_timestamp_utc": prediction.get("timestamp_utc"),
                "selected_rule": prediction.get("selected_rule"),
                "orientation": prediction.get("orientation"),
            }
        )
    return outcomes


def _candidate_rank(candidate: Mapping[str, Any]) -> tuple[float, float, int]:
    metrics = candidate.get("validation_metrics") or {}
    return (
        _rank_value(metrics.get("one_sided_95_mean_lower_pips")),
        _rank_value(metrics.get("mean_conservative_pips")),
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
            "# Walk-forward passive S5 vehicle",
            "",
            f"- vehicle candidate: `{report.get('vehicle_candidate')}`",
            f"- entry TTL: `{selected.get('entry_ttl_min')}` minutes",
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
