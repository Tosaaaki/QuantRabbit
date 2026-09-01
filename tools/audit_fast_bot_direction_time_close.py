#!/usr/bin/env python3
"""Replay raw fast-bot signals with direct/inverse executable time closes.

This is a bounded, read-only diagnostic.  It uses a receipt-backed local OANDA
S5 bid/ask cache, de-overlaps each exact candidate before reading its outcome,
and evaluates only the already-frozen technical-grid arms.  Results are
retrospective research leads, never profitability proof or activation authority.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quant_rabbit.fast_bot_historical_s5 import (  # noqa: E402
    HistoricalS5SliceRequest,
    build_historical_s5_manifest,
    load_historical_s5_slices,
)
from quant_rabbit.fast_bot_profit_candidate_audit import (  # noqa: E402
    CANDIDATE_ADMISSION_THRESHOLDS,
    canonical_sha,
)
from quant_rabbit.fast_bot_technical_grid_backtest import (  # noqa: E402
    freeze_fast_bot_technical_grid_signal_v1,
    resolve_executable_bidask_time_close_v1,
    technical_grid_arms_v1,
)
from quant_rabbit.fast_bot_truth import _fast_bot_signal_valid  # noqa: E402
from quant_rabbit.instruments import instrument_pip_factor  # noqa: E402


AUDIT_CONTRACT = "QR_FAST_BOT_DIRECTION_TIME_CLOSE_AUDIT_V1"
ENTRY_ARMS = (
    "PASSIVE_NEAR_SIDE",
    "PASSIVE_QUARTER_SPREAD",
    "PASSIVE_MID_SPREAD",
    "PASSIVE_THREE_QUARTER_SPREAD",
)
ORIENTATIONS = ("DIRECT", "INVERSE")
HYPOTHESIS_BY_METHOD = {
    "TREND_CONTINUATION": "H01",
    "RANGE_ROTATION": "H03",
    "BREAKOUT_FAILURE": "H05",
}
LANE_FIELDS = ("pair", "side", "method", "horizon_lane")


def _parse_utc(value: Any) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _activation_clock(signal: Mapping[str, Any]) -> datetime:
    closed = _parse_utc(signal.get("m1_closed_candle_utc"))
    generated = _parse_utc(signal.get("generated_at_utc"))
    quote = _parse_utc(signal.get("quote_timestamp_utc"))
    causal_floor = max(closed + timedelta(minutes=1), generated, quote)
    floored = causal_floor.replace(second=0, microsecond=0)
    return floored if floored == causal_floor else floored + timedelta(minutes=1)


def _opposite(side: str) -> str:
    if side == "LONG":
        return "SHORT"
    if side == "SHORT":
        return "LONG"
    raise ValueError("raw signal side is invalid")


def _entry_for_side(
    signal: Mapping[str, Any], *, side: str, entry_arm: str
) -> float:
    arms = signal.get("entry_experiment_arms")
    if not isinstance(arms, list):
        raise ValueError("raw signal entry experiment is missing")
    matched = [
        row
        for row in arms
        if isinstance(row, Mapping) and row.get("arm_id") == entry_arm
    ]
    if len(matched) != 1:
        raise ValueError("raw signal entry arm is missing or duplicated")
    fraction = float(matched[0]["spread_fraction_toward_market"])
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("entry spread fraction is invalid")
    bid = Decimal(str(signal["quote_bid"]))
    ask = Decimal(str(signal["quote_ask"]))
    if not bid < ask:
        raise ValueError("raw signal quote is crossed")
    width = ask - bid
    value = bid + width * Decimal(str(fraction)) if side == "LONG" else ask - width * Decimal(str(fraction))
    tick = Decimal(1) / Decimal(instrument_pip_factor(str(signal["pair"])) * 10)
    return float((value / tick).quantize(Decimal(1), rounding=ROUND_HALF_UP) * tick)


def _base_geometry(
    signal: Mapping[str, Any], *, side: str, entry: float
) -> tuple[float, float]:
    factor = Decimal(instrument_pip_factor(str(signal["pair"])))
    tick = Decimal(1) / Decimal(factor * 10)
    entry_value = Decimal(str(entry))
    tp = Decimal(str(signal["take_profit_pips"])) / factor
    sl = Decimal(str(signal["stop_loss_pips"])) / factor
    if tp <= 0 or sl <= 0:
        raise ValueError("raw signal geometry is not positive")
    target = entry_value + tp if side == "LONG" else entry_value - tp
    stop = entry_value - sl if side == "LONG" else entry_value + sl
    target = (target / tick).quantize(Decimal(1), rounding=ROUND_HALF_UP) * tick
    stop = (stop / tick).quantize(Decimal(1), rounding=ROUND_HALF_UP) * tick
    return float(target), float(stop)


def _load_unique_signals(state_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_sha: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for path in sorted(state_root.glob("*/ledgers/fast_bot_shadow_ledger.jsonl")):
        payload = path.read_bytes() if path.exists() else b""
        sources.append(
            {
                "path": str(path.relative_to(state_root)),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
        for number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not _fast_bot_signal_valid(row):
                raise ValueError(f"invalid fast-bot signal at {path}:{number}")
            sha = str(row["signal_sha256"])
            prior = by_sha.get(sha)
            if prior is not None and prior != row:
                raise ValueError("conflicting raw signal SHA rows")
            by_sha[sha] = row
    signals = sorted(
        by_sha.values(),
        key=lambda row: (str(row["generated_at_utc"]), str(row["signal_sha256"])),
    )
    return signals, sources


def _load_truth(
    truth_root: Path,
    *,
    run_id: str,
    pairs: Sequence[str],
) -> tuple[dict[str, tuple[Any, ...]], dict[str, list[datetime]], dict[str, str], dict[str, Any]]:
    manifest = build_historical_s5_manifest(
        truth_root,
        pairs=tuple(pairs),
        allowed_run_ids=(run_id,),
        scan_workers=2,
    )
    if manifest.get("complete_pair_coverage") is not True:
        raise ValueError("truth manifest does not cover every signal pair")
    if manifest.get("all_selected_sources_acquisition_receipted") is not True:
        raise ValueError("truth manifest is not acquisition-receipted")
    time_from = _parse_utc(manifest["common_declared_from_utc"])
    time_to = _parse_utc(manifest["common_declared_to_utc"])
    requests = tuple(
        HistoricalS5SliceRequest(pair=pair, time_from=time_from, time_to=time_to)
        for pair in pairs
    )
    slices = load_historical_s5_slices(manifest, requests=requests)
    candles = {item.pair: item.candles for item in slices}
    timestamps = {
        pair: [candle.timestamp_utc for candle in rows]
        for pair, rows in candles.items()
    }
    receipts = {
        str(row["pair"]): str(row["acquisition_receipt_sha256"])
        for row in manifest["selected_sources"]
    }
    return candles, timestamps, receipts, manifest


def _pessimistic_expectancy(values: Sequence[float]) -> float | None:
    if not values:
        return None
    wins = [value for value in values if value > 0.0]
    losses = [-value for value in values if value < 0.0]
    z = 1.96
    total = len(values)
    observed = len(wins) / total
    denominator = 1.0 + z * z / total
    center = observed + z * z / (2.0 * total)
    margin = z * math.sqrt(
        (observed * (1.0 - observed) + z * z / (4.0 * total)) / total
    )
    lower = max(0.0, (center - margin) / denominator)
    average_win = sum(wins) / len(wins) if wins else 0.0
    average_loss = sum(losses) / len(losses) if losses else 0.0
    return lower * average_win - (1.0 - lower) * average_loss


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    resolved = [row for row in rows if row.get("result_available") is True]
    filled = [row for row in resolved if row.get("filled") is True]
    values = [float(row["realized_pips"]) for row in filled]
    wins = [value for value in values if value > 0.0]
    losses = [-value for value in values if value < 0.0]
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    profit_factor: float | str | None
    if gross_loss > 0.0:
        profit_factor = round(gross_profit / gross_loss, 6)
    elif gross_profit > 0.0:
        profit_factor = "INF"
    else:
        profit_factor = None
    daily: defaultdict[str, list[float]] = defaultdict(list)
    for row in filled:
        daily[str(row["source_generated_at_utc"])[:10]].append(
            float(row["realized_pips"])
        )
    pessimistic = _pessimistic_expectancy(values)
    return {
        "source_signals": len(rows),
        "resolved_signals": len(resolved),
        "filled_signals": len(filled),
        "unfilled_signals": len(resolved) - len(filled),
        "active_days": len(daily),
        "wins": len(wins),
        "losses": len(losses),
        "net_pips": round(sum(values), 6),
        "mean_pips_per_fill": round(sum(values) / len(values), 6) if values else None,
        "profit_factor": profit_factor,
        "pessimistic_expectancy_pips": round(pessimistic, 6) if pessimistic is not None else None,
        "positive_day_rate": round(sum(sum(items) > 0.0 for items in daily.values()) / len(daily), 6) if daily else 0.0,
        "maximum_daily_sample_share": round(max((len(items) for items in daily.values()), default=0) / len(values), 6) if values else 1.0,
        "daily": [
            {"date": day, "filled_signals": len(items), "net_pips": round(sum(items), 6)}
            for day, items in sorted(daily.items())
        ],
        "spread_included": True,
        "exit_policy": "EXECUTABLE_TIME_CLOSE",
    }


def _admission_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    thresholds = CANDIDATE_ADMISSION_THRESHOLDS
    pf = math.inf if metrics.get("profit_factor") == "INF" else float(metrics.get("profit_factor") or 0.0)
    pessimistic = metrics.get("pessimistic_expectancy_pips")
    daily = metrics.get("daily") if isinstance(metrics.get("daily"), list) else []
    return {
        "minimum_samples": int(metrics.get("filled_signals") or 0) >= int(thresholds["minimum_samples"]),
        "minimum_active_days": int(metrics.get("active_days") or 0) >= int(thresholds["minimum_active_days"]),
        "minimum_profit_factor": pf >= float(thresholds["minimum_profit_factor"]),
        "minimum_pessimistic_expectancy_pips": pessimistic is not None and float(pessimistic) > float(thresholds["minimum_pessimistic_expectancy_pips"]),
        "minimum_positive_day_rate": float(metrics.get("positive_day_rate") or 0.0) >= float(thresholds["minimum_positive_day_rate"]),
        "maximum_daily_sample_share": float(metrics.get("maximum_daily_sample_share") or 1.0) <= float(thresholds["maximum_daily_sample_share"]),
        "every_active_day_positive": bool(daily) and all(float(row.get("net_pips") or 0.0) > 0.0 for row in daily if isinstance(row, Mapping)),
    }


def _profit_factor_value(metrics: Mapping[str, Any]) -> float:
    value = metrics.get("profit_factor")
    return math.inf if value == "INF" else float(value or 0.0)


def _performance_rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = row["metrics"]
    return (
        bool(row["admission_passed"]),
        _profit_factor_value(metrics),
        float(metrics["net_pips"]),
        int(metrics["filled_signals"]),
        str(row["candidate_id"]),
    )


def audit(
    *,
    state_root: Path,
    truth_root: Path,
    truth_run_id: str,
    arm_ids: Sequence[str],
    generated_at_utc: datetime,
) -> dict[str, Any]:
    signals, signal_sources = _load_unique_signals(state_root)
    if not signals:
        raise ValueError("no raw fast-bot signals found")
    pairs = sorted({str(row["pair"]) for row in signals})
    candles, timestamps, receipts, manifest = _load_truth(
        truth_root, run_id=truth_run_id, pairs=pairs
    )
    available_arms = {arm.arm_id: arm for arm in technical_grid_arms_v1()}
    requested_arms = tuple(str(item).upper() for item in arm_ids)
    if not requested_arms or len(set(requested_arms)) != len(requested_arms):
        raise ValueError("grid arm ids must be non-empty and unique")
    if set(requested_arms) - set(available_arms):
        raise ValueError("grid arm id is outside the frozen technical grid")

    grouped: defaultdict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    occupied_until: dict[tuple[str, ...], datetime] = {}
    deoverlap_rejections = 0
    unresolved = 0
    for raw in signals:
        method = str(raw["method"])
        hypothesis_id = HYPOTHESIS_BY_METHOD.get(method)
        if hypothesis_id is None:
            raise ValueError(f"raw method has no frozen hypothesis mapping: {method}")
        source_side = str(raw["side"])
        activation = _activation_clock(raw)
        lane = tuple(str(raw.get(field) or "") for field in LANE_FIELDS)
        pair = str(raw["pair"])
        for orientation in ORIENTATIONS:
            actual_side = source_side if orientation == "DIRECT" else _opposite(source_side)
            for entry_arm in ENTRY_ARMS:
                entry = _entry_for_side(raw, side=actual_side, entry_arm=entry_arm)
                base_tp, base_sl = _base_geometry(raw, side=actual_side, entry=entry)
                for grid_arm_id in requested_arms:
                    grid_arm = available_arms[grid_arm_id]
                    key = (*lane, orientation, entry_arm, grid_arm_id)
                    latest_maturity = activation + timedelta(
                        seconds=grid_arm.entry_ttl_seconds + grid_arm.max_hold_seconds
                    )
                    if activation < occupied_until.get(key, activation):
                        deoverlap_rejections += 1
                        continue
                    occupied_until[key] = latest_maturity
                    frozen = freeze_fast_bot_technical_grid_signal_v1(
                        pair=pair,
                        hypothesis_id=hypothesis_id,
                        orientation=orientation,
                        source_predicted_side=source_side,
                        side=actual_side,
                        arm_id=grid_arm_id,
                        order_type="LIMIT",
                        activation_at_utc=activation,
                        entry_price=entry,
                        base_take_profit_price=base_tp,
                        base_stop_loss_price=base_sl,
                        causal_source_sha256=str(raw["signal_sha256"]),
                    )
                    pair_times = timestamps[pair]
                    start = bisect.bisect_left(pair_times, activation)
                    end = bisect.bisect_right(
                        pair_times, latest_maturity + timedelta(seconds=5)
                    )
                    outcome = resolve_executable_bidask_time_close_v1(
                        frozen,
                        candles[pair][start:end],
                        truth_source_receipt_sha256=receipts[pair],
                    )
                    row = {
                        "source_signal_sha256": raw["signal_sha256"],
                        "source_generated_at_utc": raw["generated_at_utc"],
                        "pair": pair,
                        "activation_at_utc": activation.isoformat(),
                        "latest_maturity_at_utc": latest_maturity.isoformat(),
                        "grid_signal_sha256": frozen["signal_sha256"],
                        "result_available": outcome["scorecard_result_available"],
                        "filled": outcome["filled"],
                        "realized_pips": outcome["post_cost_realized_pips"],
                        "exit_reason": outcome["exit_reason"],
                    }
                    grouped[key].append(row)
                    if outcome["scorecard_result_available"] is not True:
                        unresolved += 1

    candidates: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        lane = dict(zip(LANE_FIELDS, key[: len(LANE_FIELDS)]))
        orientation, entry_arm, grid_arm_id = key[len(LANE_FIELDS) :]
        metrics = _metrics(rows)
        checks = _admission_checks(metrics)
        candidates.append(
            {
                "candidate_id": ":".join(key),
                "lane": lane,
                "orientation": orientation,
                "entry_arm": entry_arm,
                "grid_arm_id": grid_arm_id,
                "metrics": metrics,
                "admission_checks": checks,
                "admission_passed": all(checks.values()),
            }
        )
    for orientation in ORIENTATIONS:
        for entry_arm in ENTRY_ARMS:
            for grid_arm_id in requested_arms:
                portfolio_rows = [
                    row
                    for key, items in grouped.items()
                    if key[-3:] == (orientation, entry_arm, grid_arm_id)
                    for row in items
                ]
                selected_rows: list[dict[str, Any]] = []
                portfolio_occupied: dict[str, datetime] = {}
                for row in sorted(
                    portfolio_rows,
                    key=lambda item: (
                        str(item["activation_at_utc"]),
                        str(item["pair"]),
                        str(item["source_signal_sha256"]),
                    ),
                ):
                    pair = str(row["pair"])
                    activation = _parse_utc(row["activation_at_utc"])
                    if activation < portfolio_occupied.get(pair, activation):
                        continue
                    portfolio_occupied[pair] = _parse_utc(
                        row["latest_maturity_at_utc"]
                    )
                    selected_rows.append(row)
                metrics = _metrics(selected_rows)
                checks = _admission_checks(metrics)
                candidates.append(
                    {
                        "candidate_id": ":".join(
                            (
                                "MULTI_PAIR_PORTFOLIO",
                                orientation,
                                entry_arm,
                                grid_arm_id,
                            )
                        ),
                        "lane": {
                            "pair": "MULTI_PAIR_PORTFOLIO",
                            "side": "SOURCE_RELATIVE",
                            "method": "ALL_FROZEN_RAW_METHODS",
                            "horizon_lane": "PAIR_SERIALIZED",
                        },
                        "orientation": orientation,
                        "entry_arm": entry_arm,
                        "grid_arm_id": grid_arm_id,
                        "metrics": metrics,
                        "admission_checks": checks,
                        "admission_passed": all(checks.values()),
                    }
                )
    ranked = sorted(candidates, key=_performance_rank, reverse=True)
    leads = [row for row in ranked if row["admission_passed"]]
    minimum_samples = int(CANDIDATE_ADMISSION_THRESHOLDS["minimum_samples"])
    minimum_active_days = int(
        CANDIDATE_ADMISSION_THRESHOLDS["minimum_active_days"]
    )
    minimum_evidence = sorted(
        (
            row
            for row in candidates
            if int(row["metrics"]["filled_signals"]) >= minimum_samples
            and int(row["metrics"]["active_days"]) >= minimum_active_days
        ),
        key=_performance_rank,
        reverse=True,
    )
    three_day = sorted(
        (
            row
            for row in candidates
            if int(row["metrics"]["active_days"]) >= minimum_active_days
        ),
        key=_performance_rank,
        reverse=True,
    )
    portfolios = sorted(
        (
            row
            for row in candidates
            if row["lane"]["pair"] == "MULTI_PAIR_PORTFOLIO"
        ),
        key=_performance_rank,
        reverse=True,
    )
    body = {
        "contract": AUDIT_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": generated_at_utc.astimezone(timezone.utc).isoformat(),
        "status": "RESEARCH_LEADS_REQUIRE_FUTURE_PRECOMMITMENT" if leads else "NO_ADMISSIBLE_DIRECTION_TIME_CLOSE_CANDIDATE",
        "source_state_root": str(state_root.resolve()),
        "source_signal_count": len(signals),
        "source_signal_files": signal_sources,
        "source_signal_bundle_sha256": canonical_sha(signal_sources),
        "truth_manifest_sha256": manifest["manifest_sha256"],
        "truth_run_id": truth_run_id,
        "truth_source_files": manifest["selected_sources"],
        "truth_acquisition_receipted": True,
        "candidate_universe": {
            "policy": "EXACT_LANE_X_DIRECT_INVERSE_X_PASSIVE_ENTRY_X_FROZEN_OFAT_ARM_V1",
            "orientations": list(ORIENTATIONS),
            "entry_arms": list(ENTRY_ARMS),
            "grid_arm_ids": list(requested_arms),
            "candidate_count": len(candidates),
            "same_candidate_overlap_policy": "HALF_OPEN_ACTIVATION_TO_MAX_MATURITY",
            "multiple_testing_corrected": False,
        },
        "candidate_admission_thresholds": {
            **CANDIDATE_ADMISSION_THRESHOLDS,
            "every_active_day_positive": True,
        },
        "deoverlap_rejected_signal_variants": deoverlap_rejections,
        "unresolved_signal_variants": unresolved,
        "research_lead_count": len(leads),
        "research_leads": leads,
        "top_observed_candidates": ranked[:40],
        "top_minimum_evidence_candidates": minimum_evidence[:40],
        "top_three_day_candidates": three_day[:40],
        "portfolio_candidates": portfolios,
        "all_candidate_results": ranked,
        "profitability_claim": "RETROSPECTIVE_RESEARCH_ONLY_NOT_FORWARD_PROOF",
        "historical_rows_admitted_to_forward_scorecard": False,
        "automatic_candidate_activation_allowed": False,
        "automatic_adoption_allowed": False,
        "execution_authority": "NONE",
        "broker_http_methods_used": [],
        "broker_mutation_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "promotion_allowed": False,
        "live_permission": False,
        "shadow_only": True,
    }
    return {**body, "contract_sha256": canonical_sha(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--truth-run-id", required=True)
    parser.add_argument("--grid-arms", default="BASE")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        state_root=args.state_root,
        truth_root=args.truth_root,
        truth_run_id=args.truth_run_id,
        arm_ids=tuple(part.strip() for part in args.grid_arms.split(",") if part.strip()),
        generated_at_utc=datetime.now(timezone.utc),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_count": result["candidate_universe"]["candidate_count"],
                "research_lead_count": result["research_lead_count"],
                "top_observed_candidates": result["top_observed_candidates"][:5],
                "output": str(args.output),
                "contract_sha256": result["contract_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
