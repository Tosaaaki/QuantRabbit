#!/usr/bin/env python3
"""Read-only, signal-joined geometry counterfactuals for resident fast-bot shadow.

The analyzer re-fetches the exact OANDA S5 BID/ASK intervals already sealed in
the outcome ledger and requires every truth-chunk hash to match.  It never
changes runtime parameters, inventory state, or broker state.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.broker.oanda import OandaReadOnlyClient  # noqa: E402
from quant_rabbit.instruments import instrument_pip_factor  # noqa: E402
from quant_rabbit.technical_forecast_forward_outcome import S5BidAskCandle  # noqa: E402
from quant_rabbit.technical_forecast_forward_truth import fetch_frozen_s5_truth  # noqa: E402


CONTRACT = "QR_FAST_BOT_GEOMETRY_COUNTERFACTUAL_V1"
POLICY_BASELINE = "EMITTED_BASELINE"
POLICY_FIXED = "FIXED_SL_3P2_TP_2P4"
POLICY_ATR_1P0 = "ATR_SL_1P0_TP_0P75"
POLICY_ATR_1P2 = "ATR_SL_1P2_TP_0P90"
POLICY_VETO_1P0 = "ENTRY_VETO_EMITTED_SL_LT_1P0_ATR"
POLICY_VETO_1P2 = "ENTRY_VETO_EMITTED_SL_LT_1P2_ATR"
POLICY_LOT_HALF = "LOT_0P50_BASELINE_GEOMETRY"
POLICY_LOT_QUARTER = "LOT_0P25_BASELINE_GEOMETRY"
STOP_REASONS = {
    "STOP_LOSS",
    "STOP_LOSS_GAP",
    "STOP_LOSS_AMBIGUOUS_FILL_S5",
    "STOP_LOSS_GAP_AMBIGUOUS_FILL_S5",
    "STOP_LOSS_AMBIGUOUS_SAME_S5",
    "STOP_LOSS_GAP_AMBIGUOUS_SAME_S5",
}


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("aware UTC timestamp is required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(raw)


def _load_jsonl_snapshot(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}: line {line_number} is not an object")
        rows.append(value)
    return rows, _sha256_bytes(raw)


def join_filled_signals(
    signals: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    duplicate_signal_ids = 0
    for signal in signals:
        signal_id = str(signal.get("signal_id") or "")
        if not signal_id:
            continue
        if signal_id in by_id:
            duplicate_signal_ids += 1
            continue
        by_id[signal_id] = signal
    joined: list[dict[str, Any]] = []
    unmatched = 0
    duplicate_outcome_ids = 0
    seen_outcomes: set[str] = set()
    for outcome in outcomes:
        signal_id = str(outcome.get("signal_id") or "")
        if not signal_id:
            continue
        if signal_id in seen_outcomes:
            duplicate_outcome_ids += 1
            continue
        seen_outcomes.add(signal_id)
        signal = by_id.get(signal_id)
        if signal is None or signal.get("signal_sha256") != outcome.get("signal_sha256"):
            unmatched += 1
            continue
        if outcome.get("filled") is True:
            joined.append({"signal": dict(signal), "outcome": dict(outcome)})
    joined.sort(key=lambda row: (str(row["signal"].get("generated_at_utc")), str(row["signal"].get("signal_id"))))
    return joined, {
        "signal_rows": len(signals),
        "outcome_rows": len(outcomes),
        "filled_joined_rows": len(joined),
        "duplicate_signal_ids": duplicate_signal_ids,
        "duplicate_outcome_ids": duplicate_outcome_ids,
        "unmatched_outcomes": unmatched,
    }


def _price_geometry(signal: Mapping[str, Any], *, sl_pips: float, tp_pips: float) -> tuple[float, float]:
    pair = str(signal["pair"])
    side = str(signal["side"])
    entry = float(signal["entry"])
    factor = float(instrument_pip_factor(pair))
    if side == "LONG":
        return entry - sl_pips / factor, entry + tp_pips / factor
    if side == "SHORT":
        return entry + sl_pips / factor, entry - tp_pips / factor
    raise ValueError("side must be LONG or SHORT")


def candidate_specs(signal: Mapping[str, Any]) -> list[dict[str, Any]]:
    emitted_sl = float(signal["stop_loss_pips"])
    emitted_tp = float(signal["take_profit_pips"])
    atr = float(signal["m5_atr_pips"])
    return [
        {"policy": POLICY_BASELINE, "sl_pips": emitted_sl, "tp_pips": emitted_tp, "unit_weight": 1.0, "vetoed": False},
        {"policy": POLICY_FIXED, "sl_pips": 3.2, "tp_pips": 2.4, "unit_weight": 1.0, "vetoed": False},
        {
            "policy": POLICY_ATR_1P0,
            "sl_pips": round(atr, 6),
            "tp_pips": round(atr * 0.75, 6),
            "unit_weight": 1.0,
            "vetoed": False,
        },
        {
            "policy": POLICY_ATR_1P2,
            "sl_pips": round(atr * 1.2, 6),
            "tp_pips": round(atr * 0.90, 6),
            "unit_weight": 1.0,
            "vetoed": False,
        },
        {
            "policy": POLICY_VETO_1P0,
            "sl_pips": emitted_sl,
            "tp_pips": emitted_tp,
            "unit_weight": 1.0,
            "vetoed": emitted_sl < atr,
        },
        {
            "policy": POLICY_VETO_1P2,
            "sl_pips": emitted_sl,
            "tp_pips": emitted_tp,
            "unit_weight": 1.0,
            "vetoed": emitted_sl < atr * 1.2,
        },
        {"policy": POLICY_LOT_HALF, "sl_pips": emitted_sl, "tp_pips": emitted_tp, "unit_weight": 0.5, "vetoed": False},
        {"policy": POLICY_LOT_QUARTER, "sl_pips": emitted_sl, "tp_pips": emitted_tp, "unit_weight": 0.25, "vetoed": False},
    ]


def score_path(
    signal: Mapping[str, Any],
    candles: Sequence[S5BidAskCandle],
    *,
    sl_pips: float,
    tp_pips: float,
) -> dict[str, Any]:
    generated = _parse_utc(signal["generated_at_utc"])
    ttl = int(signal["entry_ttl_seconds"])
    hold = int(signal["max_hold_seconds"])
    fill_deadline = generated + timedelta(seconds=ttl)
    pair = str(signal["pair"])
    side = str(signal["side"])
    entry = float(signal["entry"])
    factor = float(instrument_pip_factor(pair))
    stop, target = _price_geometry(signal, sl_pips=sl_pips, tp_pips=tp_pips)
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
            tp_hit = candle.bid_h >= target
            sl_hit = candle.bid_l <= stop
        else:
            tp_hit = candle.ask_l <= target
            sl_hit = candle.ask_h >= stop
        if newly_filled and (tp_hit or sl_hit):
            ambiguous = True
            exit_reason = "STOP_LOSS_AMBIGUOUS_FILL_S5"
            realized = -sl_pips
            exit_at = candle.timestamp_utc
            break
        if tp_hit and sl_hit:
            ambiguous = True
            exit_reason = "STOP_LOSS_AMBIGUOUS_SAME_S5"
            realized = -sl_pips
            exit_at = candle.timestamp_utc
            break
        if sl_hit:
            opening = (
                (candle.bid_o - entry) * factor
                if side == "LONG"
                else (entry - candle.ask_o) * factor
            )
            realized = min(-sl_pips, opening)
            exit_reason = "STOP_LOSS_GAP" if realized < -sl_pips - 1e-9 else "STOP_LOSS"
            exit_at = candle.timestamp_utc
            break
        if tp_hit:
            exit_reason = "TAKE_PROFIT"
            realized = tp_pips
            exit_at = candle.timestamp_utc
            break
    if fill_at is not None and exit_at is None:
        exit_reason = "HORIZON_FULL_STOP_LOSS"
        realized = -sl_pips
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
    time_to_stop = (
        (exit_at - fill_at).total_seconds()
        if fill_at is not None and exit_at is not None and exit_reason in STOP_REASONS
        else None
    )
    return {
        "filled": fill_at is not None,
        "fill_at_utc": fill_at.isoformat() if fill_at else None,
        "exit_at_utc": exit_at.isoformat() if exit_at else None,
        "exit_reason": exit_reason,
        "realized_pips": round(realized, 6),
        "mfe_pips": round(max(0.0, mfe), 6),
        "mae_pips": round(max(0.0, mae), 6),
        "time_to_stop_seconds": time_to_stop,
        "ambiguous_same_s5": ambiguous,
        "s5_extrema_include_fill_bar": True,
    }


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
        return "REGIME_SCORE_UNKNOWN"
    if score < 0:
        return "REGIME_SCORE_NEGATIVE"
    if score > 0:
        return "REGIME_SCORE_POSITIVE"
    return "REGIME_SCORE_ZERO"


def _attributes(signal: Mapping[str, Any]) -> dict[str, Any]:
    atr = float(signal["m5_atr_pips"])
    spread = float(signal["spread_pips"])
    return {
        "pair": str(signal["pair"]),
        "strategy": str(signal["method"]),
        "atr_bucket": _atr_bucket(atr),
        "m5_atr_pips": round(atr, 6),
        "spread_bucket": f"{round(spread, 1):.1f}P",
        "spread_pips": round(spread, 6),
        "side": str(signal["side"]),
        "regime_bucket": _regime_bucket(signal.get("regime_score")),
        "regime_score": signal.get("regime_score"),
    }


def _mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return round(statistics.fmean(rows), 6) if rows else None


def _median(values: Iterable[float]) -> float | None:
    rows = list(values)
    return round(statistics.median(rows), 6) if rows else None


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("vetoed") is not True]
    filled = [row for row in eligible if row.get("filled") is True]
    values = [float(row["realized_pips"]) for row in filled]
    weighted = [float(row["weighted_pips"]) for row in filled]
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    gross_loss = abs(sum(losses))
    return {
        "signal_count": len(rows),
        "eligible_count": len(eligible),
        "vetoed_count": len(rows) - len(eligible),
        "filled_count": len(filled),
        "wins": len(wins),
        "losses": len(losses),
        "net_pips": round(sum(values), 6),
        "weighted_net_pips": round(sum(weighted), 6),
        "mean_pips_per_fill": _mean(values),
        "profit_factor": (
            round(sum(wins) / gross_loss, 6)
            if gross_loss > 0
            else "INF" if wins else None
        ),
        "stop_hit_count": sum(str(row.get("exit_reason")) in STOP_REASONS for row in filled),
        "horizon_full_stop_loss_count": sum(row.get("exit_reason") == "HORIZON_FULL_STOP_LOSS" for row in filled),
        "mean_time_to_stop_seconds": _mean(float(row["time_to_stop_seconds"]) for row in filled if row.get("time_to_stop_seconds") is not None),
        "median_time_to_stop_seconds": _median(float(row["time_to_stop_seconds"]) for row in filled if row.get("time_to_stop_seconds") is not None),
        "mean_mfe_pips": _mean(float(row["mfe_pips"]) for row in filled),
        "mean_mae_pips": _mean(float(row["mae_pips"]) for row in filled),
    }


def _group(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in keys)
        groups.setdefault(key, []).append(row)
    return [
        {**dict(zip(keys, key)), **aggregate(group)}
        for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0]))
    ]


def analyze(
    *,
    signals: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    truth_fetcher: Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[Sequence[S5BidAskCandle], Sequence[str]]],
    generated_at_utc: datetime,
    signal_ledger_sha256: str,
    outcome_ledger_sha256: str,
    release_manifest: Mapping[str, Any],
    runtime_status: Mapping[str, Any],
) -> dict[str, Any]:
    joined, join_counts = join_filled_signals(signals, outcomes)
    per_signal: list[dict[str, Any]] = []
    truth_hash_matches = 0
    for row in joined:
        signal = row["signal"]
        outcome = row["outcome"]
        candles, hashes = truth_fetcher(signal, outcome)
        expected_hashes = [str(value) for value in outcome.get("truth_chunk_sha256", [])]
        if list(hashes) != expected_hashes:
            raise ValueError(f"truth chunk hash mismatch for signal {signal['signal_id']}")
        truth_hash_matches += 1
        attrs = _attributes(signal)
        for spec in candidate_specs(signal):
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
                    "s5_extrema_include_fill_bar": True,
                }
            else:
                scored = score_path(signal, candles, sl_pips=float(spec["sl_pips"]), tp_pips=float(spec["tp_pips"]))
            item = {
                "signal_id": signal["signal_id"],
                "signal_sha256": signal["signal_sha256"],
                **attrs,
                **spec,
                **scored,
                "weighted_pips": round(float(scored["realized_pips"]) * float(spec["unit_weight"]), 6),
            }
            per_signal.append(item)
            if spec["policy"] == POLICY_BASELINE and (
                scored["filled"] is not True
                or scored["exit_reason"] != outcome.get("exit_reason")
                or not math.isclose(float(scored["realized_pips"]), float(outcome["realized_pips"]), abs_tol=1e-6)
            ):
                raise ValueError(f"baseline replay mismatch for signal {signal['signal_id']}")
    policy_order = [
        POLICY_BASELINE,
        POLICY_FIXED,
        POLICY_ATR_1P0,
        POLICY_ATR_1P2,
        POLICY_VETO_1P0,
        POLICY_VETO_1P2,
        POLICY_LOT_HALF,
        POLICY_LOT_QUARTER,
    ]
    comparison = [
        {"policy": policy, **aggregate([row for row in per_signal if row["policy"] == policy])}
        for policy in policy_order
    ]
    baseline = [row for row in per_signal if row["policy"] == POLICY_BASELINE]
    requested_signal_ids = {
        str(row["signal_id"])
        for row in baseline
        if row["pair"] == "EUR_USD"
        and row["strategy"] == "RANGE_ROTATION"
        and 2.8 <= float(row["m5_atr_pips"]) < 4.0
        and math.isclose(float(row["sl_pips"]), 3.2, abs_tol=1e-9)
        and math.isclose(float(row["tp_pips"]), 2.4, abs_tol=1e-9)
    }
    body = {
        "contract": CONTRACT,
        "schema_version": 1,
        "generated_at_utc": generated_at_utc.astimezone(timezone.utc).isoformat(),
        "analysis_scope": "FILLED_PRIMARY_OUTCOMES_JOINED_BY_SIGNAL_ID_AND_SIGNAL_SHA256",
        "signal_ledger_sha256": signal_ledger_sha256,
        "outcome_ledger_sha256": outcome_ledger_sha256,
        "join_counts": join_counts,
        "truth_hash_matches": truth_hash_matches,
        "truth_hash_mismatches": 0,
        "counterfactual_comparison": comparison,
        "requested_eurusd_range_fixed_geometry_cohort": {
            "definition": "EUR_USD_RANGE_ROTATION_M5_ATR_2P8_TO_LT_4P0_EMITTED_SL_3P2_TP_2P4",
            "signal_ids": sorted(requested_signal_ids),
            "comparison": [
                {
                    "policy": policy,
                    **aggregate(
                        [
                            row
                            for row in per_signal
                            if row["policy"] == policy
                            and str(row["signal_id"]) in requested_signal_ids
                        ]
                    ),
                }
                for policy in policy_order
            ],
        },
        "baseline_combined_breakdown": _group(
            baseline,
            ["pair", "strategy", "atr_bucket", "spread_bucket", "side", "regime_bucket"],
        ),
        "baseline_marginal_breakdowns": {
            key: _group(baseline, [key])
            for key in ("pair", "strategy", "atr_bucket", "spread_bucket", "side", "regime_bucket")
        },
        "per_signal_counterfactuals": per_signal,
        "limitations": [
            "OBSERVATIONAL_SINGLE_DAY_SMALL_SAMPLE_NO_CAUSAL_VOLATILITY_CLAIM",
            "S5_OHLC_EXTREMA_INCLUDE_THE_FILL_BAR_AND_CANNOT_ORDER_INTRABAR_MFE_MAE",
            "REGIME_BUCKET_IS_RECORDED_REGIME_SCORE_SIGN_NOT_A_RECONSTRUCTED_REGIME_LABEL",
            "LOT_POLICIES_SCALE_EXPOSURE_WEIGHT_ONLY_AND_DO_NOT_CHANGE_PRICE_PATH_PIPS",
            "INVENTORY_CONTROLLER_IS_OUT_OF_SCOPE_AND_NOT_EVALUATED",
        ],
        "authority": {
            "execution_authority": "NONE",
            "broker_http_methods_used": ["GET"],
            "broker_mutation": False,
            "external_order_attempts": int(runtime_status.get("external_order_attempts") or 0),
            "external_orders": int(runtime_status.get("external_orders") or 0),
            "live_permission": False,
            "promotion_allowed": False,
            "automatic_parameter_change_allowed": False,
            "inventory_control_evaluated": False,
            "manual_tagless_positions_policy": "NO_TOUCH",
            "existing_tp_sl_policy": "NO_TOUCH",
        },
        "runtime_release": {
            "source_commit": release_manifest.get("commit"),
            "source_bundle_sha256": release_manifest.get("source_bundle_sha256"),
            "resident_pid": runtime_status.get("pid"),
        },
    }
    body["contract_sha256"] = _canonical_sha(body)
    return body


def write_report(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# Fast-bot Shadow Geometry Counterfactual",
        "",
        f"- Contract: `{result['contract']}`",
        f"- Snapshot: signal `{result['signal_ledger_sha256']}`, outcome `{result['outcome_ledger_sha256']}`",
        f"- Filled joins: `{result['join_counts']['filled_joined_rows']}`; exact truth hash matches: `{result['truth_hash_matches']}`",
        "- Authority: `NONE`; OANDA methods: `GET`; broker mutation/order: `0`",
        "- Runtime parameter changes: `false`; inventory controller evaluated: `false`",
        "",
        "## Same-signal counterfactual",
        "",
        "| Policy | Eligible | Veto | W/L | Net pips | Weighted pips | PF | Stop hits | Horizon full SL | Mean SL sec | MFE | MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["counterfactual_comparison"]:
        lines.append(
            f"| {row['policy']} | {row['eligible_count']} | {row['vetoed_count']} | {row['wins']}/{row['losses']} | "
            f"{row['net_pips']} | {row['weighted_net_pips']} | {row['profit_factor']} | {row['stop_hit_count']} | "
            f"{row['horizon_full_stop_loss_count']} | {row['mean_time_to_stop_seconds']} | {row['mean_mfe_pips']} | {row['mean_mae_pips']} |"
        )
    lines.extend([
        "",
        "## Pair × strategy × ATR × spread × direction × regime-score sign",
        "",
        "| Pair | Strategy | ATR bucket | Spread | Side | Regime | N | W/L | Net | Stop | Horizon SL | Mean SL sec | MFE | MAE |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in result["baseline_combined_breakdown"]:
        lines.append(
            f"| {row['pair']} | {row['strategy']} | {row['atr_bucket']} | {row['spread_bucket']} | {row['side']} | "
            f"{row['regime_bucket']} | {row['filled_count']} | {row['wins']}/{row['losses']} | {row['net_pips']} | "
            f"{row['stop_hit_count']} | {row['horizon_full_stop_loss_count']} | {row['mean_time_to_stop_seconds']} | "
            f"{row['mean_mfe_pips']} | {row['mean_mae_pips']} |"
        )
    lines.extend([
        "",
        "## Requested EUR/USD RANGE_ROTATION fixed-geometry cohort",
        "",
        f"Definition: `{result['requested_eurusd_range_fixed_geometry_cohort']['definition']}`",
        "",
        "| Policy | Eligible | Veto | W/L | Net pips | Weighted pips | Stop hits | Horizon full SL | MFE | MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in result["requested_eurusd_range_fixed_geometry_cohort"]["comparison"]:
        lines.append(
            f"| {row['policy']} | {row['eligible_count']} | {row['vetoed_count']} | {row['wins']}/{row['losses']} | "
            f"{row['net_pips']} | {row['weighted_net_pips']} | {row['stop_hit_count']} | "
            f"{row['horizon_full_stop_loss_count']} | {row['mean_mfe_pips']} | {row['mean_mae_pips']} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "This is a single-day, small-sample observational split. It does not identify volatility as the cause. "
        "The ATR and veto rows are frozen counterfactuals only; no runtime parameter or inventory policy is changed.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-ledger", type=Path, required=True)
    parser.add_argument("--outcome-ledger", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--runtime-status", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 8:
        raise SystemExit("--max-workers must be inside 1..8")
    signals, signal_sha = _load_jsonl_snapshot(args.shadow_ledger)
    outcomes, outcome_sha = _load_jsonl_snapshot(args.outcome_ledger)
    release_manifest = json.loads(args.release_manifest.read_text())
    runtime_status = json.loads(args.runtime_status.read_text())
    if (
        release_manifest.get("execution_authority") != "NONE"
        or release_manifest.get("broker_http_methods_allowed") != ["GET"]
        or release_manifest.get("broker_mutation_allowed") is not False
        or runtime_status.get("external_order_attempts") != 0
        or runtime_status.get("external_orders") != 0
    ):
        raise SystemExit("resident shadow authority boundary is not zero-mutation GET-only")
    client = OandaReadOnlyClient()
    joined, _ = join_filled_signals(signals, outcomes)
    cache: dict[str, tuple[Sequence[S5BidAskCandle], Sequence[str]]] = {}

    def fetch(row: Mapping[str, Any]) -> tuple[str, tuple[Sequence[S5BidAskCandle], Sequence[str]]]:
        signal = row["signal"]
        outcome = row["outcome"]
        value = fetch_frozen_s5_truth(
            client,
            pair=str(signal["pair"]),
            time_from=_parse_utc(outcome["truth_request_from_utc"]),
            time_to=_parse_utc(outcome["truth_request_to_utc"]),
            chunk_candle_limit=4500,
        )
        return str(signal["signal_id"]), value

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for signal_id, value in executor.map(fetch, joined):
            cache[signal_id] = value

    result = analyze(
        signals=signals,
        outcomes=outcomes,
        truth_fetcher=lambda signal, _outcome: cache[str(signal["signal_id"])],
        generated_at_utc=datetime.now(timezone.utc),
        signal_ledger_sha256=signal_sha,
        outcome_ledger_sha256=outcome_sha,
        release_manifest=release_manifest,
        runtime_status=runtime_status,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_report(args.output_report, result)
    print(json.dumps({
        "status": "ANALYZED",
        "contract_sha256": result["contract_sha256"],
        "filled_joined_rows": result["join_counts"]["filled_joined_rows"],
        "truth_hash_matches": result["truth_hash_matches"],
        "broker_http_methods_used": ["GET"],
        "broker_mutation": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "automatic_parameter_change_allowed": False,
        "output_json": str(args.output_json),
        "output_report": str(args.output_report),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
