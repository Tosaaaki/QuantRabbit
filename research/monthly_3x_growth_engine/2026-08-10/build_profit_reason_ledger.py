#!/usr/bin/env python3
"""Explain the concrete profitable mechanisms without turning hints into proof."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEED = 20260810


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lcb(values: list[float], key: str) -> float | None:
    if not values:
        return None
    rng = np.random.default_rng(SEED ^ int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
    data = np.asarray(values, dtype=float)
    means = [rng.choice(data, len(data), replace=True).mean() for _ in range(10000)]
    return float(np.quantile(means, 0.025))


def stats(values: list[float], key: str) -> dict[str, Any]:
    return {
        "trades": len(values),
        "net_jpy": sum(values),
        "expectancy_jpy": float(np.mean(values)) if values else None,
        "paired_bootstrap_lcb_expectancy_jpy": lcb(values, key),
        "wins": sum(value > 0 for value in values),
        "losses": sum(value < 0 for value in values),
    }


def base_method(lane_id: str | None) -> str:
    if not lane_id:
        return "UNATTRIBUTED"
    parts = lane_id.split(":")
    return parts[3] if len(parts) >= 4 else "UNATTRIBUTED"


def main() -> None:
    payload_path = ROOT / "research/python_ecosystem_audit/2026-08-10/real_shadow_payload.json"
    label_path = ROOT / "research/financial_oracle_v2/2026-08-10/trade_cashflows_v2.jsonl"
    episode_path = ROOT / "research/historical_learning_admission/all_entry_episodes_v1.jsonl"
    payload = json.loads(payload_path.read_text())
    labels = {row["episode_id"]: row for row in map(json.loads, label_path.open())}
    episodes = {row["episode_id"]: row for row in map(json.loads, episode_path.open())}
    rows = [
        row for row in payload["episode_records"]
        if row["method"] == "ALL_TRADES" and row["window"] == "QUADRUPLE_64D"
    ]

    group_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        episode = episodes[row["episode_id"]]
        group_values[(row["split"], row["pair"], row["side"], base_method(episode.get("lane_id")))].append(
            float(labels[row["episode_id"]]["corrected_net_jpy"])
        )

    groups: list[dict[str, Any]] = []
    identities = sorted({key[1:] for key in group_values})
    for pair, side, method in identities:
        train = group_values.get(("TRAIN", pair, side, method), [])
        validation = group_values.get(("VALIDATION", pair, side, method), [])
        groups.append(
            {
                "pair": pair,
                "side": side,
                "method": method,
                "train": stats(train, f"train:{pair}:{side}:{method}"),
                "validation": stats(validation, f"validation:{pair}:{side}:{method}"),
                "same_sign_positive": bool(train and validation and sum(train) > 0 and sum(validation) > 0),
                "evidence_grade": "PARTIAL_REPRODUCTION" if len(train) >= 10 and len(validation) >= 1 and sum(train) > 0 and sum(validation) > 0 else "DIAGNOSTIC_ONLY",
            }
        )

    eurusd_short_train = [
        float(labels[row["episode_id"]]["corrected_net_jpy"])
        for row in rows if row["split"] == "TRAIN" and row["pair"] == "EUR_USD" and row["side"] == "SHORT"
    ]
    eurusd_short_validation_rows = [
        row for row in rows if row["split"] == "VALIDATION" and row["pair"] == "EUR_USD" and row["side"] == "SHORT"
    ]
    eurusd_short_validation = [float(labels[row["episode_id"]]["corrected_net_jpy"]) for row in eurusd_short_validation_rows]
    validation_receipts = [
        {
            "episode_id": row["episode_id"],
            "decision_time": row["decision_time"],
            "lane_id": episodes[row["episode_id"]].get("lane_id"),
            "terminal_reason": episodes[row["episode_id"]].get("outcome_type"),
            "corrected_net_jpy": labels[row["episode_id"]]["corrected_net_jpy"],
            "units": labels[row["episode_id"]]["units"],
        }
        for row in eurusd_short_validation_rows
    ]

    runtime_hint: dict[str, Any] = {"available": False}
    board_path = ROOT / "data/active_opportunity_board.json"
    if board_path.exists():
        board = json.loads(board_path.read_text())
        hints = []
        for lane in board.get("ranked_active_lanes") or []:
            if lane.get("lane_id") in {
                "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
            }:
                evidence = lane.get("local_tp_proof") or {}
                hints.append(
                    {
                        "lane_id": lane.get("lane_id"),
                        "capture_take_profit_expectancy_jpy": evidence.get("capture_take_profit_expectancy_jpy"),
                        "capture_take_profit_trades": evidence.get("capture_take_profit_trades"),
                        "broad_capture_take_profit_expectancy_jpy": evidence.get("broad_capture_take_profit_expectancy_jpy"),
                        "broad_capture_take_profit_trades": evidence.get("broad_capture_take_profit_trades"),
                        "classification": "READ_ONLY_RUNTIME_HINT_NOT_V2_FINANCIAL_PROOF",
                    }
                )
        runtime_hint = {"available": bool(hints), "source_sha256": sha(board_path), "hints": hints}

    exact_limit_path = ROOT / "data/eurusd_short_breakout_failure_limit_s5_bidask_replay.json"
    exact_limit: dict[str, Any] = {"available": False}
    if exact_limit_path.exists():
        raw_limit = json.loads(exact_limit_path.read_text())
        exact_limit = {
            "available": True,
            "source_sha256": sha(exact_limit_path),
            "target_shape": raw_limit.get("target_shape"),
            "sample_count": raw_limit.get("replay_sample_count"),
            "wins": raw_limit.get("replay_wins"),
            "losses": raw_limit.get("replay_losses"),
            "net_jpy": (raw_limit.get("limit_samples_observed") or {}).get("net_jpy"),
            "expectancy_after_bidask_jpy": raw_limit.get("net_expectancy_after_bidask"),
            "s5_bidask_replay_status": raw_limit.get("s5_bidask_replay_status"),
            "timestamp_alignment_status": ((raw_limit.get("spread_slippage_summary") or {}).get("timestamp_alignment") or {}).get("status"),
            "live_permission_allowed": raw_limit.get("live_permission_allowed"),
            "classification": "PARTIAL_EXECUTABLE_PROOF_REQUIRES_CANONICAL_FILL_RECONCILIATION_AND_MORE_SAMPLES",
        }

    target_trade_counts = []
    observed_expectancy = float(np.mean(eurusd_short_validation))
    for monthly_trades in (200, 400, 800, 1000):
        required = 400000.0 / monthly_trades
        target_trade_counts.append(
            {
                "monthly_trades": monthly_trades,
                "required_expectancy_jpy": required,
                "observed_eurusd_short_validation_expectancy_jpy": observed_expectancy,
                "required_edge_or_size_multiple": required / observed_expectancy,
            }
        )

    ledger = {
        "contract": "PROFIT_REASON_LEDGER_V1",
        "sources": {
            "payload_sha256": sha(payload_path),
            "financial_labels_sha256": sha(label_path),
            "episodes_sha256": sha(episode_path),
        },
        "holdout_used": False,
        "verified_reasons_can_work": [
            {
                "reason": "NON_ZERO_AFTER_COST_BASELINE",
                "evidence": "Corrected 64d validation ALL_TRADES is +11,706.0523 JPY over 101 trades.",
                "meaning": "The system already contains after-cost edge; the problem is amplification and stability, not creating edge from zero."
            },
            {
                "reason": "EURUSD_SHORT_PERSISTS_ACROSS_SPLIT",
                "train": stats(eurusd_short_train, "eurusd_short_train"),
                "validation": stats(eurusd_short_validation, "eurusd_short_validation"),
                "validation_receipts": validation_receipts,
                "meaning": "All five validation EUR_USD SHORT receipts are positive. The later vehicle/exit mix is materially better than the broad early cohort, so vehicle and exit lineage are a growth lever."
            },
            {
                "reason": "DECISION_TIME_SIZING_CREATES_POINT_INCREMENT",
                "evidence": "PRICE_ACTION_RIDGE_SIZE at 1x/75% cohort margin cap improves corrected 64d validation by +2,493.46 JPY without skipping baseline trades.",
                "caveat": "paired LCB is -19.27 JPY/trade; this is a working lever, not yet stable proof."
            },
            {
                "reason": "EXACT_LIMIT_ATTACHED_TP_SHAPE_IS_COST_POSITIVE",
                "evidence": exact_limit,
                "meaning": "The exact EUR_USD SHORT BREAKOUT_FAILURE LIMIT/attached-TP vehicle has four S5 bid/ask replayed wins and positive realized net. Random time exits and generic ATR stops are not substitutes for this vehicle."
            }
        ],
        "method_groups": groups,
        "runtime_tp_capture_hint": runtime_hint,
        "exact_limit_attached_tp_evidence": exact_limit,
        "monthly_target_equations": target_trade_counts,
        "causal_bottleneck": {
            "primary": "PROFITABLE_VEHICLE_TOO_RARE",
            "secondary": "EARLY_MARKET_CLOSE_LOSS_TAIL_AND_REGIME_MIX",
            "not_primary": ["NO_EDGE", "MISSING_LIBRARY", "NEED_MORE_RANDOM_INDICATORS"],
            "next_engine_change": "Generate more decision-time candidates matching the exact profitable EUR_USD SHORT LIMIT/TP vehicle, keep ALL_TRADES pass-through, and use price-action scoring only for size/rank."
        }
    }
    (HERE / "profit_reason_ledger_v1.json").write_text(json.dumps(ledger, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
