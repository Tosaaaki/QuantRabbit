#!/usr/bin/env python3
"""Build a zero-authority profitability frontier from existing replay truth.

The frontier does not invent a profitable strategy.  It rejects negative
walk-forward candidates, classifies positive-but-thin exact-vehicle evidence,
and identifies the only lanes worth collecting prospectively in shadow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quant_rabbit.fast_bot_profitability_gate import (  # noqa: E402
    assess_profitability_evidence,
    build_profitability_evidence,
)


SHOCK_CONTRACT = "QR_FAST_BOT_SHOCK_PROFITABILITY_WALK_FORWARD_V1"
NONSHOCK_CONTRACT = "QR_FAST_BOT_NONSHOCK_WALK_FORWARD_V1"
FRONTIER_CONTRACT = "QR_FAST_BOT_PROFITABILITY_FRONTIER_V1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact must be a mapping: {path}")
    return value


def _validate_walk_forward(value: Mapping[str, Any], contract: str) -> None:
    if value.get("contract") != contract:
        raise ValueError(f"walk-forward contract mismatch: {contract}")
    if value.get("execution_authority") != "NONE":
        raise ValueError("walk-forward execution authority must be NONE")
    if value.get("broker_mutation_allowed") is not False:
        raise ValueError("walk-forward broker mutation must be false")
    if value.get("external_order_attempts") != 0 or value.get("external_orders") != 0:
        raise ValueError("walk-forward external order counts must be zero")
    selection = value.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("walk-forward selection missing")
    if selection.get("live_promotion_allowed") is not False:
        raise ValueError("walk-forward live promotion must be false")


def _best_validation_cell(value: Mapping[str, Any]) -> dict[str, Any]:
    cells = value.get("pre_holdout_cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("pre_holdout_cells missing")

    def rank(cell: Mapping[str, Any]) -> tuple[float, float]:
        validation = cell.get("validation")
        if not isinstance(validation, Mapping):
            return (-1.0, float("-inf"))
        profit_factor = validation.get("risk_scaled_profit_factor")
        net = validation.get("risk_scaled_net_pip_units")
        return (
            float(profit_factor) if profit_factor is not None else -1.0,
            float(net) if net is not None else float("-inf"),
        )

    best = max((cell for cell in cells if isinstance(cell, Mapping)), key=rank)
    validation = best.get("validation")
    assert isinstance(validation, Mapping)
    result = {
        "candidate": best.get("candidate"),
        "geometry": best.get("geometry"),
        "target_r": best.get("target_r"),
        "pre_holdout_qualified": best.get("pre_holdout_qualified") is True,
        "validation_trades": validation.get("trades"),
        "validation_net_pips": validation.get("net_pips"),
        "validation_profit_factor": validation.get("profit_factor"),
        "validation_risk_scaled_net_pip_units": validation.get(
            "risk_scaled_net_pip_units"
        ),
        "validation_risk_scaled_profit_factor": validation.get(
            "risk_scaled_profit_factor"
        ),
        "validation_p05_trade_pips": validation.get("p05_trade_pips"),
        "validation_maximum_loss_streak": validation.get("maximum_loss_streak"),
    }
    return result


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def build_frontier(
    *,
    shock: Mapping[str, Any],
    nonshock: Mapping[str, Any],
    audjpy: Mapping[str, Any],
    shock_sha256: str,
    nonshock_sha256: str,
    audjpy_sha256: str,
    generated_at_utc: datetime,
) -> dict[str, Any]:
    _validate_walk_forward(shock, SHOCK_CONTRACT)
    _validate_walk_forward(nonshock, NONSHOCK_CONTRACT)

    exact = audjpy.get("exact_shape_replay")
    rank = audjpy.get("rank_only_precision_subset")
    requested = audjpy.get("requested_shape")
    if not isinstance(exact, Mapping) or not isinstance(rank, Mapping):
        raise ValueError("AUD/JPY replay metrics missing")
    if not isinstance(requested, Mapping):
        raise ValueError("AUD/JPY requested shape missing")
    if audjpy.get("live_side_effects") not in ([], None):
        raise ValueError("AUD/JPY source contains live side effects")

    pair = str(requested.get("pair") or "").upper()
    side = str(requested.get("side") or "").upper()
    method = str(requested.get("method") or "").upper()
    order_type = str(requested.get("order_type") or "").upper()
    common = {
        "lane_id": str(
            requested.get("lane_id")
            or f"failure_trader:{pair}:{side}:{method}:{order_type}"
        ),
        "pair": pair,
        "side": side,
        "method": method,
        "order_type": order_type,
        "source_artifact_sha256": audjpy_sha256,
        "generated_at_utc": generated_at_utc,
    }
    exact_evidence = build_profitability_evidence(
        **common,
        metrics=exact,
        evidence_end_utc=_parse_utc(str(exact["replay_window_utc"]["last"])),
        rank_only=False,
    )
    rank_evidence = build_profitability_evidence(
        **common,
        metrics=rank,
        evidence_end_utc=_parse_utc(str(rank["replay_window_utc"]["last"])),
        rank_only=True,
    )
    exact_gate = assess_profitability_evidence(exact_evidence)
    rank_gate = assess_profitability_evidence(rank_evidence)

    shock_best = _best_validation_cell(shock)
    nonshock_best = _best_validation_cell(nonshock)
    rejected_loss = sum(
        max(0.0, -float(item.get("validation_net_pips") or 0.0))
        for item in (shock_best, nonshock_best)
    )
    collect = [
        gate
        for gate in (exact_gate, rank_gate)
        if gate["status"] == "COLLECT_MORE_INDEPENDENT_DAYS"
    ]
    ready = [
        gate
        for gate in (exact_gate, rank_gate)
        if gate["status"] == "SHADOW_FORWARD_OBSERVATION_READY"
    ]
    return {
        "contract": FRONTIER_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": generated_at_utc.astimezone(timezone.utc).isoformat(),
        "status": "CAPITAL_PRESERVATION_IMPROVED_PROFITABILITY_UNPROVEN",
        "source_artifacts": {
            "shock_walk_forward_sha256": shock_sha256,
            "nonshock_walk_forward_sha256": nonshock_sha256,
            "audjpy_limit_replay_sha256": audjpy_sha256,
        },
        "rejected_primary_candidates": [
            {
                "lane": "EUR_USD_SHOCK_CONTINUATION",
                "decision": "REJECT_PRE_HOLDOUT_NEGATIVE_EXPECTANCY",
                "best_validation_cell": shock_best,
            },
            {
                "lane": "EUR_USD_NONSHOCK_HOURLY",
                "decision": "REJECT_PRE_HOLDOUT_NEGATIVE_EXPECTANCY",
                "best_validation_cell": nonshock_best,
            },
        ],
        "replay_loss_pips_avoided_by_rejecting_best_negative_cells": round(
            rejected_loss, 6
        ),
        "audjpy_limit_evidence": {
            "exact_shape": {"evidence": exact_evidence, "gate": exact_gate},
            "rank_only_precision": {"evidence": rank_evidence, "gate": rank_gate},
        },
        "shadow_forward_observation_ready": ready,
        "shadow_collect_more_independent_days": collect,
        "trade_eligible_candidates": [],
        "profitability_claim": "UNPROVEN",
        "next_profit_work": (
            "COLLECT_DECONCENTRATED_EXACT_LIMIT_FORWARD_TRUTH"
            if collect
            else "SEARCH_NEW_PREREGISTERED_ENTRY_FAMILIES"
        ),
        "execution_authority": "NONE",
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "live_order_gateway_invocation_count": 0,
        "external_order_attempts": 0,
        "external_orders": 0,
        "manual_tagless_policy": "NO_TOUCH",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shock", type=Path, required=True)
    parser.add_argument("--nonshock", type=Path, required=True)
    parser.add_argument("--audjpy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        "shock": args.shock,
        "nonshock": args.nonshock,
        "audjpy": args.audjpy,
    }
    payload = build_frontier(
        shock=_load(args.shock),
        nonshock=_load(args.nonshock),
        audjpy=_load(args.audjpy),
        shock_sha256=hashlib.sha256(args.shock.read_bytes()).hexdigest(),
        nonshock_sha256=hashlib.sha256(args.nonshock.read_bytes()).hexdigest(),
        audjpy_sha256=hashlib.sha256(args.audjpy.read_bytes()).hexdigest(),
        generated_at_utc=datetime.now(timezone.utc),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
