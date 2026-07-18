#!/usr/bin/env python3
"""Seal the adopted Monday stack as ONE machine-readable tuning contract.

Everything Codex needs to wire the bot — adopted parameters, rejected
techniques with death codes, gate wiring map, supervision vocabulary, and
the provenance sha of every sealed experiment behind each decision — in a
single digest-sealed artifact.  The bot consumes THIS, not the prose
ledger.  The contract grants no live permission; activation still follows
the runbook's operator-approval phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

REQUIRED_SOURCES = {
    "survivor_lock": "adaptive_exact_s5_train_lock_v1.json",
    "validation_replication": "adaptive_exact_s5_validation_replication_v1.json",
    "throttle_comparison": "throttle_mode_comparison_v1.json",
    "overlay_sweep": "overlay_sweep_lab_v1.json",
    "nanpin_margin_feasible": "nanpin_margin_feasible_v1.json",
    "nanpin_true_concurrency": "nanpin_true_concurrency_v1.json",
    "monthly_distribution": "monthly_multiple_distribution_v1.json",
    "all_weather_attribution": "all_weather_attribution_v1.json",
    "cell_gating": "regime_cell_gating_rehearsal_v1.json",
    "lane_addition": "lane_addition_combination_v1.json",
}
OPTIONAL_SOURCES = {"discovery_batch_2": "discovery_batch_2_v1.json"}
DIGEST_KEYS = (
    "lock_sha256", "evaluation_sha256", "comparison_sha256", "sweep_sha256",
    "rehearsal_sha256", "distribution_sha256", "attribution_sha256",
    "combination_sha256", "batch_sha256",
)


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_sealed(path: Path) -> tuple[dict[str, Any], str]:
    """Accept the digest key whose recomputation verifies.

    Artifacts may carry REFERENCE shas to other artifacts (e.g. the
    validation replication holds lock_sha256) alongside their own seal, so
    a key is the self-digest only if removing it reproduces its value.
    """

    value = json.loads(path.read_text(encoding="utf-8"))
    for key in DIGEST_KEYS:
        if key in value:
            body = {k: v for k, v in value.items() if k != key}
            if value[key] == _canonical_sha(body):
                return value, str(value[key])
    raise ValueError(f"{path.name}: no digest key verifies")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    provenance: dict[str, str] = {}
    sources: dict[str, dict[str, Any]] = {}
    for name, filename in REQUIRED_SOURCES.items():
        value, sha = _load_sealed(args.data_dir / filename)
        sources[name] = value
        provenance[name] = sha
    for name, filename in OPTIONAL_SOURCES.items():
        path = args.data_dir / filename
        if path.exists():
            value, sha = _load_sealed(path)
            sources[name] = value
            provenance[name] = sha

    lock = sources["survivor_lock"]
    body: dict[str, Any] = {
        "contract": "QR_ADOPTED_STACK_TUNING_CONTRACT_V1",
        "schema_version": 1,
        "adopted": {
            "strategy": {
                "spec_id": lock["spec"]["spec_id"],
                "spec": dict(lock["spec"]),
                "evaluation_policy": dict(lock["evaluation_policy"]),
                "lock_sha256": lock["lock_sha256"],
                "status": "VALIDATION_REPLICATED_UNPROVEN_UNTIL_FUTURE_WINDOW",
            },
            "intraday_stop": {
                "policy": "CAUSAL_INTRADAY_REALIZED_LOSS_STOP_V1",
                "stop_pips": 50.0,
                "mode": "SKIP",
                "module": "quant_rabbit.daily_loss_overlay",
                "evidence": provenance["throttle_comparison"],
            },
            "sizing": {
                "leverage_cap": 25.0,
                "executable_base_leverage": 21.4,
                "reason": "observed peak 14 concurrent units breaks 25x/12 feasibility",
                "slots": 12,
                "risk_fractions": "quant_rabbit.conviction_ladder (0.25% base, 2x/4x by named conditions, -3% daily stop)",
                "evidence": provenance["nanpin_true_concurrency"],
            },
            "margin_cap": {
                "peak_usage_cap": 0.92,
                "enforcement": "RUNTIME_REFUSE_ORDER_WHEN_HEADROOM_INSUFFICIENT",
                "never_calibrate_from_history": True,
                "operator_grant": "92-95% allowed as PEAK cap, not constant usage",
                "evidence": provenance["nanpin_margin_feasible"],
            },
            "entry_gates_in_order": [
                {"gate": "portfolio_inventory_reconciliation", "fail_closed": "UNRECONCILED blocks all entries"},
                {"gate": "close_distance_gate", "params": {"safety_margin_minutes": 5}},
                {"gate": "cost_window_mask", "params": {"masked_utc_windows": [[1260, 1320]]}},
                {"gate": "currency_exposure_guard", "params": {"currency_cap_fraction": 0.5}},
            ],
            "supervision": {
                "contract": "QR_AI_REGIME_SUPERVISION_V2",
                "cadence_hours": 6,
                "layer2_owner": "CODEX_AI_TRADER",
                "scoring": "quant_rabbit.supervision_outcome_scorer (auto-CAUTION below 50% accuracy)",
            },
        },
        "rejected_do_not_wire": [
            {"technique": "day_skip_after_losses", "death_code": "ROBUSTNESS_FLOOR_FAILED", "val_cost_pips": 1658.5},
            {"technique": "time_of_day_filters", "death_code": "ROBUSTNESS_FLOOR_FAILED", "note": "operator rule: clock heuristics banned"},
            {"technique": "min_abs_score_floor", "death_code": "ROBUSTNESS_FLOOR_FAILED"},
            {"technique": "spread_to_score_ratio", "death_code": "ROBUSTNESS_FLOOR_FAILED"},
            {"technique": "score_proportional_sizing", "death_code": "DIRECTION_WRONG", "note": "|score| is not a conviction proxy"},
            {"technique": "trailing_stop_30p", "death_code": "EXIT_TIMING_LEFT_PROFIT"},
            {"technique": "breakeven_after_20p", "death_code": "EXIT_TIMING_LEFT_PROFIT"},
            {"technique": "half_take_20p_runner", "death_code": "EXIT_TIMING_LEFT_PROFIT"},
            {"technique": "extend_hold_24h_unconditional", "death_code": "ROBUSTNESS_FLOOR_FAILED", "note": "regime-conditional variant under test in discovery batch 2"},
            {"technique": "pyramid_adds", "death_code": "DIRECTION_WRONG"},
            {"technique": "naive_range_rails_60m", "death_code": "COST_SPREAD_DOMINATES_EDGE"},
            {"technique": "nanpin_for_growth", "death_code": "EXECUTION_INFEASIBLE", "note": "reclassified as drawdown shaper; re-evaluate under T1"},
            {"technique": "static_side_bias_rules", "death_code": "REGIME_MISMATCH", "note": "side dominance reversed between windows; operator no-direction-bias rule proven"},
            {"technique": "constant_95pct_single_position", "death_code": "EXECUTION_INFEASIBLE", "note": "~26 pips to margin closeout"},
        ],
        "codex_wiring_order": [
            "merge branch codex/episode-s5-outcome",
            "wire entry gates + ladder into trader cycle (runbook Phase 1)",
            "implement supervision V2 consumption in fast bot",
            "operator approval for entry authority (runbook Phase 2, option A)",
            "run final-test subcommand after 2026-08-03",
            "M5 validator repairs -> acquisition -> lane research (P0-2/P0-3)",
        ],
        "provenance_sha256": provenance,
        "monthly_distribution_at_adoption": sources["monthly_distribution"]["distribution_rows"],
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "activation_requires_operator_approval": True,
    }
    sealed = {**body, "tuning_contract_sha256": _canonical_sha(body)}
    payload = json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_name, args.output)
    print(
        json.dumps(
            {
                "status": "TUNING_CONTRACT_SEALED",
                "tuning_contract_sha256": sealed["tuning_contract_sha256"],
                "sources_bound": sorted(provenance),
                "rejected_count": len(body["rejected_do_not_wire"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
