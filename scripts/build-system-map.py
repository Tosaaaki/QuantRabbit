#!/usr/bin/env python3
"""Seal the comprehensive system map: every part, its role, and Codex's action.

One artifact that lets Codex grasp the whole system without reading the
session: every module, script, contract, and sealed artifact, annotated
with its layer in the pro-trader brain, its wiring status, and the exact
Codex action it awaits.  Every listed path is verified to exist at build
time; the count mismatch fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_legacy_admission import (
    LegacyAdmissionError,
    load_current_goal_board,
)

MODULES: list[dict[str, str]] = [
    {"path": "src/quant_rabbit/adaptive_exact_s5_profit_engine.py", "layer": "research", "role": "legacy exact-S5 cross-sectional research engine", "wiring": "LEGACY_DIAGNOSTIC", "codex": "do not treat old lock/validation as positive; register new proof through current DOJO"},
    {"path": "src/quant_rabbit/exact28_m5_history_manifest.py", "layer": "data", "role": "long-horizon M5 manifest validator (acquisition gate)", "wiring": "COMPLETE", "codex": "P0-2 repairs then acquire 2020-2026 M5"},
    {"path": "src/quant_rabbit/fast_bot_shadow_orchestrator.py", "layer": "bot-shadow", "role": "5096-cell shadow orchestrator + admission binding for future GO", "wiring": "SHADOW_WIRED", "codex": "consume admission binding in live GO contract"},
    {"path": "src/quant_rabbit/fast_bot_technical_grid_backtest.py", "layer": "bot-shadow", "role": "sealed 182-candidate grid backtest (audited, ambiguity=full-SL)", "wiring": "COMPLETE", "codex": "none"},
    {"path": "src/quant_rabbit/close_distance_gate.py", "layer": "gate", "role": "pre-entry close-crossing gate (close-leak family fix)", "wiring": "MODULE_ONLY", "codex": "wire into trader cycle entry path"},
    {"path": "src/quant_rabbit/cost_window_mask.py", "layer": "gate", "role": "pre-declared high-cost UTC window mask", "wiring": "MODULE_ONLY", "codex": "wire into trader cycle entry path"},
    {"path": "src/quant_rabbit/currency_exposure_guard.py", "layer": "gate", "role": "per-currency net NAV exposure cap", "wiring": "MODULE_ONLY", "codex": "wire as pre-order check"},
    {"path": "src/quant_rabbit/portfolio_covariance_shadow.py", "layer": "risk", "role": "covariance vol-target sizing + Meucci effective bets", "wiring": "MODULE_ONLY", "codex": "feed rolling correlations; wire as sizing check"},
    {"path": "src/quant_rabbit/conviction_ladder.py", "layer": "sizing", "role": "pre-declared conviction sizing + daily stop", "wiring": "MODULE_ONLY", "codex": "wire into order sizing"},
    {"path": "src/quant_rabbit/daily_loss_overlay.py", "layer": "risk", "role": "causal intraday 50p stop (ADOPTED) + rejected day-skip", "wiring": "MODULE_ONLY", "codex": "wire stop into cycle; never wire day-skip"},
    {"path": "src/quant_rabbit/regime_classifier_shadow.py", "layer": "perception", "role": "measured regime x vol classifier (scale-free thresholds)", "wiring": "MODULE_ONLY", "codex": "run per cycle; feeds router+supervision"},
    {"path": "src/quant_rabbit/regime_family_router.py", "layer": "routing", "role": "regime-cell -> family eligibility (honest uncovered cells)", "wiring": "MODULE_ONLY", "codex": "wire into candidate admission"},
    {"path": "src/quant_rabbit/regime_supervision_v2.py", "layer": "tuning", "role": "AI tuning vocabulary: pair x family GO/CAUTION/STOP + regime", "wiring": "MODULE_ONLY", "codex": "AI trader emits; bot consumes with TTL"},
    {"path": "src/quant_rabbit/supervision_outcome_scorer.py", "layer": "metacognition", "role": "scores supervision accuracy; auto-CAUTION below floor", "wiring": "MODULE_ONLY", "codex": "monthly scoring job once live data accrues"},
    {"path": "src/quant_rabbit/conviction_calibration_shadow.py", "layer": "metacognition", "role": "condition grounding + Brier/ECE calibration multiplier", "wiring": "MODULE_ONLY", "codex": "build calibration table from live ledger"},
    {"path": "src/quant_rabbit/shadow_trading_brain.py", "layer": "composition", "role": "composed brain cycle; layer-2 read reserved for CODEX_AI_TRADER", "wiring": "MODULE_ONLY", "codex": "mirror this composition in the live cycle"},
    {"path": "src/quant_rabbit/portfolio_inventory_reconciliation.py", "layer": "portfolio", "role": "three-way broker/ledger/lane stocktake, fail-closed", "wiring": "MODULE_ONLY", "codex": "wire per cycle; UNRECONCILED blocks entries"},
    {"path": "src/quant_rabbit/gate_throughput_slo.py", "layer": "meta", "role": "gate funnel SLO; names killer gates on breach", "wiring": "MODULE_ONLY", "codex": "wire funnel recording into cycle"},
    {"path": "src/quant_rabbit/prospective_registry.py", "layer": "research-discipline", "role": "parallel pre-registered future windows (WRC denominators)", "wiring": "MODULE_ONLY", "codex": "register every new candidate before its window"},
    {"path": "src/quant_rabbit/rejection_taxonomy.py", "layer": "research-discipline", "role": "mandatory death codes for rejected candidates", "wiring": "IN_USE", "codex": "record every rejection"},
    {"path": "src/quant_rabbit/micro_live_promotion_contract.py", "layer": "promotion", "role": "T1 micro-live tier; operator-approval-bound activation", "wiring": "MODULE_ONLY", "codex": "implement T1 order path (option B, post-Monday)"},
    {"path": "src/quant_rabbit/asymmetric_exit_shadow.py", "layer": "tactic-lab", "role": "structure-break exit evaluator (paired re-scoring)", "wiring": "LAB", "codex": "use in M5-era exit research"},
    {"path": "src/quant_rabbit/addon_ladder_shadow.py", "layer": "tactic-lab", "role": "bounded pyramid/nanpin ladder with add-fill epochs", "wiring": "LAB", "codex": "use for state-conditional add research"},
    {"path": "src/quant_rabbit/range_rail_shadow.py", "layer": "tactic-lab", "role": "passive rail mean-reversion evaluator (lane F mechanism)", "wiring": "LAB", "codex": "lane F research on M5 corpus"},
    {"path": "src/quant_rabbit/nav_normalization.py", "layer": "accounting", "role": "pips -> NAV fraction conversion", "wiring": "MODULE_ONLY", "codex": "add NAV columns to scorecards"},
]
KEY_SCRIPTS = [
    "scripts/run-adaptive-exact-s5-profit-research.py",
    "scripts/build-exact28-m5-history-manifest.py",
    "scripts/run-overlay-sweep-lab.py",
    "scripts/run-state-conditional-tactics.py",
    "scripts/run-monthly-multiple-distribution.py",
    "scripts/build-adopted-stack-tuning-contract.py",
    "scripts/build-system-map.py",
]
KEY_DOCS = [
    "docs/design_weakness_ledger_20260718.md",
    "docs/DOJO_PROGRAM.md",
    "docs/DOJO_WORKTREE_INVENTORY_20260719.md",
    "docs/virtual_market_environment.md",
]
LEGACY_RESEARCH_DOCS = [
    "docs/design_profit_path_4x_20260718.md",
    "docs/design_profit_structure_repair_20260718.md",
    "docs/runbook_monday_go_live_20260720.md",
]
LEGACY_POSITIVE_ARTIFACTS = [
    "research/data/adopted_stack_tuning_contract_v1.json",
    "research/data/adaptive_exact_s5_train_lock_v1.json",
    "research/data/adaptive_exact_s5_prospective_final_lock_v1.json",
    "research/data/monthly_multiple_distribution_v1.json",
]


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root

    try:
        board_path, board = load_current_goal_board(root / "research/registries")
    except LegacyAdmissionError as exc:
        raise ValueError(f"current DOJO goal board is invalid: {exc}") from exc
    try:
        board_relative = board_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("current DOJO goal board must be inside repo root") from exc
    key_artifacts = [board_relative]

    missing = [
        m["path"] for m in MODULES if not (root / m["path"]).is_file()
    ] + [
        p
        for p in KEY_SCRIPTS + KEY_DOCS + LEGACY_RESEARCH_DOCS + key_artifacts
        if not (root / p).is_file()
    ]
    if missing:
        raise ValueError(f"system map lists missing paths: {missing}")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
        check=True,
    ).stdout.strip()

    body: dict[str, Any] = {
        "contract": "QR_SYSTEM_MAP_V2",
        "schema_version": 2,
        "branch_head": head,
        "brain_layers": {
            "1_perception": "evidence packet + regime_classifier_shadow (+W23 indicators after M5)",
            "2_intuition": "CODEX_AI_TRADER market read (reserved; brain never authors)",
            "3_metacognition": "supervision_outcome_scorer + conviction_calibration_shadow",
            "4_discipline": "gates + conviction_ladder + covariance + inventory + margin runtime cap",
        },
        "modules": MODULES,
        "key_scripts": KEY_SCRIPTS,
        "key_docs": KEY_DOCS,
        "key_artifacts": key_artifacts,
        "single_entry_point_for_codex": board_relative,
        "current_dojo_validity": {
            "goal_board_path": board_relative,
            "goal_board_sha256": board["board_sha256"],
            "edge_status": board.get("edge_status"),
            "goal_status": board.get("goal_status"),
            "promotion_possible": board["proof_admission"]["promotion_possible"],
        },
        "legacy_positive_artifacts": {
            "paths": LEGACY_POSITIVE_ARTIFACTS,
            "status": "NOT_AN_ENTRY_POINT_REQUIRES_CURRENT_DOJO_ADMISSION",
            "builder": "scripts/build-adopted-stack-tuning-contract.py",
        },
        "legacy_research_docs": {
            "paths": LEGACY_RESEARCH_DOCS,
            "status": "DIAGNOSTIC_HISTORY_NOT_AN_ADMISSION_SURFACE",
        },
        "wiring_status_legend": {
            "COMPLETE": "done and tested on this branch",
            "SHADOW_WIRED": "wired into the shadow orchestrator",
            "MODULE_ONLY": "tested module awaiting live-cycle wiring by Codex",
            "LAB": "research evaluator, not a runtime component",
            "IN_USE": "already used by tonight's research pipeline",
            "LEGACY_DIAGNOSTIC": (
                "historical research code; current DOJO registration is mandatory "
                "before any positive reuse"
            ),
        },
        "shadow_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    sealed = {**body, "system_map_sha256": _canonical_sha(body)}
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
                "status": "SYSTEM_MAP_SEALED",
                "modules": len(MODULES),
                "system_map_sha256": sealed["system_map_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
