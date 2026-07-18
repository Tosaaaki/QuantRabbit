#!/usr/bin/env python3
"""Run the fixed DOJO grid with immutable, pessimistic evidence gates.

Every gate is evaluated on both OHLC and OLHC synthetic paths.  The lower
terminal-equity result is authoritative.  TRAIN, VAL, and FINAL are disjoint;
the previously screened 2026-05-10..2026-07-04 interval is excluded from both
holdouts.  All current historical windows have already been inspected, so
these reruns remain hypothesis-only and cannot promote a strategy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_rabbit.dojo_lab_provenance import (  # noqa: E402
    LEGACY_CONTAMINATION_BLOCKERS,
    combine_intrabar_results,
    create_run_root,
    create_trial_dir,
    reserve_window_plan,
    score_session_ledger,
    validate_window_plan,
    write_new_json,
)


PY = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
START_BALANCE_JPY = 200_000.0
HARDENED_SLIPPAGE_PIPS = 0.3
HARDENED_FINANCING_PIPS_PER_DAY = 0.8
M1_ROOT = "/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history_m1_2020_2026"
BOT_MODULE_PATH = REPO / "bots/lab_bot.py"
TRAIN = ("2024-01-02T00:00:00", "2025-07-01T00:00:00")
VAL = ("2025-07-01T00:00:00", "2026-05-10T00:00:00")
FINAL = ("2026-07-04T00:00:00", "2026-07-19T00:00:00")
LEGACY_S5_SCREEN = ("2026-05-10T00:00:00", "2026-07-04T00:00:00")
WINDOWS = {"TRAIN": TRAIN, "VAL": VAL, "FINAL": FINAL}
SCREENED_WINDOWS = {"LEGACY_S5_SCREEN": LEGACY_S5_SCREEN}

EXITS = {
    "E1_tp3_nosl_c240": {"tp_pips": 3, "sl_pips": None, "ceiling_min": 240},
    "E2_tp6_nosl_c480": {"tp_pips": 6, "sl_pips": None, "ceiling_min": 480},
    "E3_tp4_sl12_c480": {"tp_pips": 4, "sl_pips": 12, "ceiling_min": 480},
    "E4_tp10_sl30_c1440": {"tp_pips": 10, "sl_pips": 30, "ceiling_min": 1440},
}
SIGNALS = ["burst", "pullback_limit", "range_fade_limit"]
INTRABAR_PATHS = ("OHLC", "OLHC")


def run_config(
    name: str,
    cfg: dict[str, Any],
    window_role: str,
    window: tuple[str, str],
    run_root: Path,
    *,
    intrabar: str,
    reservation_evidence: dict[str, Any],
) -> dict[str, Any]:
    trial_key = f"{window_role.lower()}__{name}__{intrabar.lower()}"
    session = create_trial_dir(run_root, trial_key)
    owned_cfg = dict(cfg)
    owned_cfg["strategy_owner_id"] = f"dojo:{window_role}:{name}"
    config_text = json.dumps(
        owned_cfg, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    bot_module_sha256 = hashlib.sha256(BOT_MODULE_PATH.read_bytes()).hexdigest()
    env = dict(os.environ)
    env.pop("DOJO_BOT_COMBO", None)
    env["DOJO_BOT_CONFIG"] = config_text
    env["PYTHONPATH"] = str(REPO / "src")
    proc = subprocess.run(
        [
            PY,
            str(REPO / "scripts/run-virtual-market-session.py"),
            "--feed",
            "replay",
            "--session-dir",
            str(session),
            "--pairs",
            "USD_JPY",
            "--balance",
            str(START_BALANCE_JPY),
            "--corpus-root",
            M1_ROOT,
            "--from",
            window[0],
            "--to",
            window[1],
            "--bars-per-second",
            "100000",
            "--state-every",
            "5000",
            "--fast-ledger",
            "--intrabar",
            intrabar,
            "--slippage-pips",
            str(HARDENED_SLIPPAGE_PIPS),
            "--financing-pips-day",
            str(HARDENED_FINANCING_PIPS_PER_DAY),
            "--bot-module",
            str(BOT_MODULE_PATH) + ":Bot",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=7200,
    )
    base = {
        "name": name,
        "window_role": window_role,
        "intrabar": intrabar,
        "trial_key": trial_key,
        "trial_dir": str(session),
        "hardened_costs": {
            "slippage_pips_per_fill": HARDENED_SLIPPAGE_PIPS,
            "financing_pips_per_day": HARDENED_FINANCING_PIPS_PER_DAY,
        },
    }
    if proc.returncode != 0:
        return {
            **base,
            "status": "INVALID_RUNNER_FAILURE",
            "economic_gate_passed": False,
            "promotion_eligible": False,
            "evidence_tier": "HYPOTHESIS_ONLY",
            "promotion_blockers": ["RUNNER_FAILURE", *LEGACY_CONTAMINATION_BLOCKERS],
            "error": proc.stderr[-500:],
        }
    try:
        score = score_session_ledger(
            session / "ledger.jsonl",
            start_balance_jpy=START_BALANCE_JPY,
            window_role=window_role,
            window=window,
            intrabar=intrabar,
            legacy_contaminated=True,
            expected_pairs=("USD_JPY",),
            expected_granularity="M1",
            expected_bot_bar="feed",
            expected_slippage_pips=HARDENED_SLIPPAGE_PIPS,
            expected_financing_pips_per_day=HARDENED_FINANCING_PIPS_PER_DAY,
            expected_bot_module_path=BOT_MODULE_PATH,
            expected_bot_module_sha256=bot_module_sha256,
            expected_bot_config_sha256=hashlib.sha256(
                config_text.encode("utf-8")
            ).hexdigest(),
            expected_bot_config_length=len(config_text),
            reservation_evidence=reservation_evidence,
        )
    except ValueError as exc:
        return {
            **base,
            "status": "INVALID_UNSCOREABLE_TRIAL",
            "economic_gate_passed": False,
            "promotion_eligible": False,
            "evidence_tier": "HYPOTHESIS_ONLY",
            "promotion_blockers": ["UNSCOREABLE_TRIAL", *LEGACY_CONTAMINATION_BLOCKERS],
            "error": str(exc),
        }
    return {**base, **score}


def run_gate(
    name: str,
    cfg: dict[str, Any],
    window_role: str,
    window: tuple[str, str],
    run_root: Path,
    *,
    reservation_evidence: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [
        run_config(
            name,
            cfg,
            window_role,
            window,
            run_root,
            intrabar=intrabar,
            reservation_evidence=reservation_evidence,
        )
        for intrabar in INTRABAR_PATHS
    ]
    try:
        combined = combine_intrabar_results(rows)
    except ValueError as exc:
        combined = {
            "status": "INVALID_INTRABAR_EVIDENCE",
            "gate_passed": False,
            "promotion_eligible": False,
            "evidence_tier": "HYPOTHESIS_ONLY",
            "promotion_blockers": [
                "INCOMPLETE_OHLC_OLHC_GATE",
                *LEGACY_CONTAMINATION_BLOCKERS,
            ],
            "error": str(exc),
        }
    summary = {
        "name": name,
        "window_role": window_role,
        "config": cfg,
        **combined,
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_root",
        nargs="?",
        type=Path,
        help="append-only parent directory for this run (defaults to a temporary directory)",
    )
    parser.add_argument(
        "--global-reservation-registry",
        type=Path,
        help="append-only global holdout reservation JSONL; absent means hypothesis-only",
    )
    args = parser.parse_args()
    plan = validate_window_plan(WINDOWS, screened_windows=SCREENED_WINDOWS)
    out_root = args.output_root or Path(tempfile.mkdtemp())
    run_id, run_root = create_run_root(out_root)
    reservation = reserve_window_plan(
        args.global_reservation_registry,
        run_id=run_id,
        experiment_id="QR_DOJO_LAB_V3",
        plan=plan,
    )
    write_new_json(run_root / "window_reservation.json", reservation)
    scoreboard: list[dict[str, Any]] = []
    gate_summaries: list[dict[str, Any]] = []
    grid: list[tuple[str, dict[str, Any]]] = []
    for signal in SIGNALS:
        for exit_name, exit_cfg in EXITS.items():
            cfg = {
                "signal": signal,
                **exit_cfg,
                "max_concurrent": 3,
                "per_pos_lev": 4.3,
                "atr_floor_pips": 1.0,
                "pull_atr": 0.6,
                "fade_atr": 1.2,
                "eff_max": 0.2,
            }
            grid.append((f"{signal}__{exit_name}", cfg))
    print(f"run_id={run_id} declared grid={len(grid)}", flush=True)

    train_positive: list[tuple[str, dict[str, Any]]] = []
    for name, cfg in grid:
        rows, gate = run_gate(
            name,
            cfg,
            "TRAIN",
            TRAIN,
            run_root,
            reservation_evidence=reservation,
        )
        scoreboard.extend(rows)
        gate_summaries.append(gate)
        print(json.dumps(gate, ensure_ascii=False), flush=True)
        if gate.get("gate_passed") is True:
            train_positive.append((name, cfg))

    val_positive: list[tuple[str, dict[str, Any]]] = []
    for name, cfg in train_positive:
        rows, gate = run_gate(
            name,
            cfg,
            "VAL",
            VAL,
            run_root,
            reservation_evidence=reservation,
        )
        scoreboard.extend(rows)
        gate_summaries.append(gate)
        print(json.dumps(gate, ensure_ascii=False), flush=True)
        if gate.get("gate_passed") is True:
            val_positive.append((name, cfg))

    final_hypotheses: list[dict[str, Any]] = []
    for name, cfg in val_positive:
        rows, gate = run_gate(
            name,
            cfg,
            "FINAL",
            FINAL,
            run_root,
            reservation_evidence=reservation,
        )
        scoreboard.extend(rows)
        gate_summaries.append(gate)
        print(json.dumps(gate, ensure_ascii=False), flush=True)
        if gate.get("gate_passed") is True:
            final_hypotheses.append(gate)

    result = {
        "contract": "QR_DOJO_LAB_V3",
        "run_id": run_id,
        "run_root": str(run_root),
        "declared_grid_size": len(grid),
        "window_plan": plan,
        "window_reservation": reservation,
        "hardened_costs": {
            "slippage_pips_per_fill": HARDENED_SLIPPAGE_PIPS,
            "financing_pips_per_day": HARDENED_FINANCING_PIPS_PER_DAY,
        },
        "intrabar_gate": "BOTH_REQUIRED_IDENTICAL_MANIFEST_PESSIMISTIC_RESOLVED_BALANCE",
        "terminal_score_basis": "FULLY_RESOLVED_BALANCE_ZERO_OPEN_POSITIONS_AND_ORDERS",
        "zero_trade_policy": "FAIL_CLOSED",
        "legacy_results_promotion_eligible": False,
        "promotion_blockers": list(
            dict.fromkeys(
                [
                    *LEGACY_CONTAMINATION_BLOCKERS,
                    *(
                        [str(reservation["promotion_blocker"])]
                        if reservation.get("promotion_blocker")
                        else []
                    ),
                ]
            )
        ),
        "hypothesis_survivors": final_hypotheses,
        "promotion_survivors": [],
        "gate_summaries": gate_summaries,
        "scoreboard": scoreboard,
    }
    out = run_root / "scoreboard.json"
    write_new_json(out, result)
    print(f"scoreboard -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
