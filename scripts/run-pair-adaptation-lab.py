#!/usr/bin/env python3
"""Per-pair DOJO adaptation with immutable, disjoint evidence lanes.

The old S5-positive list is recorded only as contaminated history.  It no
longer injects candidates into selection, and its 2026-05-10..2026-07-04
screen interval is excluded from VAL and FINAL.  Every phase runs both OHLC
and OLHC; the lower terminal-equity result is authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_rabbit.dojo_lab_provenance import (  # noqa: E402
    LEGACY_CONTAMINATION_BLOCKERS,
    combine_intrabar_results,
    create_run_root,
    create_trial_dir,
    owner_concurrency_caps_from_config,
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
S5_ROOT = "/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history"
TRAIN = ("2024-07-01T00:00:00", "2025-07-01T00:00:00")
VAL = ("2025-07-01T00:00:00", "2026-05-10T00:00:00")
FINAL = ("2026-07-04T00:00:00", "2026-07-19T00:00:00")
LEGACY_S5_SCREEN = ("2026-05-10T00:00:00", "2026-07-04T00:00:00")
WINDOWS = {"TRAIN": TRAIN, "VAL": VAL, "FINAL": FINAL}
SCREENED_WINDOWS = {"LEGACY_S5_SCREEN": LEGACY_S5_SCREEN}
INTRABAR_PATHS = ("OHLC", "OLHC")
BOT_MODULE_PATH = REPO / "bots/lab_bot.py"
BOT_DEPENDENCY_PATHS = (
    REPO / "src" / "quant_rabbit" / "dojo_bot_catalog.py",
    REPO / "src" / "quant_rabbit" / "dojo_lab_provenance.py",
    REPO / "src" / "quant_rabbit" / "virtual_broker.py",
)

NEW_PAIRS = [
    "EUR_GBP",
    "EUR_AUD",
    "EUR_CAD",
    "EUR_CHF",
    "EUR_NZD",
    "GBP_AUD",
    "GBP_CAD",
    "GBP_CHF",
    "GBP_NZD",
    "AUD_NZD",
    "AUD_CAD",
    "AUD_CHF",
    "NZD_CAD",
    "NZD_CHF",
    "CAD_CHF",
    "USD_CHF",
    "USD_CAD",
]
LEGACY_S5_SCREEN_POSITIVE = [
    "NZD_JPY",
    "CAD_JPY",
    "AUD_CHF",
    "NZD_CAD",
    "EUR_NZD",
]
GRID = [
    {"fade_atr": fade_atr, "eff_max": eff_max, "tp_atr": tp_atr}
    for fade_atr in (1.0, 1.5)
    for eff_max in (0.15, 0.25)
    for tp_atr in (2.0, 3.0)
]


def base_cfg(pair: str, geometry: dict[str, float]) -> dict[str, Any]:
    return {
        "signal": "range_fade_limit",
        "pairs": [pair],
        "tp_atr": geometry["tp_atr"],
        "sl_pips": None,
        "ceiling_min": 480,
        "max_concurrent": 1,
        "per_pos_lev": 4.3,
        "atr_floor_pips": 0.5,
        "fade_atr": geometry["fade_atr"],
        "eff_max": geometry["eff_max"],
    }


def feed_pairs(pair: str) -> str:
    """Include the quote-conversion feeds needed for JPY accounting."""

    quote = pair.split("_")[1]
    feeds = [pair]
    if quote != "JPY":
        feeds.append("USD_JPY")
        if quote != "USD" and f"{quote}_JPY" != pair:
            feeds.append(f"{quote}_JPY")
    return ",".join(dict.fromkeys(feeds))


def run(
    pair: str,
    cfg: dict[str, Any],
    window_role: str,
    window: tuple[str, str],
    tag: str,
    run_root: Path,
    *,
    intrabar: str,
    reservation_evidence: dict[str, Any],
    s5: bool = False,
) -> dict[str, Any]:
    trial_key = f"pa_{tag}_{intrabar.lower()}"
    session = create_trial_dir(run_root, trial_key)
    owned_cfg = dict(cfg)
    owned_cfg["strategy_owner_id"] = f"pair:{window_role}:{tag}"
    strategy_owner_id = owned_cfg["strategy_owner_id"]
    config_text = json.dumps(
        owned_cfg, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    bot_module_sha256 = hashlib.sha256(BOT_MODULE_PATH.read_bytes()).hexdigest()
    env = dict(os.environ)
    env.pop("DOJO_BOT_COMBO", None)
    env["DOJO_BOT_CONFIG"] = config_text
    env["PYTHONPATH"] = str(REPO / "src")
    command = [
        PY,
        str(REPO / "scripts/run-virtual-market-session.py"),
        "--feed",
        "replay",
        "--session-dir",
        str(session),
        "--pairs",
        feed_pairs(pair),
        "--balance",
        str(START_BALANCE_JPY),
        "--from",
        window[0],
        "--to",
        window[1],
        "--bars-per-second",
        "100000",
        "--state-every",
        "100000",
        "--fast-ledger",
        "--intrabar",
        intrabar,
        "--slippage-pips",
        str(HARDENED_SLIPPAGE_PIPS),
        "--financing-pips-day",
        str(HARDENED_FINANCING_PIPS_PER_DAY),
        "--bot-module",
        str(BOT_MODULE_PATH) + ":Bot",
        "--settle-at-end",
        "--strategy-owner-id",
        strategy_owner_id,
        *[
            item
            for path in BOT_DEPENDENCY_PATHS
            for item in ("--bot-dependency", str(path))
        ],
    ]
    if s5:
        command += [
            "--granularity",
            "S5",
            "--bot-bar",
            "M1",
            "--corpus-root",
            S5_ROOT,
        ]
    else:
        command += ["--corpus-root", M1_ROOT]
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        timeout=3600,
    )
    base = {
        "tag": tag,
        "pair": pair,
        "window_role": window_role,
        "intrabar": intrabar,
        "trial_key": trial_key,
        "trial_dir": str(session),
        "granularity": "S5" if s5 else "M1",
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
        pair_cap, global_cap = owner_concurrency_caps_from_config(owned_cfg)
        score = score_session_ledger(
            session / "ledger.jsonl",
            start_balance_jpy=START_BALANCE_JPY,
            window_role=window_role,
            window=window,
            intrabar=intrabar,
            legacy_contaminated=True,
            expected_pairs=tuple(feed_pairs(pair).split(",")),
            expected_granularity="S5" if s5 else "M1",
            expected_bot_bar="M1" if s5 else "feed",
            expected_period_end_settlement=True,
            expected_slippage_pips=HARDENED_SLIPPAGE_PIPS,
            expected_financing_pips_per_day=HARDENED_FINANCING_PIPS_PER_DAY,
            expected_bot_module_path=BOT_MODULE_PATH,
            expected_bot_module_sha256=bot_module_sha256,
            expected_bot_dependency_sha256={
                str(path.relative_to(REPO)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in BOT_DEPENDENCY_PATHS
            },
            expected_strategy_owner_id=strategy_owner_id,
            expected_bot_config_sha256=hashlib.sha256(
                config_text.encode("utf-8")
            ).hexdigest(),
            expected_bot_config_length=len(config_text),
            reservation_evidence=reservation_evidence,
            expected_max_concurrent_per_pair=pair_cap,
            expected_global_max_concurrent=global_cap,
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
    pair: str,
    cfg: dict[str, Any],
    window_role: str,
    window: tuple[str, str],
    tag: str,
    run_root: Path,
    *,
    reservation_evidence: dict[str, Any],
    s5: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [
        run(
            pair,
            cfg,
            window_role,
            window,
            tag,
            run_root,
            intrabar=intrabar,
            reservation_evidence=reservation_evidence,
            s5=s5,
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
    return rows, {
        "tag": tag,
        "pair": pair,
        "window_role": window_role,
        "geometry": {key: cfg[key] for key in ("fade_atr", "eff_max", "tp_atr")},
        **combined,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_root",
        type=Path,
        help="append-only parent directory for this run",
    )
    parser.add_argument(
        "--global-reservation-registry",
        type=Path,
        help="append-only global holdout reservation JSONL; absent means hypothesis-only",
    )
    args = parser.parse_args()
    plan = validate_window_plan(WINDOWS, screened_windows=SCREENED_WINDOWS)
    out_root = args.output_root
    run_id, run_root = create_run_root(out_root)
    reservation = reserve_window_plan(
        args.global_reservation_registry,
        run_id=run_id,
        experiment_id="QR_PAIR_ADAPTATION_LAB_V3",
        plan=plan,
    )
    write_new_json(run_root / "window_reservation.json", reservation)
    board: list[dict[str, Any]] = []
    gate_summaries: list[dict[str, Any]] = []

    default_geometry = {"fade_atr": 1.2, "eff_max": 0.2, "tp_atr": 3.0}
    phase1_pass: list[str] = []
    for pair in NEW_PAIRS:
        rows, gate = run_gate(
            pair,
            base_cfg(pair, default_geometry),
            "TRAIN",
            TRAIN,
            f"p1_{pair}",
            run_root,
            reservation_evidence=reservation,
        )
        board.extend(rows)
        gate.update({"phase": 1, "screen_threshold_jpy": -20_000})
        gate_summaries.append(gate)
        print(json.dumps(gate, ensure_ascii=False), flush=True)
        entries = gate.get("intrabar_entries") or {}
        closeouts = gate.get("intrabar_margin_closeouts") or {}
        resolved = gate.get("intrabar_terminal_resolution") or {}
        if (
            all(int(entries.get(path, 0)) > 0 for path in INTRABAR_PATHS)
            and all(int(closeouts.get(path, 0)) == 0 for path in INTRABAR_PATHS)
            and all(bool(resolved.get(path)) for path in INTRABAR_PATHS)
            and float(gate.get("pessimistic_terminal_net_jpy", -(10**12))) > -20_000
        ):
            phase1_pass.append(pair)

    # The legacy S5-positive list came from the interval now quarantined as a
    # screen.  It is recorded for audit but cannot inject a holdout candidate.
    candidates = sorted(set(phase1_pass))
    selected: dict[str, dict[str, float]] = {}
    for pair in candidates:
        best: dict[str, Any] | None = None
        for index, geometry in enumerate(GRID):
            rows, gate = run_gate(
                pair,
                base_cfg(pair, geometry),
                "TRAIN",
                TRAIN,
                f"p2_{pair}_{index}",
                run_root,
                reservation_evidence=reservation,
            )
            board.extend(rows)
            gate.update({"phase": 2, "grid_index": index})
            gate_summaries.append(gate)
            print(json.dumps(gate, ensure_ascii=False), flush=True)
            if gate.get("gate_passed") is True and (
                best is None
                or float(gate["pessimistic_terminal_net_jpy"])
                > float(best["pessimistic_terminal_net_jpy"])
            ):
                best = gate
        if best is not None:
            selected[pair] = dict(best["geometry"])

    val_positive: dict[str, dict[str, float]] = {}
    for pair, geometry in selected.items():
        rows, gate = run_gate(
            pair,
            base_cfg(pair, geometry),
            "VAL",
            VAL,
            f"p3v_{pair}",
            run_root,
            reservation_evidence=reservation,
        )
        board.extend(rows)
        gate.update({"phase": 3, "gate": "VAL"})
        gate_summaries.append(gate)
        print(json.dumps(gate, ensure_ascii=False), flush=True)
        if gate.get("gate_passed") is True:
            val_positive[pair] = geometry

    final_hypotheses: list[dict[str, Any]] = []
    for pair, geometry in val_positive.items():
        rows, gate = run_gate(
            pair,
            base_cfg(pair, geometry),
            "FINAL",
            FINAL,
            f"p3f_{pair}",
            run_root,
            reservation_evidence=reservation,
            s5=True,
        )
        board.extend(rows)
        gate.update({"phase": 3, "gate": "FINAL_S5_HARDENED"})
        gate_summaries.append(gate)
        print(json.dumps(gate, ensure_ascii=False), flush=True)
        if gate.get("gate_passed") is True:
            final_hypotheses.append(gate)

    result = {
        "contract": "QR_PAIR_ADAPTATION_LAB_V3",
        "run_id": run_id,
        "run_root": str(run_root),
        "window_plan": plan,
        "window_reservation": reservation,
        "hardened_costs": {
            "slippage_pips_per_fill": HARDENED_SLIPPAGE_PIPS,
            "financing_pips_per_day": HARDENED_FINANCING_PIPS_PER_DAY,
        },
        "grid_size_per_pair": len(GRID),
        "phase1_screened": len(NEW_PAIRS),
        "phase2_candidates": candidates,
        "selected": selected,
        "legacy_s5_screen_positive_ignored": LEGACY_S5_SCREEN_POSITIVE,
        "legacy_screen_reuse_allowed": False,
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
        "board": board,
    }
    out = run_root / "pair_adaptation_board.json"
    write_new_json(out, result)
    print(
        f"SURVIVORS (hypothesis only): {json.dumps(final_hypotheses, ensure_ascii=False)}"
    )
    print(f"board -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
