#!/usr/bin/env python3
"""Run the preregistered USD/JPY inventory-release proof ladder.

TRAIN compares one fixed candidate with the unchanged baseline.  Only a
positive, risk-non-worse TRAIN selection advances the identical candidate to
untouched VAL and S5.  Every arm uses exact bid/ask data, BASE/STRESS costs,
both OHLC/OLHC paths and leaves end-of-replay exposure unresolved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3")
SESSION_RUNNER = REPO_ROOT / "scripts/run-virtual-market-session.py"
BASELINE_BOT = REPO_ROOT / "bots/lab_bot.py"
CANDIDATE_BOT = REPO_ROOT / "bots/inventory_release_candidate.py"
M1_MANIFEST = REPO_ROOT / "config/dojo_entry_guard_m1_sources_v1.json"
S5_MANIFEST = REPO_ROOT / "config/dojo_entry_guard_s5_sources_v1.json"
SETTLEMENT_EVENTS = {
    "CLOSE",
    "EXIT_TP",
    "EXIT_SL",
    "MARGIN_CLOSE",
    "MARGIN_CLOSEOUT",
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def ledger_metrics(session_dir: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (session_dir / "ledger.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    snapshot = load_json(session_dir / "broker_snapshot.json")
    fills: dict[str, dict[str, Any]] = {}
    settlements: list[dict[str, Any]] = []
    margin_events = 0
    release_decisions = 0
    for row in rows:
        event = row.get("event")
        payload = row.get("payload") or {}
        trade_id = payload.get("trade_id")
        if str(event).startswith("FILL") and trade_id:
            fills[str(trade_id)] = row
        if event in SETTLEMENT_EVENTS and isinstance(
            payload.get("pl_jpy"), (int, float)
        ):
            settlements.append(row)
        if str(event).startswith("MARGIN"):
            margin_events += 1
        if event == "INVENTORY_RELEASE_DECISION":
            release_decisions += 1

    pnl = [float(row["payload"]["pl_jpy"]) for row in settlements]
    gross_profit = sum(value for value in pnl if value > 0.0)
    gross_loss = -sum(value for value in pnl if value < 0.0)
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0.0
        else float("inf") if gross_profit > 0.0 else 0.0
    )
    cumulative = 0.0
    high_water = 0.0
    realized_drawdown = 0.0
    daily: dict[str, float] = defaultdict(float)
    holds: list[float] = []
    active_days: set[str] = set()
    for row, value in zip(settlements, pnl):
        cumulative += value
        high_water = max(high_water, cumulative)
        realized_drawdown = max(realized_drawdown, high_water - cumulative)
        exit_time = datetime.fromisoformat(row["ts_utc"])
        day_jst = (exit_time + timedelta(hours=9)).date().isoformat()
        daily[day_jst] += value
        active_days.add(day_jst)
        trade_id = str(row["payload"].get("trade_id") or "")
        fill = fills.get(trade_id)
        if fill is not None:
            opened = datetime.fromisoformat(fill["ts_utc"])
            holds.append((exit_time - opened).total_seconds() / 60.0)

    return {
        "settlements": len(settlements),
        "wins": sum(1 for value in pnl if value > 0.0),
        "win_rate": (
            sum(1 for value in pnl if value > 0.0) / len(pnl) if pnl else 0.0
        ),
        "net_jpy": sum(pnl),
        "profit_factor": profit_factor,
        "expectancy_jpy": sum(pnl) / len(pnl) if pnl else 0.0,
        "worst_day_jpy": min(daily.values()) if daily else 0.0,
        "realized_drawdown_jpy": realized_drawdown,
        "active_days": len(active_days),
        "average_hold_minutes": sum(holds) / len(holds) if holds else 0.0,
        "margin_events": margin_events,
        "release_decisions": release_decisions,
        "unresolved_positions": len(snapshot.get("positions") or []),
        "unresolved_orders": len(snapshot.get("orders") or []),
        "terminal_ledger_sha256": snapshot.get("ledger_sha"),
    }


def run_arm(
    *,
    output_root: Path,
    candidate_id: str,
    window_name: str,
    window: dict[str, Any],
    policy: str,
    cost_name: str,
    costs: dict[str, Any],
    intrabar: str,
) -> dict[str, Any]:
    granularity = "S5" if window_name == "S5" else "M1"
    source_manifest = S5_MANIFEST if granularity == "S5" else M1_MANIFEST
    session_dir = (
        output_root
        / window_name.lower()
        / policy.lower()
        / cost_name.lower()
        / intrabar.lower()
    )
    session_dir.mkdir(parents=True, exist_ok=False)
    bot_path = CANDIDATE_BOT if policy == "CANDIDATE" else BASELINE_BOT
    config = {
        "signal": "prev_day_extreme_fade",
        "pairs": ["USD_JPY"],
        "tp_atr": 3.0,
        "sl_pips": 25.0,
        "ceiling_min": 60,
        "max_concurrent": 1,
        "max_concurrent_per_pair": 1,
        "global_max_concurrent": 1,
        "per_pos_lev": 2.0,
        "atr_floor_pips": 0.5,
        "strategy_tag": (
            "W_PREV_DAY_EXTREME_USDJPY_INVENTORY_RELEASE_CANDIDATE"
            if policy == "CANDIDATE"
            else "W_PREV_DAY_EXTREME_USDJPY_BASELINE"
        ),
        "inventory_release_min_age_min": 30.0,
        "inventory_release_efficiency_min": 0.25,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "DOJO_BOT_CONFIG": json.dumps(
            config, sort_keys=True, separators=(",", ":")
        ),
    }
    command = [
        str(PYTHON),
        str(SESSION_RUNNER),
        "--feed",
        "replay",
        "--session-dir",
        str(session_dir),
        "--pairs",
        "USD_JPY",
        "--balance",
        "200000",
        "--from",
        window["from_utc"],
        "--to",
        window["to_utc"],
        "--source-manifest",
        str(source_manifest),
        "--granularity",
        granularity,
        "--bot-bar",
        "M1",
        "--bot-module",
        f"{bot_path}:Bot",
        "--bot-config-env",
        "DOJO_BOT_CONFIG",
        "--bars-per-second",
        "100000",
        "--state-every",
        "50000",
        "--fast-ledger",
        "--slippage-pips",
        str(costs["slippage_pips_per_fill"]),
        "--financing-pips-day",
        str(costs["financing_pips_per_day"]),
        "--leverage",
        "25",
        "--paper-proof-mode",
        "diagnostic",
        "--room-kind",
        "ai",
        "--experiment-id",
        f"dojo-improve-{candidate_id[:12]}",
        "--room-id",
        f"{window_name.lower()}-{policy.lower()}-{cost_name.lower()}-{intrabar.lower()}",
        "--candidate-id",
        candidate_id,
        "--intrabar",
        intrabar,
        "--runtime-dependency",
        str(Path(__file__).resolve()),
        "--runtime-dependency",
        str(source_manifest),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (session_dir / "runner.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"replay arm failed ({completed.returncode}): {session_dir}"
        )
    return {
        "window": window_name,
        "policy": policy,
        "cost": cost_name,
        "intrabar": intrabar,
        "command_sha256": canonical_sha256(command),
        "metrics": ledger_metrics(session_dir),
    }


def train_select(rows: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    by_key = {
        (row["policy"], row["cost"], row["intrabar"]): row["metrics"]
        for row in rows
    }
    for cost in ("BASE", "STRESS"):
        for path in ("OHLC", "OLHC"):
            baseline = by_key[("BASELINE", cost, path)]
            candidate = by_key[("CANDIDATE", cost, path)]
            if candidate["settlements"] < 30 or candidate["active_days"] < 20:
                reasons.append(f"{cost}/{path}: insufficient TRAIN sample")
            if candidate["net_jpy"] <= baseline["net_jpy"]:
                reasons.append(f"{cost}/{path}: net did not improve")
            if candidate["expectancy_jpy"] <= baseline["expectancy_jpy"]:
                reasons.append(f"{cost}/{path}: expectancy did not improve")
            if (
                candidate["worst_day_jpy"] < baseline["worst_day_jpy"]
                or candidate["realized_drawdown_jpy"]
                > baseline["realized_drawdown_jpy"]
            ):
                reasons.append(f"{cost}/{path}: risk worsened")
            if candidate["margin_events"] > baseline["margin_events"]:
                reasons.append(f"{cost}/{path}: margin events worsened")
            if (
                candidate["unresolved_positions"]
                > baseline["unresolved_positions"]
                or candidate["unresolved_orders"]
                > baseline["unresolved_orders"]
            ):
                reasons.append(f"{cost}/{path}: unresolved exposure worsened")
    return not reasons, reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    candidate_id = str(spec["candidate_id"])
    args.output_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for policy in ("BASELINE", "CANDIDATE"):
        for cost_name in ("BASE", "STRESS"):
            for intrabar in ("OHLC", "OLHC"):
                rows.append(
                    run_arm(
                        output_root=args.output_root,
                        candidate_id=candidate_id,
                        window_name="TRAIN",
                        window=spec["windows"]["TRAIN"],
                        policy=policy,
                        cost_name=cost_name,
                        costs=spec["costs"][cost_name],
                        intrabar=intrabar,
                    )
                )
    selected, reasons = train_select(rows)
    if selected:
        for window_name in ("VAL", "S5"):
            for policy in ("BASELINE", "CANDIDATE"):
                for cost_name in ("BASE", "STRESS"):
                    for intrabar in ("OHLC", "OLHC"):
                        rows.append(
                            run_arm(
                                output_root=args.output_root,
                                candidate_id=candidate_id,
                                window_name=window_name,
                                window=spec["windows"][window_name],
                                policy=policy,
                                cost_name=cost_name,
                                costs=spec["costs"][cost_name],
                                intrabar=intrabar,
                            )
                        )
    result = {
        "contract": "QR_DOJO_INVENTORY_RELEASE_REPLAY_RESULT_V1",
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
        "candidate_id": candidate_id,
        "spec_sha256": spec["spec_sha256"],
        "train_selected": selected,
        "train_rejection_reasons": reasons,
        "arms": rows,
    }
    sealed = {**result, "result_sha256": canonical_sha256(result)}
    (args.output_root / "summary_all.json").write_text(
        json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
