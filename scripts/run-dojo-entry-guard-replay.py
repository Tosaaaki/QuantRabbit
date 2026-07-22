#!/usr/bin/env python3
"""Run the preregistered isolated DOJO entry-guard replay comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "config/dojo_entry_guard_replay_v1.json"
DEFAULT_OUT = ROOT / "research/data/dojo_entry_guard_replay_v1"
SETTLED_EVENTS = {"EXIT_TP", "EXIT_SL", "CLOSE", "MARGIN_CLOSEOUT"}
ZERO_SHA = "0" * 64


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources(spec: dict[str, Any]) -> None:
    for granularity, source in (spec.get("sources") or {}).items():
        root = Path(str(source.get("root")))
        if not root.is_dir():
            raise RuntimeError(f"source root is missing: {root}")
        manifest_path = ROOT / str(source.get("manifest"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("contract") != "QR_VIRTUAL_REPLAY_SOURCE_MANIFEST_V1"
            or manifest.get("granularity") != granularity
        ):
            raise RuntimeError(f"source manifest contract mismatch: {manifest_path}")
        manifest_rows = {
            (row.get("path"), row.get("sha256"))
            for row in manifest.get("files") or []
        }
        for row in source.get("files") or []:
            path = Path(row["path"])
            if not path.is_file():
                raise RuntimeError(f"source file is missing: {path}")
            actual = sha256_file(path)
            if actual != row.get("sha256"):
                raise RuntimeError(f"source hash mismatch: {path}")
            if (row.get("path"), row.get("sha256")) not in manifest_rows:
                raise RuntimeError(f"source is not bound by manifest: {path}")


def read_ledger(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    previous = ZERO_SHA
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            body = {key: row[key] for key in ("ts_utc", "event", "payload", "prev_sha")}
            expected = hashlib.sha256(canonical_bytes(body)).hexdigest()
            if row.get("prev_sha") != previous or row.get("sha") != expected:
                raise RuntimeError(f"ledger chain mismatch at {path}:{line_number}")
            previous = row["sha"]
            rows.append(row)
    if not rows:
        raise RuntimeError(f"empty ledger: {path}")
    return rows, previous


def metrics(session: Path) -> dict[str, Any]:
    rows, terminal_sha = read_ledger(session / "ledger.jsonl")
    snapshot = json.loads((session / "broker_snapshot.json").read_text())
    if snapshot.get("ledger_sha") != terminal_sha:
        raise RuntimeError(f"snapshot checkpoint mismatch: {session}")
    if rows[-1]["event"] != "SESSION_STOP":
        raise RuntimeError(f"replay has no terminal SESSION_STOP: {session}")

    outcomes = []
    daily: dict[str, float] = defaultdict(float)
    fills = 0
    missing_context = 0
    invalid_context_hash = 0
    breaker_events = 0
    for row in rows:
        event = row["event"]
        payload = row.get("payload") or {}
        if event.startswith("FILL_"):
            fills += 1
            context = payload.get("entry_context")
            digest = payload.get("entry_context_sha256")
            if not isinstance(context, dict):
                missing_context += 1
            elif hashlib.sha256(canonical_bytes(context)).hexdigest() != digest:
                invalid_context_hash += 1
        if event == "ENTRY_CIRCUIT_BREAKER":
            breaker_events += 1
        if event in SETTLED_EVENTS and payload.get("pl_jpy") is not None:
            value = float(payload["pl_jpy"])
            outcomes.append(value)
            quote_ts = str((payload.get("quote") or {}).get("ts") or row["ts_utc"])
            daily[quote_ts[:10]] += value

    wins = [value for value in outcomes if value > 0]
    losses = [value for value in outcomes if value <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    equity = peak = 200_000.0
    max_drawdown = 0.0
    for value in outcomes:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    net = sum(outcomes)
    return {
        "settled": len(outcomes),
        "wins": len(wins),
        "losses": len(losses),
        "net_jpy": round(net, 2),
        "profit_factor": gross_win / gross_loss if gross_loss else None,
        "expectancy_jpy": net / len(outcomes) if outcomes else None,
        "win_rate": len(wins) / len(outcomes) if outcomes else None,
        "max_realized_drawdown_jpy": round(max_drawdown, 2),
        "worst_day_jpy": round(min(daily.values()), 2) if daily else None,
        "active_days": len(daily),
        "fills": fills,
        "fills_missing_entry_context": missing_context,
        "fills_invalid_entry_context_hash": invalid_context_hash,
        "entry_circuit_breaker_events": breaker_events,
        "open_positions_at_end": len(snapshot.get("positions") or []),
        "resting_orders_at_end": len(snapshot.get("orders") or []),
        "end_of_replay_forced_close_used": False,
        "ledger_terminal_sha256": terminal_sha,
    }


def case_id(candidate_id: str, window_id: str, intrabar: str) -> str:
    return f"{candidate_id}__{window_id.lower()}__{intrabar.lower()}"


def run_case(
    *,
    spec_path: Path,
    spec: dict[str, Any],
    out_root: Path,
    candidate: dict[str, Any],
    window: dict[str, Any],
    intrabar: str,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    identifier = case_id(candidate_id, window["window_id"], intrabar)
    session = out_root / identifier
    result_path = session / "metrics.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    if session.exists() and any(session.iterdir()):
        raise RuntimeError(f"partial replay requires explicit review: {session}")
    session.mkdir(parents=True, exist_ok=True)

    config = {**spec["baseline"], **(candidate.get("overrides") or {})}
    env = dict(os.environ)
    env["DOJO_BOT_CONFIG"] = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    env["PYTHONPATH"] = str(ROOT / "src")
    granularity = str(window["granularity"])
    source_root = Path(spec["sources"][granularity]["root"])
    source_manifest = ROOT / spec["sources"][granularity]["manifest"]
    command = [
        sys.executable,
        str(ROOT / "scripts/run-virtual-market-session.py"),
        "--feed", "replay",
        "--session-dir", str(session),
        "--pairs", "USD_JPY",
        "--balance", "200000",
        "--from", str(window["from_utc"]),
        "--to", str(window["to_utc"]),
        "--corpus-root", str(source_root),
        "--source-manifest", str(source_manifest),
        "--granularity", granularity,
        "--bars-per-second", "100000",
        "--state-every", "100000",
        "--fast-ledger",
        "--intrabar", intrabar,
        "--bot-module", str(ROOT / "bots/lab_bot.py") + ":Bot",
        "--bot-config-env", "DOJO_BOT_CONFIG",
        "--slippage-pips", str(float(window["slippage_pips_per_fill"])),
        "--financing-pips-day", str(float(window["financing_pips_per_day"])),
        "--leverage", "25",
        "--paper-proof-mode", "diagnostic",
        "--room-kind", "single_strategy",
        "--experiment-id", str(spec["experiment_id"]),
        "--room-id", identifier,
        "--candidate-id", candidate_id,
        "--runtime-dependency", str(spec_path),
        "--runtime-dependency", str(Path(__file__).resolve()),
    ]
    if granularity == "S5":
        command += ["--bot-bar", "M1"]
    print(json.dumps({"status": "RUNNING", "case": identifier}), flush=True)
    proc = subprocess.run(command, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"replay failed {identifier}: {(proc.stderr or proc.stdout)[-1200:]}"
        )
    result = {
        "case_id": identifier,
        "candidate_id": candidate_id,
        "window_id": window["window_id"],
        "granularity": granularity,
        "intrabar": intrabar,
        "costs": {
            "slippage_pips_per_fill": float(window["slippage_pips_per_fill"]),
            "financing_pips_per_day": float(window["financing_pips_per_day"]),
        },
        "metrics": metrics(session),
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": "DONE", **result}, ensure_ascii=False), flush=True)
    return result


def pf_value(value: float | None) -> float:
    return math.inf if value is None else float(value)


def evaluate(spec: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {
        (row["candidate_id"], row["window_id"], row["intrabar"]): row["metrics"]
        for row in results
    }
    decisions = []
    proof_windows = set(spec["evaluation"]["proof_windows"])
    candidates = [
        row["candidate_id"]
        for row in spec["candidates"]
        if row["candidate_id"] != "baseline"
    ]
    for candidate_id in candidates:
        comparisons = []
        all_pass = True
        any_net_failure = False
        for (row_candidate, window_id, intrabar), candidate_metrics in sorted(
            by_key.items()
        ):
            if row_candidate != candidate_id or window_id not in proof_windows:
                continue
            baseline = by_key[("baseline", window_id, intrabar)]
            retention = (
                candidate_metrics["settled"] / baseline["settled"]
                if baseline["settled"]
                else 0.0
            )
            checks = {
                "net_gt_baseline": candidate_metrics["net_jpy"] > baseline["net_jpy"],
                "pf_gte_baseline": pf_value(candidate_metrics["profit_factor"])
                >= pf_value(baseline["profit_factor"]),
                "max_drawdown_lte_baseline": candidate_metrics[
                    "max_realized_drawdown_jpy"
                ] <= baseline["max_realized_drawdown_jpy"],
                "worst_day_gte_baseline": candidate_metrics["worst_day_jpy"]
                >= baseline["worst_day_jpy"],
                "trade_retention_gte_50pct": retention >= 0.5,
                "entry_context_complete": candidate_metrics[
                    "fills_missing_entry_context"
                ] == 0
                and candidate_metrics["fills_invalid_entry_context_hash"] == 0,
            }
            all_pass = all_pass and all(checks.values())
            any_net_failure = any_net_failure or not checks["net_gt_baseline"]
            comparisons.append(
                {
                    "window_id": window_id,
                    "intrabar": intrabar,
                    "baseline": baseline,
                    "candidate": candidate_metrics,
                    "trade_retention": retention,
                    "checks": checks,
                }
            )
        decision = "ACCEPT" if all_pass else ("REJECT" if any_net_failure else "TEST")
        decisions.append(
            {
                "candidate_id": candidate_id,
                "decision": decision,
                "comparisons": comparisons,
            }
        )
    return {
        "contract": "QR_DOJO_ENTRY_GUARD_REPLAY_RESULT_V1",
        "experiment_id": spec["experiment_id"],
        "decisions": decisions,
        "guarantee": False,
        "live_permission": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--case", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text())
    if spec.get("contract") != "QR_DOJO_ENTRY_GUARD_REPLAY_V1":
        raise RuntimeError("unsupported replay specification")
    if spec.get("order_authority") != "NONE" or spec.get("live_permission") is not False:
        raise RuntimeError("replay authority boundary is invalid")
    verify_sources(spec)
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.jobs < 1 or args.jobs > 3:
        raise RuntimeError("--jobs must be in 1..3")
    cases = []
    for candidate in spec["candidates"]:
        for window in spec["windows"]:
            for intrabar in window["intrabar_paths"]:
                identifier = case_id(
                    candidate["candidate_id"], window["window_id"], intrabar
                )
                if args.case and args.case != identifier:
                    continue
                cases.append(
                    {
                        "spec_path": spec_path,
                        "spec": spec,
                        "out_root": out_root,
                        "candidate": candidate,
                        "window": window,
                        "intrabar": intrabar,
                    }
                )
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = list(executor.map(lambda kwargs: run_case(**kwargs), cases))
    if args.case:
        return 0
    result = evaluate(spec, results)
    result["spec_sha256"] = sha256_file(spec_path)
    result["results"] = results
    report = out_root / "result.json"
    report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": "COMPLETE", "report": str(report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
