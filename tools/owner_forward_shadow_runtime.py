#!/usr/bin/env python3
"""Resident, owner-attested, OANDA GET-only fast-bot forward collector.

This process is intentionally not a trading wrapper.  Its subprocess allowlist
contains only the read-only broker/chart commands and the deterministic shadow
and outcome programs.  Runtime evidence is kept outside the repository in a
commit-addressed cohort directory.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import plistlib
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL = "com.quantrabbit.owner-forward-shadow"
DEFAULT_STATE_ROOT = Path.home() / ".codex" / "state" / "quantrabbit" / "owner-forward-shadow-v1"
DEFAULT_EURUSD_POLICY_POINTER = Path.home() / ".codex" / "state" / "quantrabbit" / "eurusd-outcome-learning-v1" / "current.json"
DEFAULT_ENV_FILE = Path("/Users/tossaki/App/QuantRabbit/.env.local")
SHADOW_PAIRS = ("EUR_USD", "USD_JPY")
PROFIT_HOLDOUT_POLICY_PATH = Path("config/fast_bot_profit_holdout_v3.json")
# Fetch the slowest views first and the latency-sensitive M1 view last.  The
# pair-chart command processes each pair's requested timeframes in order, so
# leading with M1 can make an otherwise contiguous candle stale before the
# deterministic shock guard gets to evaluate it.
SLOW_TIMEFRAMES = "D,H4,H1,M30,M15,M5,M1"
# One pair-chart subprocess performs one bounded OANDA GET per pair/timeframe.
# The broker client may consume up to roughly 30 seconds on a slow request, so
# the process budget scales with the exact request count plus fixed startup and
# atomic-write overhead.  This replaces the unrelated generic 240-second cap;
# it does not relax candle freshness or completeness.
PAIR_CHART_GET_REQUEST_BUDGET_SECONDS = 30.0
PAIR_CHART_PROCESS_OVERHEAD_SECONDS = 30.0
STOP = False

SOURCE_BUNDLE_PATHS = (
    Path("tools/owner_forward_shadow_runtime.py"),
    Path("tools/run_fast_bot_corrective_challenger.py"),
    Path("tools/run_fast_bot_knowledge.py"),
    Path("tools/run_fast_bot_shock_follow.py"),
    Path("tools/run_fast_bot_shock_guard.py"),
    Path("tools/run_fast_bot_profit_holdout.py"),
    Path("tools/audit_fast_bot_resident_profit_candidates.py"),
    Path("tools/run_eurusd_outcome_learning.py"),
    Path("scripts/run-fast-bot-shadow.py"),
    Path("scripts/resolve-fast-bot-shadow-outcomes.py"),
    Path("src/quant_rabbit/cli.py"),
    Path("src/quant_rabbit/guardian_observation.py"),
    Path("src/quant_rabbit/fast_bot.py"),
    Path("src/quant_rabbit/fast_bot_corrective_challenger.py"),
    Path("src/quant_rabbit/fast_bot_knowledge.py"),
    Path("src/quant_rabbit/fast_bot_shock_follow.py"),
    Path("src/quant_rabbit/fast_bot_shock_guard.py"),
    Path("src/quant_rabbit/fast_bot_profit_holdout.py"),
    Path("src/quant_rabbit/fast_bot_profit_candidate_audit.py"),
    Path("src/quant_rabbit/fast_bot_profitability_gate.py"),
    Path("src/quant_rabbit/analysis/chart_reader.py"),
    Path("src/quant_rabbit/eurusd_outcome_learning.py"),
    Path("src/quant_rabbit/fast_bot_truth.py"),
    Path("src/quant_rabbit/broker/oanda.py"),
    Path("config/oanda_spread_calibration_v1.json"),
    Path("config/oanda_spread_calibration_source_v1.json.gz"),
    Path("config/fast_bot_corrective_challenger_v1.json"),
    Path("config/fast_bot_shock_follow_v1.json"),
    Path("config/fast_bot_shock_guard_v1.json"),
    Path("config/fast_bot_profit_holdout_v1.json"),
    Path("config/fast_bot_profit_holdout_v2.json"),
    PROFIT_HOLDOUT_POLICY_PATH,
    Path("config/eurusd_learned_policy_v1.json"),
)


class RuntimeBlocked(RuntimeError):
    """A fail-closed release, authority, or collection failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _git(*args: str, repo_root: Path = REPO_ROOT) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def source_bundle(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    files = {str(path): sha256_file(repo_root / path) for path in SOURCE_BUNDLE_PATHS}
    return {"files": files, "sha256": canonical_sha(files)}


def verify_release(
    *,
    expected_commit: str,
    expected_source_sha256: str,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    if len(expected_commit) != 40 or any(c not in "0123456789abcdef" for c in expected_commit):
        raise RuntimeBlocked("EXPECTED_COMMIT_INVALID")
    if len(expected_source_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_source_sha256):
        raise RuntimeBlocked("EXPECTED_SOURCE_SHA256_INVALID")
    top = Path(_git("rev-parse", "--show-toplevel", repo_root=repo_root)).resolve()
    if top != repo_root.resolve():
        raise RuntimeBlocked("REPO_TOP_MISMATCH")
    head = _git("rev-parse", "HEAD", repo_root=repo_root)
    if head != expected_commit:
        raise RuntimeBlocked("COMMIT_DRIFT")
    if _git("status", "--porcelain", "--untracked-files=all", repo_root=repo_root):
        raise RuntimeBlocked("WORKTREE_DIRTY")
    bundle = source_bundle(repo_root)
    if bundle["sha256"] != expected_source_sha256:
        raise RuntimeBlocked("SOURCE_BUNDLE_DRIFT")
    calibration = read_object(repo_root / "config/oanda_spread_calibration_v1.json")
    if calibration.get("broker_write_performed") is not False:
        raise RuntimeBlocked("CALIBRATION_WRITE_PROVENANCE_INVALID")
    if calibration.get("broker_http_methods_used") != ["GET"]:
        raise RuntimeBlocked("CALIBRATION_HTTP_METHODS_INVALID")
    return {
        "repo_root": str(repo_root),
        "branch": _git("branch", "--show-current", repo_root=repo_root),
        "commit": head,
        "git_tree": _git("rev-parse", "HEAD^{tree}", repo_root=repo_root),
        "source_bundle_sha256": bundle["sha256"],
        "source_files": bundle["files"],
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_sha256": sha256_file(Path(sys.executable).resolve()),
        "calibration_sha256": calibration.get("calibration_sha256"),
        "calibration_source_evidence_sha256": calibration.get("source_evidence_sha256"),
    }


def cohort_root(state_root: Path, expected_commit: str, expected_source_sha256: str) -> Path:
    return state_root / f"{expected_commit}-{expected_source_sha256[:16]}"


def child_environment(env_file: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "QR_OANDA_ENV_FILE": str(env_file),
            "QR_LIVE_ENABLED": "0",
            "AI_ORDER_AUTHORITY": "NONE",
            "QR_AUTOTRADE_LOCK_HELD": "0",
        }
    )
    env.pop("QR_AUTOTRADE_LOCK_OWNER_TOKEN", None)
    return env


def run_command(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    repo_root: Path = REPO_ROOT,
    timeout: float = 240.0,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(argv),
            cwd=repo_root,
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        label = _command_label(argv)
        raise RuntimeBlocked(
            f"COMMAND_TIMEOUT:{label}:budget_seconds={timeout:g}"
        ) from exc
    result = {
        "argv": list(argv),
        "returncode": completed.returncode,
        "wall_seconds": round(time.monotonic() - started, 6),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        raise RuntimeBlocked(
            f"COMMAND_FAILED:{Path(argv[0]).name}:{completed.returncode}:"
            f"{completed.stderr[-300:]}"
        )
    return result


def _command_label(argv: Sequence[str]) -> str:
    values = [str(value) for value in argv]
    if "-m" in values:
        module_index = values.index("-m") + 1
        if module_index < len(values):
            module = values[module_index]
            command = values[module_index + 1] if module_index + 1 < len(values) else ""
            return f"{module}:{command}" if command else module
    if len(values) > 1:
        return Path(values[1]).name
    return Path(values[0]).name if values else "unknown"


def _pair_chart_timeout_seconds(
    *, pairs: Sequence[str] = SHADOW_PAIRS, timeframes: str = SLOW_TIMEFRAMES
) -> float:
    timeframe_count = len([value for value in timeframes.split(",") if value])
    if not pairs or timeframe_count <= 0:
        raise ValueError("pair-chart request scope must be non-empty")
    return (
        len(pairs) * timeframe_count * PAIR_CHART_GET_REQUEST_BUDGET_SECONDS
        + PAIR_CHART_PROCESS_OVERHEAD_SECONDS
    )


def _json_stdout(result: Mapping[str, Any]) -> dict[str, Any]:
    text = str(result.get("stdout_tail") or "").strip()
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _pair_charts_refresh_due(path: Path, *, now: datetime | None = None) -> bool:
    payload = read_object(path)
    raw = payload.get("generated_at_utc")
    if not isinstance(raw, str):
        return True
    try:
        generated = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return True
    if generated.tzinfo is None:
        return True
    clock = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age = (clock - generated.astimezone(timezone.utc)).total_seconds()
    return age < 0.0 or age >= 60.0


def _latest_quote_timestamp(snapshot: Mapping[str, Any]) -> str | None:
    quotes = snapshot.get("quotes")
    if not isinstance(quotes, Mapping):
        return None
    values = [
        str(item.get("timestamp_utc"))
        for item in quotes.values()
        if isinstance(item, Mapping) and item.get("timestamp_utc")
    ]
    return max(values) if values else None


def _latest_bar_timestamp(packet: Mapping[str, Any]) -> str | None:
    values: list[str] = []
    charts = packet.get("charts")
    if not isinstance(charts, list):
        return None
    for chart in charts:
        if not isinstance(chart, Mapping):
            continue
        for view in chart.get("views") or []:
            if not isinstance(view, Mapping):
                continue
            candles = view.get("recent_candles")
            if isinstance(candles, list):
                for candle in candles[-1:]:
                    if isinstance(candle, Mapping):
                        raw = candle.get("t") or candle.get("timestamp_utc") or candle.get("time")
                        if raw:
                            values.append(str(raw))
    return max(values) if values else None


def _write_zero_authority_inputs(root: Path) -> None:
    generated = utc_now()
    atomic_json(
        root / "state" / "guardian_events.json",
        {
            "generated_at_utc": generated,
            "events": [],
            "external_order_attempts": 0,
            "external_orders": 0,
            "broker_mutation_allowed": False,
        },
    )
    atomic_json(
        root / "state" / "ai_regime_supervision.json",
        {
            "generated_at_utc": generated,
            "status": "NO_ACTIVE_SUPERVISION_RECEIPT",
            "execution_authority": "NONE",
            "live_permission": False,
            "broker_mutation_allowed": False,
        },
    )


def _base_status(
    *,
    manifest: Mapping[str, Any],
    root: Path,
    started_at: str,
    counters: Mapping[str, int],
    restart_count: int,
) -> dict[str, Any]:
    return {
        "contract": "QR_OWNER_FORWARD_SHADOW_RESIDENT_V1",
        "schema_version": 1,
        "pid": os.getpid(),
        "started_at_utc": started_at,
        "heartbeat_at_utc": utc_now(),
        "run_state": "STARTING",
        "restart_count": restart_count,
        "executable": manifest["python_executable"],
        "executable_sha256": manifest["python_executable_sha256"],
        "source_commit": manifest["commit"],
        "source_bundle_sha256": manifest["source_bundle_sha256"],
        "execution_authority": "NONE",
        "broker_http_methods_allowed": ["GET"],
        "broker_mutation_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "gateway_invocations": 0,
        "manual_tagless_positions_policy": "NO_TOUCH",
        "existing_tp_sl_policy": "NO_TOUCH",
        "live_permission": False,
        "promotion_allowed": False,
        "state_path": str(root / "state" / "status.json"),
        "shadow_ledger_path": str(root / "ledgers" / "fast_bot_shadow_ledger.jsonl"),
        "outcome_ledger_path": str(root / "ledgers" / "fast_bot_outcome_ledger.jsonl"),
        "scorecard_path": str(root / "scorecard" / "fast_bot_scorecard.json"),
        "profit_holdout_selected_ledger_path": str(root / "ledgers" / "fast_bot_profit_holdout_signal_ledger.jsonl"),
        "profit_holdout_outcome_ledger_path": str(root / "ledgers" / "fast_bot_profit_holdout_outcome_ledger.jsonl"),
        "profit_holdout_scorecard_path": str(root / "scorecard" / "fast_bot_profit_holdout_scorecard.json"),
        "last_profit_holdout_selection_result": {},
        "last_profit_holdout_outcome_result": {},
        "last_profit_holdout_scorecard_result": {},
        "corrective_challenger_ledger_path": str(root / "ledgers" / "fast_bot_corrective_challenger_ledger.jsonl"),
        "corrective_challenger_scorecard_path": str(root / "scorecard" / "fast_bot_corrective_challenger_scorecard.json"),
        "learning_episode_ledger_path": str(root / "ledgers" / "fast_bot_learning_episode_ledger.jsonl"),
        "knowledge_ledger_path": str(root / "ledgers" / "fast_bot_knowledge_ledger.jsonl"),
        "learning_scorecard_path": str(root / "scorecard" / "fast_bot_learning_scorecard.json"),
        "shock_follow_signal_ledger_path": str(root / "ledgers" / "fast_bot_shock_follow_signal_ledger.jsonl"),
        "shock_follow_outcome_ledger_path": str(root / "ledgers" / "fast_bot_shock_follow_outcome_ledger.jsonl"),
        "shock_follow_scorecard_path": str(root / "scorecard" / "fast_bot_shock_follow_scorecard.json"),
        "shock_guard_state_path": str(root / "state" / "fast_bot_shock_guard_state.json"),
        "shock_guard_decision_ledger_path": str(root / "ledgers" / "fast_bot_shock_guard_decision_ledger.jsonl"),
        "shock_guard_scorecard_path": str(root / "scorecard" / "fast_bot_shock_guard_scorecard.json"),
        "eurusd_learned_policy_pointer_path": str(DEFAULT_EURUSD_POLICY_POINTER),
        "eurusd_learned_decision_ledger_path": str(root / "ledgers" / "eurusd_learned_policy_prospective_decision_ledger.jsonl"),
        "eurusd_learned_outcome_ledger_path": str(root / "ledgers" / "eurusd_learned_policy_prospective_outcome_ledger.jsonl"),
        "eurusd_learned_scorecard_path": str(root / "scorecard" / "eurusd_learned_policy_prospective_scorecard.json"),
        "baseline_control_observation_only": True,
        "worst_lane_new_entry_policy": "NO_AUTOMATIC_ADOPTION_CORRECTIVE_ARMS_DIAGNOSTIC_ONLY",
        "counters": dict(counters),
    }


def run_cycle(
    *,
    root: Path,
    env: Mapping[str, str],
    state: dict[str, Any],
    command_runner: Callable[..., dict[str, Any]] = run_command,
) -> dict[str, Any]:
    py = str(Path(sys.executable).resolve())
    paths = {
        "snapshot": root / "market" / "broker_snapshot.json",
        "charts": root / "market" / "eurusd_usdjpy_pair_charts.json",
        "events": root / "state" / "guardian_events.json",
        "supervision": root / "state" / "ai_regime_supervision.json",
        "regime": root / "state" / "hierarchical_bot_regime.json",
        "shadow": root / "state" / "fast_bot_shadow.json",
        "shadow_ledger": root / "ledgers" / "fast_bot_shadow_ledger.jsonl",
        "outcome_ledger": root / "ledgers" / "fast_bot_outcome_ledger.jsonl",
        "scorecard": root / "scorecard" / "fast_bot_scorecard.json",
        "profit_holdout_selection": root / "state" / "fast_bot_profit_holdout_selection.json",
        "profit_holdout_selected_ledger": root / "ledgers" / "fast_bot_profit_holdout_signal_ledger.jsonl",
        "profit_holdout_decision_ledger": root / "ledgers" / "fast_bot_profit_holdout_decision_ledger.jsonl",
        "profit_holdout_outcome_ledger": root / "ledgers" / "fast_bot_profit_holdout_outcome_ledger.jsonl",
        "profit_holdout_truth_scorecard": root / "scorecard" / "fast_bot_profit_holdout_truth_scorecard.json",
        "profit_holdout_scorecard": root / "scorecard" / "fast_bot_profit_holdout_scorecard.json",
        "challenger_ledger": root / "ledgers" / "fast_bot_corrective_challenger_ledger.jsonl",
        "challenger_scorecard": root / "scorecard" / "fast_bot_corrective_challenger_scorecard.json",
        "learning_episode_ledger": root / "ledgers" / "fast_bot_learning_episode_ledger.jsonl",
        "knowledge_ledger": root / "ledgers" / "fast_bot_knowledge_ledger.jsonl",
        "learning_scorecard": root / "scorecard" / "fast_bot_learning_scorecard.json",
        "shock_signal_ledger": root / "ledgers" / "fast_bot_shock_follow_signal_ledger.jsonl",
        "shock_outcome_ledger": root / "ledgers" / "fast_bot_shock_follow_outcome_ledger.jsonl",
        "shock_scorecard": root / "scorecard" / "fast_bot_shock_follow_scorecard.json",
        "shock_guard_state": root / "state" / "fast_bot_shock_guard_state.json",
        "shock_guard_decisions": root / "ledgers" / "fast_bot_shock_guard_decision_ledger.jsonl",
        "shock_guard_scorecard": root / "scorecard" / "fast_bot_shock_guard_scorecard.json",
        "guarded_shadow": root / "state" / "fast_bot_shock_guarded_shadow.json",
        "eurusd_decision_ledger": root / "ledgers" / "eurusd_learned_policy_prospective_decision_ledger.jsonl",
        "eurusd_outcome_ledger": root / "ledgers" / "eurusd_learned_policy_prospective_outcome_ledger.jsonl",
        "eurusd_scorecard": root / "scorecard" / "eurusd_learned_policy_prospective_scorecard.json",
        "report": root / "reports" / "fast_bot_shadow_report.md",
        "profit_holdout_selection_report": root / "reports" / "fast_bot_profit_holdout_selection_report.md",
        "profit_holdout_scorecard_report": root / "reports" / "fast_bot_profit_holdout_scorecard_report.md",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    snapshot_result = command_runner(
        [py, "-m", "quant_rabbit.cli", "broker-snapshot", "--pairs", ",".join(SHADOW_PAIRS), "--output", str(paths["snapshot"])],
        env=env,
    )
    refreshed = _pair_charts_refresh_due(paths["charts"])
    bot_result: dict[str, Any] = {}
    pair_chart_result = state.get("last_pair_chart_result") or {}
    if refreshed:
        pair_chart_result = command_runner(
            [py, "-m", "quant_rabbit.cli", "pair-charts", "--pairs", ",".join(SHADOW_PAIRS), "--timeframes", SLOW_TIMEFRAMES, "--count", "120", "--output", str(paths["charts"]), "--report", str(root / "reports" / "eurusd_usdjpy_pair_charts_report.md"), "--require-complete"],
            env=env,
            timeout=_pair_chart_timeout_seconds(),
        )
        command_runner(
            [py, "-m", "quant_rabbit.cli", "broker-snapshot", "--pairs", ",".join(SHADOW_PAIRS), "--output", str(paths["snapshot"])],
            env=env,
        )
        bot_result = _json_stdout(
            command_runner(
                [py, str(REPO_ROOT / "scripts/run-fast-bot-shadow.py"), "--fast-pair-charts", str(paths["charts"]), "--slow-pair-charts", str(paths["charts"]), "--broker-snapshot", str(paths["snapshot"]), "--guardian-events", str(paths["events"]), "--ai-supervision", str(paths["supervision"]), "--regime-output", str(paths["regime"]), "--output", str(paths["shadow"]), "--ledger", str(paths["shadow_ledger"]), "--report", str(paths["report"])],
                env=env,
            )
        )
        shock_guard_result = _json_stdout(
            command_runner(
                [
                    py,
                    str(REPO_ROOT / "tools/run_fast_bot_shock_guard.py"),
                    "--pair-charts", str(paths["charts"]),
                    "--shadow", str(paths["shadow"]),
                    "--config", str(REPO_ROOT / "config/fast_bot_shock_guard_v1.json"),
                    "--state", str(paths["shock_guard_state"]),
                    "--decision-ledger", str(paths["shock_guard_decisions"]),
                    "--scorecard", str(paths["shock_guard_scorecard"]),
                    "--output", str(paths["guarded_shadow"]),
                ],
                env=env,
            )
        )
        # Progressive preflight consumes last_shadow_result.shadow_output.  The
        # baseline control ledger remains unchanged, while only the separately
        # sealed guarded candidate can reach later promotion gates.
        bot_result = {
            **bot_result,
            "baseline_shadow_output": bot_result.get("shadow_output"),
            "shadow_output": shock_guard_result.get("shadow_output"),
            "shock_guard": shock_guard_result,
        }
        profit_holdout_selection_result = _json_stdout(
            command_runner(
                [
                    py,
                    str(REPO_ROOT / "tools/run_fast_bot_profit_holdout.py"),
                    "select",
                    "--shadow", str(paths["shadow"]),
                    "--raw-signal-ledger", str(paths["shadow_ledger"]),
                    "--policy", str(REPO_ROOT / PROFIT_HOLDOUT_POLICY_PATH),
                    "--selected-ledger", str(paths["profit_holdout_selected_ledger"]),
                    "--decision-ledger", str(paths["profit_holdout_decision_ledger"]),
                    "--output", str(paths["profit_holdout_selection"]),
                    "--report", str(paths["profit_holdout_selection_report"]),
                ],
                env=env,
            )
        )
    else:
        shock_guard_result = state.get("last_shock_guard_result") or {}
        profit_holdout_selection_result = state.get("last_profit_holdout_selection_result") or {}

    outcome_result = _json_stdout(
        command_runner(
            [py, str(REPO_ROOT / "scripts/resolve-fast-bot-shadow-outcomes.py"), "--shadow-ledger", str(paths["shadow_ledger"]), "--outcome-ledger", str(paths["outcome_ledger"]), "--scorecard", str(paths["scorecard"])],
            env=env,
        )
    )
    profit_holdout_outcome_result = _json_stdout(
        command_runner(
            [
                py,
                str(REPO_ROOT / "scripts/resolve-fast-bot-shadow-outcomes.py"),
                "--shadow-ledger", str(paths["profit_holdout_selected_ledger"]),
                "--outcome-ledger", str(paths["profit_holdout_outcome_ledger"]),
                "--scorecard", str(paths["profit_holdout_truth_scorecard"]),
            ],
            env=env,
        )
    )
    profit_holdout_scorecard_result = _json_stdout(
        command_runner(
            [
                py,
                str(REPO_ROOT / "tools/run_fast_bot_profit_holdout.py"),
                "evaluate",
                "--policy", str(REPO_ROOT / PROFIT_HOLDOUT_POLICY_PATH),
                "--raw-signal-ledger", str(paths["shadow_ledger"]),
                "--selected-ledger", str(paths["profit_holdout_selected_ledger"]),
                "--decision-ledger", str(paths["profit_holdout_decision_ledger"]),
                "--outcome-ledger", str(paths["profit_holdout_outcome_ledger"]),
                "--truth-scorecard", str(paths["profit_holdout_truth_scorecard"]),
                "--output", str(paths["profit_holdout_scorecard"]),
                "--report", str(paths["profit_holdout_scorecard_report"]),
            ],
            env=env,
        )
    )
    challenger_result = _json_stdout(
        command_runner(
            [
                py,
                str(REPO_ROOT / "tools/run_fast_bot_corrective_challenger.py"),
                "--shadow-ledger", str(paths["shadow_ledger"]),
                "--outcome-ledger", str(paths["outcome_ledger"]),
                "--challenger-ledger", str(paths["challenger_ledger"]),
                "--scorecard", str(paths["challenger_scorecard"]),
                "--config", str(REPO_ROOT / "config/fast_bot_corrective_challenger_v1.json"),
                "--max-due", "12",
            ],
            env=env,
        )
    )
    knowledge_result = _json_stdout(
        command_runner(
            [
                py,
                str(REPO_ROOT / "tools/run_fast_bot_knowledge.py"),
                "--shadow-ledger", str(paths["shadow_ledger"]),
                "--outcome-ledger", str(paths["outcome_ledger"]),
                "--challenger-ledger", str(paths["challenger_ledger"]),
                "--config", str(REPO_ROOT / "config/fast_bot_corrective_challenger_v1.json"),
                "--episode-ledger", str(paths["learning_episode_ledger"]),
                "--knowledge-ledger", str(paths["knowledge_ledger"]),
                "--scorecard", str(paths["learning_scorecard"]),
            ],
            env=env,
        )
    )
    shock_follow_result = _json_stdout(
        command_runner(
            [
                py,
                str(REPO_ROOT / "tools/run_fast_bot_shock_follow.py"),
                "--pair-charts", str(paths["charts"]),
                "--broker-snapshot", str(paths["snapshot"]),
                "--signal-ledger", str(paths["shock_signal_ledger"]),
                "--outcome-ledger", str(paths["shock_outcome_ledger"]),
                "--scorecard", str(paths["shock_scorecard"]),
                "--corrective-ledger", str(paths["challenger_ledger"]),
                "--config", str(REPO_ROOT / "config/fast_bot_shock_follow_v1.json"),
                "--max-due", "12",
            ],
            env=env,
        )
    )
    if DEFAULT_EURUSD_POLICY_POINTER.is_file():
        eurusd_learning_result = _json_stdout(
            command_runner(
                [
                    py,
                    str(REPO_ROOT / "tools/run_eurusd_outcome_learning.py"),
                    "observe",
                    "--current-pointer", str(DEFAULT_EURUSD_POLICY_POINTER),
                    "--config", str(REPO_ROOT / "config/eurusd_learned_policy_v1.json"),
                    "--shock-signal-ledger", str(paths["shock_signal_ledger"]),
                    "--shock-outcome-ledger", str(paths["shock_outcome_ledger"]),
                    "--decision-ledger", str(paths["eurusd_decision_ledger"]),
                    "--prospective-outcome-ledger", str(paths["eurusd_outcome_ledger"]),
                    "--prospective-scorecard", str(paths["eurusd_scorecard"]),
                ],
                env=env,
            )
        )
    else:
        eurusd_learning_result = {
            "status": "POLICY_POINTER_NOT_AVAILABLE_NO_TRADE",
            "execution_authority": "NONE",
            "broker_http_methods_allowed": ["GET"],
            "broker_mutation": False,
            "external_order_attempts": 0,
            "external_orders": 0,
            "automatic_adoption_allowed": False,
            "promotion_allowed": False,
            "live_permission": False,
        }
    snapshot = read_object(paths["snapshot"])
    fast_packet = read_object(paths["charts"])
    scorecard = read_object(paths["scorecard"])
    state["event_count"] = int(state.get("event_count", 0)) + 1
    state["proposal_count"] = int(state.get("proposal_count", 0)) + int(bot_result.get("signal_count") or 0)
    state["virtual_fill_count"] = int(scorecard.get("filled_signals") or 0)
    state["last_event_timestamp_utc"] = utc_now()
    state["latest_quote_timestamp_utc"] = _latest_quote_timestamp(snapshot)
    state["latest_bar_timestamp_utc"] = _latest_bar_timestamp(fast_packet)
    state["last_cycle_refreshed_market_packet"] = refreshed
    state["last_shadow_result"] = bot_result
    state["last_outcome_result"] = outcome_result
    state["last_profit_holdout_selection_result"] = profit_holdout_selection_result
    state["last_profit_holdout_outcome_result"] = profit_holdout_outcome_result
    state["last_profit_holdout_scorecard_result"] = profit_holdout_scorecard_result
    state["last_corrective_challenger_result"] = challenger_result
    state["last_knowledge_result"] = knowledge_result
    state["last_shock_follow_result"] = shock_follow_result
    state["last_shock_guard_result"] = shock_guard_result
    state["last_pair_chart_result"] = pair_chart_result
    state["last_eurusd_learning_result"] = eurusd_learning_result
    state["last_snapshot_result"] = _json_stdout(snapshot_result)
    return state


def run_resident(args: argparse.Namespace, *, once: bool = False) -> int:
    global STOP
    STOP = False
    manifest = verify_release(
        expected_commit=args.expected_commit,
        expected_source_sha256=args.expected_source_sha256,
    )
    root = cohort_root(args.state_root, args.expected_commit, args.expected_source_sha256)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    for ledger in (
        root / "ledgers" / "fast_bot_shadow_ledger.jsonl",
        root / "ledgers" / "fast_bot_outcome_ledger.jsonl",
        root / "ledgers" / "fast_bot_profit_holdout_signal_ledger.jsonl",
        root / "ledgers" / "fast_bot_profit_holdout_decision_ledger.jsonl",
        root / "ledgers" / "fast_bot_profit_holdout_outcome_ledger.jsonl",
        root / "ledgers" / "fast_bot_corrective_challenger_ledger.jsonl",
        root / "ledgers" / "fast_bot_learning_episode_ledger.jsonl",
        root / "ledgers" / "fast_bot_knowledge_ledger.jsonl",
        root / "ledgers" / "fast_bot_shock_follow_signal_ledger.jsonl",
        root / "ledgers" / "fast_bot_shock_follow_outcome_ledger.jsonl",
        root / "ledgers" / "fast_bot_shock_guard_decision_ledger.jsonl",
        root / "ledgers" / "eurusd_learned_policy_prospective_decision_ledger.jsonl",
        root / "ledgers" / "eurusd_learned_policy_prospective_outcome_ledger.jsonl",
    ):
        ledger.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        ledger.touch(exist_ok=True, mode=0o600)
        os.chmod(ledger, 0o600)
    lock_path = root / "runtime.lock"
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeBlocked("DUPLICATE_RESIDENT_PROCESS") from exc
    started_at = utc_now()
    prior = read_object(root / "state" / "status.json")
    restart_count = int(prior.get("restart_count") or 0) + 1
    counters = {
        "event_count": int((prior.get("counters") or {}).get("event_count") or 0),
        "proposal_count": int((prior.get("counters") or {}).get("proposal_count") or 0),
        "virtual_fill_count": int((prior.get("counters") or {}).get("virtual_fill_count") or 0),
        "cycle_failures": int((prior.get("counters") or {}).get("cycle_failures") or 0),
    }
    status = _base_status(
        manifest=manifest,
        root=root,
        started_at=started_at,
        counters=counters,
        restart_count=restart_count,
    )
    manifest_body = {
        "contract": "QR_OWNER_FORWARD_SHADOW_RELEASE_V1",
        "sealed_at_utc": utc_now(),
        **manifest,
        "execution_authority": "NONE",
        "broker_http_methods_allowed": ["GET"],
        "broker_mutation_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "live_permission": False,
        "promotion_allowed": False,
        "paper_campaign_namespace": f"paper-owner-{args.expected_commit[:12]}",
        "live_campaign_namespace": "DISTINCT_NOT_USED",
        "shadow_pairs": list(SHADOW_PAIRS),
        "strategy_methods": [
            "TREND_CONTINUATION",
            "RANGE_ROTATION",
            "BREAKOUT_FAILURE",
            "SHOCK_BREAKOUT_FOLLOW",
            "SHOCK_PULLBACK_CONTINUATION",
        ],
    }
    manifest_body["manifest_sha256"] = canonical_sha(
        {key: value for key, value in manifest_body.items() if key != "sealed_at_utc"}
    )
    existing_manifest = read_object(root / "release_manifest.json")
    if existing_manifest and existing_manifest != manifest_body:
        stable_existing = {k: v for k, v in existing_manifest.items() if k != "sealed_at_utc"}
        stable_new = {k: v for k, v in manifest_body.items() if k != "sealed_at_utc"}
        if stable_existing != stable_new:
            raise RuntimeBlocked("COHORT_MANIFEST_CONFLICT")
    else:
        atomic_json(root / "release_manifest.json", manifest_body)
    _write_zero_authority_inputs(root)
    state = dict(counters)
    atomic_json(root / "state" / "status.json", status)
    env = child_environment(args.oanda_env_file)

    def stop(*_: object) -> None:
        global STOP
        STOP = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while not STOP:
        cycle_started = time.monotonic()
        try:
            verify_release(
                expected_commit=args.expected_commit,
                expected_source_sha256=args.expected_source_sha256,
            )
            state = run_cycle(root=root, env=env, state=state)
            status.update(
                run_state="RUNNING",
                heartbeat_at_utc=utc_now(),
                last_error=None,
                last_event_timestamp_utc=state.get("last_event_timestamp_utc"),
                latest_quote_timestamp_utc=state.get("latest_quote_timestamp_utc"),
                latest_bar_timestamp_utc=state.get("latest_bar_timestamp_utc"),
                last_cycle_refreshed_market_packet=state.get("last_cycle_refreshed_market_packet"),
                last_shadow_result=state.get("last_shadow_result"),
                last_outcome_result=state.get("last_outcome_result"),
                last_profit_holdout_selection_result=state.get(
                    "last_profit_holdout_selection_result"
                ),
                last_profit_holdout_outcome_result=state.get(
                    "last_profit_holdout_outcome_result"
                ),
                last_profit_holdout_scorecard_result=state.get(
                    "last_profit_holdout_scorecard_result"
                ),
                last_corrective_challenger_result=state.get("last_corrective_challenger_result"),
                last_knowledge_result=state.get("last_knowledge_result"),
                last_shock_follow_result=state.get("last_shock_follow_result"),
                last_shock_guard_result=state.get("last_shock_guard_result"),
                last_pair_chart_result=state.get("last_pair_chart_result"),
                last_eurusd_learning_result=state.get("last_eurusd_learning_result"),
                counters={
                    "event_count": int(state.get("event_count", 0)),
                    "proposal_count": int(state.get("proposal_count", 0)),
                    "virtual_fill_count": int(state.get("virtual_fill_count", 0)),
                    "cycle_failures": int(state.get("cycle_failures", 0)),
                },
            )
        except Exception as exc:
            state["cycle_failures"] = int(state.get("cycle_failures", 0)) + 1
            status.update(
                run_state="DEGRADED_RETRYING",
                heartbeat_at_utc=utc_now(),
                last_error=f"{type(exc).__name__}: {exc}"[:600],
                counters={
                    "event_count": int(state.get("event_count", 0)),
                    "proposal_count": int(state.get("proposal_count", 0)),
                    "virtual_fill_count": int(state.get("virtual_fill_count", 0)),
                    "cycle_failures": int(state.get("cycle_failures", 0)),
                },
            )
        atomic_json(root / "state" / "status.json", status)
        if once:
            succeeded = status["run_state"] == "RUNNING"
            status.update(run_state="STOPPED_AFTER_ONCE", heartbeat_at_utc=utc_now())
            atomic_json(root / "state" / "status.json", status)
            return 0 if succeeded else 2
        remaining = max(1.0, float(args.interval_seconds) - (time.monotonic() - cycle_started))
        deadline = time.monotonic() + remaining
        while not STOP and time.monotonic() < deadline:
            status["heartbeat_at_utc"] = utc_now()
            atomic_json(root / "state" / "status.json", status)
            time.sleep(min(5.0, max(0.1, deadline - time.monotonic())))
    status.update(run_state="STOPPED", heartbeat_at_utc=utc_now())
    atomic_json(root / "state" / "status.json", status)
    return 0


def desired_plist(args: argparse.Namespace) -> dict[str, Any]:
    root = cohort_root(args.state_root, args.expected_commit, args.expected_source_sha256)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    (root / "logs").mkdir(parents=True, exist_ok=True, mode=0o700)
    return {
        "Label": LABEL,
        "ProgramArguments": [
            str(Path(sys.executable).resolve()),
            str(REPO_ROOT / "tools/owner_forward_shadow_runtime.py"),
            "run",
            "--expected-commit", args.expected_commit,
            "--expected-source-sha256", args.expected_source_sha256,
            "--state-root", str(args.state_root),
            "--oanda-env-file", str(args.oanda_env_file),
            "--interval-seconds", str(args.interval_seconds),
        ],
        "WorkingDirectory": str(REPO_ROOT),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 15,
        "ProcessType": "Background",
        "StandardOutPath": str(root / "logs" / "stdout.log"),
        "StandardErrorPath": str(root / "logs" / "stderr.log"),
    }


def install_launchagent(args: argparse.Namespace) -> dict[str, Any]:
    manifest = verify_release(
        expected_commit=args.expected_commit,
        expected_source_sha256=args.expected_source_sha256,
    )
    if not args.oanda_env_file.is_file():
        raise RuntimeBlocked("OANDA_ENV_FILE_MISSING")
    plist = desired_plist(args)
    agents = Path.home() / "Library" / "LaunchAgents"
    agents.mkdir(parents=True, exist_ok=True)
    target = agents / f"{LABEL}.plist"
    desired = plistlib.dumps(plist, fmt=plistlib.FMT_XML, sort_keys=True)
    if target.exists() and target.read_bytes() != desired:
        loaded = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/{LABEL}"],
            capture_output=True,
            text=True,
        ).returncode == 0
        if loaded:
            raise RuntimeBlocked("LOADED_DIFFERENT_OWNER_SHADOW_RELEASE")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(desired)
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", domain, str(target)], capture_output=True, text=True)
    subprocess.run(["launchctl", "bootstrap", domain, str(target)], check=True, capture_output=True, text=True)
    subprocess.run(["launchctl", "enable", f"{domain}/{LABEL}"], check=True, capture_output=True, text=True)
    return {
        "status": "INSTALLED_AND_STARTED",
        "label": LABEL,
        "plist_path": str(target),
        "source_commit": manifest["commit"],
        "source_bundle_sha256": manifest["source_bundle_sha256"],
        "cohort_root": str(cohort_root(args.state_root, args.expected_commit, args.expected_source_sha256)),
        "external_order_attempts": 0,
        "external_orders": 0,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("run", "once", "preinstall", "install", "status"):
        command = sub.add_parser(name)
        command.add_argument("--expected-commit", required=True)
        command.add_argument("--expected-source-sha256", required=True)
        command.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
        command.add_argument("--oanda-env-file", type=Path, default=DEFAULT_ENV_FILE)
        command.add_argument("--interval-seconds", type=float, default=30.0)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            return run_resident(args)
        if args.command == "once":
            return run_resident(args, once=True)
        if args.command == "preinstall":
            manifest = verify_release(
                expected_commit=args.expected_commit,
                expected_source_sha256=args.expected_source_sha256,
            )
            print(json.dumps({"status": "PREINSTALL_OK", **manifest}, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "install":
            print(json.dumps(install_launchagent(args), ensure_ascii=False, sort_keys=True))
            return 0
        root = cohort_root(args.state_root, args.expected_commit, args.expected_source_sha256)
        print(json.dumps(read_object(root / "state" / "status.json"), ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
