#!/usr/bin/env python3
"""Run the frozen two-candidate normalized-passive prospective observer.

The resident is a separate, content-attested GET-only collector so the narrow
post-M1 observation window is not delayed by the slower multi-timeframe owner
cycle.  It evaluates one rejected historical anchor and one distinct-pair
exploratory candidate under a corrected fixed family contract, and writes only
local shadow ledgers and scorecards.
"""

from __future__ import annotations

import argparse
import fcntl
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


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "src", ROOT / "tools"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from owner_forward_shadow_runtime import (  # noqa: E402
    DEFAULT_ENV_FILE,
    DEFAULT_STATE_ROOT,
    RuntimeBlocked,
    atomic_json,
    cohort_root,
    read_object,
    verify_release,
)
from quant_rabbit.broker.oanda import OandaReadOnlyClient  # noqa: E402
from quant_rabbit.fast_bot_normalized_passive_family_forward import (  # noqa: E402
    observe_from_oanda,
    resolve_due_outcomes_from_oanda,
)


LABEL = "com.quantrabbit.normalized-passive-forward"
POLICY_PATH = Path("config/fast_bot_normalized_passive_forward_v2.json")
STOP = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def lane_paths(root: Path) -> dict[str, Path]:
    return {
        "status": root / "state" / "fast_bot_normalized_passive_forward_status.json",
        "latest_decision": root / "state" / "fast_bot_normalized_passive_forward_latest_decision.json",
        "decision_ledger": root / "ledgers" / "fast_bot_normalized_passive_forward_decision_ledger.jsonl",
        "outcome_ledger": root / "ledgers" / "fast_bot_normalized_passive_forward_outcome_ledger.jsonl",
        "scorecard": root / "scorecard" / "fast_bot_normalized_passive_forward_scorecard.json",
        "lock": root / "fast_bot_normalized_passive_forward.lock",
        "stdout": root / "logs" / "fast_bot_normalized_passive_forward_stdout.log",
        "stderr": root / "logs" / "fast_bot_normalized_passive_forward_stderr.log",
    }


def run_cycle(
    *,
    root: Path,
    oanda_env_file: Path,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    paths = lane_paths(root)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    client_factory = lambda: OandaReadOnlyClient(env_file=oanda_env_file)
    observe = observe_from_oanda(
        policy_path=ROOT / POLICY_PATH,
        latest_decision_path=paths["latest_decision"],
        decision_ledger_path=paths["decision_ledger"],
        outcome_ledger_path=paths["outcome_ledger"],
        client_factory=client_factory,
        clock=clock,
    )
    resolve = resolve_due_outcomes_from_oanda(
        policy_path=ROOT / POLICY_PATH,
        decision_ledger_path=paths["decision_ledger"],
        outcome_ledger_path=paths["outcome_ledger"],
        scorecard_path=paths["scorecard"],
        client_factory=client_factory,
        clock=clock,
    )
    return _zero_authority(
        {
            "status": "CYCLE_OK",
            "observed_at_utc": utc_now(),
            "observation": _bounded_result(observe),
            "resolution": _bounded_result(resolve),
            "latest_decision_path": str(paths["latest_decision"]),
            "decision_ledger_path": str(paths["decision_ledger"]),
            "outcome_ledger_path": str(paths["outcome_ledger"]),
            "scorecard_path": str(paths["scorecard"]),
        }
    )


def _base_status(
    *,
    manifest: Mapping[str, Any],
    root: Path,
    started_at_utc: str,
    restart_count: int,
    cycle_count: int,
    cycle_failures: int,
) -> dict[str, Any]:
    paths = lane_paths(root)
    return _zero_authority(
        {
            "contract": "QR_FAST_BOT_NORMALIZED_PASSIVE_FORWARD_RESIDENT_V2",
            "schema_version": 2,
            "pid": os.getpid(),
            "started_at_utc": started_at_utc,
            "heartbeat_at_utc": utc_now(),
            "run_state": "STARTING",
            "restart_count": restart_count,
            "cycle_count": cycle_count,
            "cycle_failures": cycle_failures,
            "source_commit": manifest["commit"],
            "source_bundle_sha256": manifest["source_bundle_sha256"],
            "policy_path": str(ROOT / POLICY_PATH),
            "status_path": str(paths["status"]),
            "latest_decision_path": str(paths["latest_decision"]),
            "decision_ledger_path": str(paths["decision_ledger"]),
            "outcome_ledger_path": str(paths["outcome_ledger"]),
            "scorecard_path": str(paths["scorecard"]),
            "last_cycle": {},
            "last_error": None,
        }
    )


def run_resident(args: argparse.Namespace, *, once: bool = False) -> int:
    global STOP
    STOP = False
    manifest = verify_release(
        expected_commit=args.expected_commit,
        expected_source_sha256=args.expected_source_sha256,
    )
    if not args.oanda_env_file.is_file():
        raise RuntimeBlocked("OANDA_ENV_FILE_MISSING")
    root = cohort_root(args.state_root, args.expected_commit, args.expected_source_sha256)
    paths = lane_paths(root)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for key in ("decision_ledger", "outcome_ledger"):
        paths[key].touch(exist_ok=True, mode=0o600)
        os.chmod(paths[key], 0o600)
    lock_handle = paths["lock"].open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeBlocked("DUPLICATE_NORMALIZED_PASSIVE_RESIDENT") from exc

    prior = read_object(paths["status"])
    cycle_count = int(prior.get("cycle_count") or 0)
    cycle_failures = int(prior.get("cycle_failures") or 0)
    status = _base_status(
        manifest=manifest,
        root=root,
        started_at_utc=utc_now(),
        restart_count=int(prior.get("restart_count") or 0) + 1,
        cycle_count=cycle_count,
        cycle_failures=cycle_failures,
    )
    atomic_json(paths["status"], status)

    def stop(*_: object) -> None:
        global STOP
        STOP = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while not STOP:
        started = time.monotonic()
        try:
            verify_release(
                expected_commit=args.expected_commit,
                expected_source_sha256=args.expected_source_sha256,
            )
            result = run_cycle(root=root, oanda_env_file=args.oanda_env_file)
            cycle_count += 1
            status.update(
                run_state="RUNNING",
                heartbeat_at_utc=utc_now(),
                cycle_count=cycle_count,
                cycle_failures=cycle_failures,
                last_cycle=result,
                last_error=None,
            )
        except Exception as exc:
            cycle_failures += 1
            status.update(
                run_state="DEGRADED_RETRYING",
                heartbeat_at_utc=utc_now(),
                cycle_count=cycle_count,
                cycle_failures=cycle_failures,
                last_error=f"{type(exc).__name__}: {exc}"[:600],
            )
        atomic_json(paths["status"], status)
        if once:
            succeeded = status["run_state"] == "RUNNING"
            status.update(run_state="STOPPED_AFTER_ONCE", heartbeat_at_utc=utc_now())
            atomic_json(paths["status"], status)
            return 0 if succeeded else 2
        remaining = max(1.0, float(args.interval_seconds) - (time.monotonic() - started))
        deadline = time.monotonic() + remaining
        while not STOP and time.monotonic() < deadline:
            status["heartbeat_at_utc"] = utc_now()
            atomic_json(paths["status"], status)
            time.sleep(min(5.0, max(0.1, deadline - time.monotonic())))
    status.update(run_state="STOPPED", heartbeat_at_utc=utc_now())
    atomic_json(paths["status"], status)
    return 0


def desired_plist(args: argparse.Namespace) -> dict[str, Any]:
    root = cohort_root(args.state_root, args.expected_commit, args.expected_source_sha256)
    paths = lane_paths(root)
    paths["stdout"].parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return {
        "Label": LABEL,
        "ProgramArguments": [
            str(Path(sys.executable).resolve()),
            str(ROOT / "tools/run_fast_bot_normalized_passive_forward.py"),
            "run",
            "--expected-commit",
            args.expected_commit,
            "--expected-source-sha256",
            args.expected_source_sha256,
            "--state-root",
            str(args.state_root),
            "--oanda-env-file",
            str(args.oanda_env_file),
            "--interval-seconds",
            str(args.interval_seconds),
        ],
        "WorkingDirectory": str(ROOT),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 15,
        "ProcessType": "Background",
        "StandardOutPath": str(paths["stdout"]),
        "StandardErrorPath": str(paths["stderr"]),
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
    domain = f"gui/{os.getuid()}"
    subprocess.run(
        ["launchctl", "bootout", domain, str(target)],
        capture_output=True,
        text=True,
    )
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(desired)
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
    subprocess.run(
        ["launchctl", "bootstrap", domain, str(target)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["launchctl", "enable", f"{domain}/{LABEL}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return _zero_authority(
        {
            "status": "INSTALLED_AND_STARTED",
            "label": LABEL,
            "plist_path": str(target),
            "source_commit": manifest["commit"],
            "source_bundle_sha256": manifest["source_bundle_sha256"],
            "cohort_root": str(
                cohort_root(args.state_root, args.expected_commit, args.expected_source_sha256)
            ),
        }
    )


def _bounded_result(value: Mapping[str, Any]) -> dict[str, Any]:
    keep = (
        "status",
        "decision_id",
        "decision_at_utc",
        "activation_at_utc",
        "observed_at_utc",
        "qualifying_return",
        "source_direction",
        "selected_side",
        "normalized_return",
        "decision_spread_pips",
        "selected_due_count",
        "ledger_appended_count",
        "scorecard_status",
        "prospective_gate_passed",
        "errors",
        "candidate_results",
        "terminal_candidate_ids",
        "execution_authority",
        "external_order_attempts",
        "external_orders",
        "live_permission",
        "promotion_allowed",
    )
    return {key: value.get(key) for key in keep if key in value}


def _zero_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(value),
        "shadow_only": True,
        "execution_authority": "NONE",
        "broker_http_methods_allowed": ["GET"],
        "broker_mutation_allowed": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "gateway_invocations": 0,
        "live_permission": False,
        "promotion_allowed": False,
        "automatic_adoption_allowed": False,
        "primary_trading_candidate_allowed": False,
        "manual_tagless_policy": "NO_TOUCH",
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
        command.add_argument("--interval-seconds", type=float, default=20.0)
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
        print(json.dumps(read_object(lane_paths(root)["status"]), ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
