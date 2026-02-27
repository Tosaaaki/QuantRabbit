#!/usr/bin/env python3
"""Run the V2 health audit for core services and control/strategy execution flow.

Checks:
- critical unit active state
- EnvironmentFile wiring for V2 core + strategy/manager services
- runtime env required key sanity
- 405 / Method Not Allowed for open_positions path
- Git HEAD / origin/main consistency check
- legacy/unit drift (disabled monolith)

Outputs:
- latest JSON: logs/ops_v2_audit_latest.json
- history JSON: logs/ops_v2_audit_<timestamp>.json
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_JOURNAL_405_TIMEOUT_SEC = max(
    5.0,
    float(os.getenv("OPS_V2_AUDIT_JOURNAL_TIMEOUT_SEC", "45")),
)


@dataclass
class Finding:
    level: str
    component: str
    message: str
    details: dict[str, Any]


def _repo_dir() -> Path:
    return Path(os.getenv("OPS_AUDIT_REPO", "/home/tossaki/QuantRabbit")).expanduser()


def _logs_dir(repo_dir: Path) -> Path:
    log_dir = repo_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run(cmd: list[str], *, timeout: float = 10.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        detail = f"timeout after {timeout:.1f}s"
        if stderr:
            detail = f"{detail}: {stderr}"
        return 124, stdout, detail
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _unit_file_exists(unit: str) -> bool:
    rc, stdout, _ = _run(["systemctl", "list-unit-files", "--no-legend", "--no-pager", "--type=service", unit])
    if rc != 0:
        return False
    return unit in stdout


def _is_active(unit: str) -> bool | None:
    rc, stdout, _ = _run(["systemctl", "is-active", unit])
    if rc == 0:
        return stdout.strip() == "active"
    return None


def _cat_unit(unit: str) -> str:
    rc, stdout, stderr = _run(["systemctl", "cat", unit])
    if rc != 0:
        return stderr
    return stdout


def _extract_environment_files(unit_content: str) -> list[str]:
    env_lines: list[str] = []
    for line in unit_content.splitlines():
        if not line.startswith("EnvironmentFile="):
            continue
        value = line.split("=", 1)[1].strip()
        value = value.strip('"')
        if value.startswith("-"):
            value = value[1:]
        env_lines.append(value)
    return env_lines


def _parse_env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip().strip('"').strip("'")
        result[key] = value
    return result


def _parse_csv_set(raw_value: str) -> set[str]:
    items: set[str] = set()
    for item in raw_value.split(","):
        item = item.strip()
        if item:
            items.add(item)
    return items


def _journal_405_count(hours: int = 3) -> int:
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    method_not_allowed = re.compile(r"method not allowed", re.IGNORECASE)
    status_405 = re.compile(r"status\s*=\s*405\b", re.IGNORECASE)
    request_405 = re.compile(
        r'"[A-Z]+\s+/position/open_positions(?:\?[^"]*)?\s+HTTP/\d\.\d"\s+405\b'
    )
    cmd = [
        "journalctl",
        "-u",
        "quant-position-manager.service",
        "-u",
        "quant-order-manager.service",
        "--since",
        since,
        "--no-pager",
        "--output",
        "short-iso",
        "--grep",
        "open_positions|method not allowed|status\\s*=\\s*405",
    ]
    rc, stdout, _ = _run(cmd, timeout=_JOURNAL_405_TIMEOUT_SEC)
    if rc != 0 or not stdout:
        return 0
    count = 0
    for line in stdout.splitlines():
        low = line.lower()
        if "/position/open_positions" not in low:
            continue
        if method_not_allowed.search(low):
            count += 1
            continue
        if status_405.search(low):
            count += 1
            continue
        if request_405.search(line):
            count += 1
    return count


def _git_ref(repo_dir: Path, ref: str) -> str | None:
    if not (repo_dir / ".git").exists():
        return None
    rc, stdout, _ = _run(["git", "-C", str(repo_dir), "rev-parse", ref], timeout=10.0)
    if rc != 0:
        return None
    return stdout.strip()


def _emit_findings_summary(findings: list[Finding]) -> tuple[int, int, int]:
    severity = {"critical": 0, "warn": 0, "info": 0}
    for finding in findings:
        lvl = finding.level.lower()
        if lvl in severity:
            severity[lvl] += 1
    return severity["critical"], severity["warn"], severity["info"]


def _add_finding(results: list[Finding], *, level: str, component: str, message: str, details: dict[str, Any] | None = None) -> None:
    results.append(Finding(level=level, component=component, message=message, details=details or {}))


def main() -> int:
    repo_dir = _repo_dir()
    log_dir = _logs_dir(repo_dir)

    env_dir = repo_dir / "ops" / "env"
    runtime_env = env_dir / "quant-v2-runtime.env"
    order_manager_env = env_dir / "quant-order-manager.env"
    position_manager_env = env_dir / "quant-position-manager.env"

    core_services = [
        "quant-market-data-feed.service",
        "quant-strategy-control.service",
        "quant-order-manager.service",
        "quant-position-manager.service",
    ]

    mandatory_pairs = [
        ("quant-scalp-ping-5s-b.service", "quant-scalp-ping-5s-b-exit.service"),
        ("quant-micro-rangebreak.service", "quant-micro-rangebreak-exit.service"),
        ("quant-micro-levelreactor.service", "quant-micro-levelreactor-exit.service"),
        ("quant-micro-vwapbound.service", "quant-micro-vwapbound-exit.service"),
        ("quant-micro-vwaprevert.service", "quant-micro-vwaprevert-exit.service"),
        ("quant-micro-momentumburst.service", "quant-micro-momentumburst-exit.service"),
        ("quant-micro-momentumstack.service", "quant-micro-momentumstack-exit.service"),
        ("quant-micro-pullbackema.service", "quant-micro-pullbackema-exit.service"),
        ("quant-micro-trendmomentum.service", "quant-micro-trendmomentum-exit.service"),
        ("quant-micro-trendretest.service", "quant-micro-trendretest-exit.service"),
        ("quant-micro-compressionrevert.service", "quant-micro-compressionrevert-exit.service"),
        ("quant-micro-momentumpulse.service", "quant-micro-momentumpulse-exit.service"),
        ("quant-scalp-macd-rsi-div.service", "quant-scalp-macd-rsi-div-exit.service"),
        ("quant-scalp-tick-imbalance.service", "quant-scalp-tick-imbalance-exit.service"),
        ("quant-scalp-squeeze-pulse-break.service", "quant-scalp-squeeze-pulse-break-exit.service"),
        ("quant-scalp-wick-reversal-blend.service", "quant-scalp-wick-reversal-blend-exit.service"),
        ("quant-scalp-wick-reversal-pro.service", "quant-scalp-wick-reversal-pro-exit.service"),
        ("quant-m1scalper.service", "quant-m1scalper-exit.service"),
    ]

    optional_pairs = [
        ("quant-scalp-macd-rsi-div-b.service", "quant-scalp-macd-rsi-div-b-exit.service"),
        ("quant-scalp-rangefader.service", "quant-scalp-rangefader-exit.service"),
        ("quant-scalp-extrema-reversal.service", "quant-scalp-extrema-reversal-exit.service"),
    ]

    required_env_by_service: dict[str, list[str]] = {
        "quant-market-data-feed.service": [str(runtime_env)],
        "quant-strategy-control.service": [str(runtime_env)],
        "quant-order-manager.service": [str(runtime_env), str(order_manager_env)],
        "quant-position-manager.service": [str(runtime_env), str(position_manager_env)],
    }

    for entry, exit_service in mandatory_pairs + optional_pairs:
        entry_service_base = entry.replace(".service", "")
        exit_service_base = exit_service.replace(".service", "")
        required_env_by_service[entry] = [
            str(runtime_env),
            str(env_dir / f"{entry_service_base}.env"),
        ]
        required_env_by_service[exit_service] = [
            str(runtime_env),
            str(env_dir / f"{exit_service_base}.env"),
        ]

    runtime_env_expectations = {
        "WORKER_ONLY_MODE": "true",
        "MAIN_TRADING_ENABLED": "1",
        "SIGNAL_GATE_ENABLED": "0",
        "ORDER_FORWARD_TO_SIGNAL_GATE": "0",
        "ORDER_PATTERN_GATE_GLOBAL_OPT_IN": "0",
        "BRAIN_ENABLED": "0",
        "POLICY_HEURISTIC_PERF_BLOCK_ENABLED": "0",
        "ENTRY_GUARD_ENABLED": "0",
    }

    forbidden_env_file = "/etc/quantrabbit.env"
    forbidden_env_services = {
        "quant-online-tuner.service",
        "quant-autotune-ui.service",
        "quant-bq-sync.service",
        "quant-health-snapshot.service",
        "quant-level-map.service",
        "quant-strategy-optimizer.service",
        "quant-ui-snapshot.service",
    }

    disallowed_services = [
        "quantrabbit.service",
        "quant-impulse-retest-s5.service",
        "quant-impulse-retest-s5-exit.service",
        "quant-micro-adaptive-revert.service",
        "quant-micro-adaptive-revert-exit.service",
        "quant-trend-reclaim-long.service",
        "quant-trend-reclaim-long-exit.service",
    ]

    # 6) disallowed units allowlist (temporary holdover during legacy strategy retention).
    allowed_legacy_services = _parse_csv_set(
        os.environ.get(
            "OPS_V2_ALLOWED_LEGACY_SERVICES",
            _parse_env_file(runtime_env).get("OPS_V2_ALLOWED_LEGACY_SERVICES", ""),
        )
    )

    findings: list[Finding] = []

    # 1) unit presence / active
    for service in core_services:
        if not _unit_file_exists(service):
            _add_finding(
                findings,
                level="critical",
                component="systemd",
                message=f"Required service unit is missing: {service}",
                details={"service": service},
            )
            continue

        active = _is_active(service)
        if active is None:
            _add_finding(
                findings,
                level="warn",
                component="systemd",
                message=f"Unable to get active state for required service: {service}",
                details={"service": service},
            )
        elif not active:
            _add_finding(
                findings,
                level="critical",
                component="systemd",
                message=f"Required service is not active: {service}",
                details={"service": service},
            )

    def _service_state_pair(entry_service: str, exit_service: str) -> None:
        for service in (entry_service, exit_service):
            exists = _unit_file_exists(service)
            if not exists:
                _add_finding(
                    findings,
                    level="warn",
                    component="systemd",
                    message=f"Strategy pair unit is missing: {service}",
                    details={"service": service, "pair": f"{entry_service}/{exit_service}"},
                )
                continue

            active = _is_active(service)
            if active is None:
                _add_finding(
                    findings,
                    level="warn",
                    component="systemd",
                    message=f"Unable to get active state for strategy unit: {service}",
                    details={"service": service, "pair": f"{entry_service}/{exit_service}"},
                )
            elif not active:
                _add_finding(
                    findings,
                    level="warn",
                    component="systemd",
                    message=f"Strategy unit is not active: {service}",
                    details={"service": service, "pair": f"{entry_service}/{exit_service}"},
                )

    for pair in mandatory_pairs:
        _service_state_pair(*pair)

    for pair in optional_pairs:
        for service in pair:
            if _unit_file_exists(service):
                _service_state_pair(*pair)

    # 2) env files wiring and forbidden legacy
    for service, required_files in required_env_by_service.items():
        if not _unit_file_exists(service):
            continue
        unit_content = _cat_unit(service)
        env_files_ordered = _extract_environment_files(unit_content)
        env_files = set(env_files_ordered)
        dup_env_files = {
            path: count
            for path, count in Counter(env_files_ordered).items()
            if count > 1
        }
        if dup_env_files:
            _add_finding(
                findings,
                level="warn",
                component="systemd-env",
                message="Duplicate EnvironmentFile entry detected",
                details={"service": service, "duplicates": dup_env_files},
            )

        for required in required_files:
            if required not in env_files:
                if required == str(runtime_env) and service in {
                    "quant-market-data-feed.service",
                    "quant-strategy-control.service",
                    "quant-order-manager.service",
                    "quant-position-manager.service",
                }:
                    level = "critical"
                else:
                    level = "warn"
                _add_finding(
                    findings,
                    level=level,
                    component="systemd-env",
                    message="Missing EnvironmentFile entry",
                    details={
                        "service": service,
                        "required_file": required,
                    },
                )

        for env_file in required_files:
            if env_file == str(runtime_env):
                continue
            if env_file in env_files and not Path(env_file).exists():
                _add_finding(
                    findings,
                    level="warn",
                    component="systemd-env",
                    message="EnvironmentFile path not found on disk",
                    details={"service": service, "environment_file": env_file},
                )

        if forbidden_env_file in env_files:
            _add_finding(
                findings,
                level="warn",
                component="systemd-env",
                message="Legacy EnvironmentFile is still referenced",
                details={"service": service, "environment_file": forbidden_env_file},
            )

    for service in sorted(forbidden_env_services):
        if not _unit_file_exists(service):
            continue
        unit_content = _cat_unit(service)
        env_files = set(_extract_environment_files(unit_content))
        if forbidden_env_file in env_files:
            _add_finding(
                findings,
                level="warn",
                component="systemd-env",
                message="Legacy EnvironmentFile should be removed for ops service",
                details={"service": service, "environment_file": forbidden_env_file},
            )

    # 3) runtime env audit
    runtime_values = _parse_env_file(runtime_env)
    for key, expected in runtime_env_expectations.items():
        actual = runtime_values.get(key)
        if actual is None:
            _add_finding(
                findings,
                level="warn",
                component="runtime-env",
                message=f"Required runtime key is missing: {key}",
                details={"key": key, "expected": expected, "file": str(runtime_env)},
            )
            continue
        if actual != expected:
            _add_finding(
                findings,
                level="warn",
                component="runtime-env",
                message=f"Runtime key value mismatch: {key}",
                details={"key": key, "expected": expected, "actual": actual},
            )

    # 4) git consistency
    head_ref = _git_ref(repo_dir, "HEAD")
    origin_ref = _git_ref(repo_dir, "origin/main")
    if head_ref is None:
        _add_finding(
            findings,
            level="warn",
            component="deploy-state",
            message="git HEAD not readable on VM",
        )
    elif origin_ref is None:
        _add_finding(
            findings,
            level="warn",
            component="deploy-state",
            message="origin/main not readable (remote refs may be missing or stale)",
            details={"HEAD": head_ref},
        )
    elif head_ref != origin_ref:
        _add_finding(
            findings,
            level="warn",
            component="deploy-state",
            message="HEAD != origin/main",
            details={"HEAD": head_ref, "origin_main": origin_ref},
        )

    # 5) runtime method audit
    count_405 = _journal_405_count(hours=3)
    if count_405 > 0:
        _add_finding(
            findings,
            level="warn",
            component="position-manager",
            message="Detected open_positions method mismatch risk",
            details={"count_405_last_3h": count_405},
        )

    # 6) disallowed units
    for svc in disallowed_services:
        if not _unit_file_exists(svc):
            continue
        active = _is_active(svc)
        if active:
            if svc in allowed_legacy_services:
                _add_finding(
                    findings,
                    level="warn",
                    component="systemd",
                    message=f"Allowed legacy V2 unit is active: {svc}",
                    details={
                        "service": svc,
                        "note": "Listed in OPS_V2_ALLOWED_LEGACY_SERVICES",
                    },
                )
            else:
                _add_finding(
                    findings,
                    level="critical",
                    component="systemd",
                    message=f"Disallowed V2 legacy unit is active: {svc}",
                    details={"service": svc},
                )

    # normalize and dump
    findings.sort(key=lambda item: (item.level, item.component, item.message))
    critical, warn, info = _emit_findings_summary(findings)

    result = {
        "generated_at": _now_iso(),
        "hostname": socket.gethostname(),
        "repository": str(repo_dir),
        "summary": {
            "critical": critical,
            "warn": warn,
            "info": info,
            "total": len(findings),
        },
        "findings": [f.__dict__ for f in findings],
    }

    latest_report = log_dir / "ops_v2_audit_latest.json"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_report = log_dir / f"ops_v2_audit_{timestamp}.json"
    latest_report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    archive_report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ops-v2-audit] core findings: critical={critical}, warn={warn}, info={info}")
    for finding in findings[:12]:
        print(f"- {finding.level.upper()} [{finding.component}] {finding.message}")
    if len(findings) > 12:
        print(f"- ... and {len(findings)-12} more, see {latest_report}")

    return 1 if critical > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
