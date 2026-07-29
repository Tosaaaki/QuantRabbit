#!/usr/bin/env python3
"""Build a read-only inventory of legacy QuantRabbit strategies and VM evidence.

The scanner deliberately emits identifiers and evidence paths only. It never emits
environment values, command lines, file bodies, credentials, or current cloud state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


INFRA_WORKERS = {
    "forecast",
    "hedge_balancer",
    "macro_exit",
    "market_data_feed",
    "micro_exit",
    "micro_runtime",
    "order_manager",
    "position_manager",
    "regime_router",
    "scalp_exit",
    "strategy_control",
}
UTILITY_STRATEGIES = {"common"}

ALIASES = {
    "TrendMA": "trend_ma",
    "macro_trendma": "trend_ma",
    "ma_cross": "trend_ma",
    "PulseBreak": "pulse_break",
    "scalp_pulsebreak": "pulse_break",
    "scalp_squeeze_pulse_break": "pulse_break",
    "pulse_break": "pulse_break",
    "M1Scalper": "m1_scalper",
    "scalp_m1scalper": "m1_scalper",
    "m1_scalper": "m1_scalper",
    "RangeFader": "range_fader",
    "scalp_rangefader": "range_fader",
    "range_fader": "range_fader",
    "macro_donchian55": "donchian55",
    "donchian55": "donchian55",
    "macro_h1momentum": "h1_momentum",
    "trend_h1": "h1_momentum",
    "h1_momentum": "h1_momentum",
    "scalp_impulseretrace": "impulse_retrace",
    "impulse_retrace": "impulse_retrace",
    "scalp_trend_reclaim": "trend_reclaim",
    "trend_reclaim_long": "trend_reclaim",
    "micro_bbrsi": "bb_rsi",
    "bb_rsi": "bb_rsi",
    "micro_compressionrevert": "compression_revert",
    "micro_levelreactor": "level_reactor",
    "micro_momentumburst": "momentum_burst",
    "micro_momentumpulse": "momentum_pulse",
    "micro_momentumstack": "momentum_stack",
    "micro_pullbackema": "pullback_ema",
    "micro_rangebreak": "range_break",
    "micro_trendmomentum": "trend_momentum",
    "micro_trendretest": "trend_retest",
    "micro_vwapbound": "vwap_bound_revert",
    "micro_vwaprevert": "micro_vwap_revert",
    "scalp_failed_break_reverse": "failed_break_reverse",
    "scalp_pullback_continuation": "pullback_continuation",
    "scalp_trend_breakout": "trend_breakout",
}

KNOWN_EVALUATED = {"trend_ma", "pulse_break", "m1_scalper", "range_fader"}
REPLAY_COMPATIBLE = {
    "impulse_break_s5",
    "impulse_retest_s5",
    "impulse_momentum_s5",
    "pullback_s5",
    "vwap_magnet_s5",
    "stop_run_reversal",
    "session_open",
    "trend_breakout",
    "pullback_continuation",
    "failed_break_reverse",
}

WORKER_PATH_RE = re.compile(r"(?:^|/)workers/([^/]+)/worker\.py$")
STRATEGY_PATH_RE = re.compile(r"(?:^|/)strategies/[^/]+/([^/]+)\.py$")
PAIR_RE = re.compile(r"\b(?:USD[_/-]?JPY|EUR[_/-]?USD|EUR[_/-]?JPY|GBP[_/-]?JPY)\b")
TIMEFRAME_RE = re.compile(r"\b(?:S5|S10|S15|M1|M5|M15|M30|H1|H4|D1)\b")
FUNC_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)
UNIT_RE = re.compile(r"\b(?:quant|qr)-[a-z0-9][a-z0-9_.@-]*\.service\b", re.I)
RESOURCE_NAME = r"[a-z](?:[a-z0-9-]{1,60}[a-z0-9])"
GCLOUD_INSTANCE_RE = re.compile(
    r"\bgcloud\s+compute\s+instances\s+"
    r"(?:create|describe|start|stop|delete|add-metadata|remove-metadata|reset)\s+"
    rf"({RESOURCE_NAME})\b",
    re.I,
)
GCLOUD_SSH_RE = re.compile(
    rf"\bgcloud\s+compute\s+ssh\s+({RESOURCE_NAME})\b",
    re.I,
)
GCP_RESOURCE_RE = re.compile(
    r"\bgcloud\s+compute\s+(?:disks|snapshots|images)\s+"
    rf"(?:create|describe|delete)\s+({RESOURCE_NAME})\b",
    re.I,
)
SECRET_PATH_RE = re.compile(r"/secrets/([A-Za-z][A-Za-z0-9_-]{2,127})")
NAME_ASSIGN_RE = re.compile(
    r"(?m)^[ \t]*(?:export[ \t]+)?"
    r"(?:VM|INSTANCE|HOST|SERVER|TARGET)[A-Z0-9_]*(?:_NAME|_ID)?[ \t]*="
    rf"[ \t]*['\"]?({RESOURCE_NAME})",
)
SECRET_ENV_ASSIGN_RE = re.compile(
    r"(?m)^[ \t]*(?:export[ \t]+)?"
    r"([A-Z][A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|PASSWD|API_KEY|PRIVATE_KEY|CREDENTIALS?)"
    r"(?:_NAME|_ID|_PATH|_FILE)?)[ \t]*="
)


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def _canonical(raw: str) -> str:
    if raw in ALIASES:
        return ALIASES[raw]
    value = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower()
    return ALIASES.get(value, value)


def _candidate_from_path(path: str) -> str | None:
    match = WORKER_PATH_RE.search(path)
    if match:
        raw = match.group(1)
        return None if raw in INFRA_WORKERS else _canonical(raw)
    match = STRATEGY_PATH_RE.search(path)
    if match and match.group(1) not in {"__init__", *UTILITY_STRATEGIES}:
        return _canonical(match.group(1))
    return None


def _iter_git_paths(repo: Path) -> set[str]:
    output = _git(
        repo,
        "log",
        "--all",
        "--pretty=format:",
        "--name-only",
        "--",
        "workers/*/worker.py",
        "strategies/**/*.py",
    )
    return {line.strip() for line in output.splitlines() if _candidate_from_path(line.strip())}


def _git_path_commits(repo: Path) -> dict[str, str]:
    output = _git(
        repo,
        "log",
        "--all",
        "--format=COMMIT:%H",
        "--name-only",
        "--",
        "workers/*/worker.py",
        "strategies/**/*.py",
        "archive/workers/*/worker.py",
        "archive/strategies/**/*.py",
    )
    current_commit: str | None = None
    commits: dict[str, str] = {}
    for line in output.splitlines():
        value = line.strip()
        if value.startswith("COMMIT:"):
            current_commit = value.removeprefix("COMMIT:")
        elif current_commit and _candidate_from_path(value):
            commits.setdefault(value, current_commit)
    return commits


def _physical_sources(archive_root: Path) -> list[Path]:
    sources: list[Path] = []
    for pattern in ("archive/workers/*/worker.py", "archive/strategies/**/*.py"):
        sources.extend(p for p in archive_root.glob(pattern) if _candidate_from_path(str(p)))
    return sorted(set(sources))


def _read_small_text(path: Path, limit: int = 512_000) -> str:
    try:
        if path.stat().st_size > limit:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _static_contract(text: str, path: str) -> dict[str, object]:
    pairs = sorted({value.replace("/", "_").replace("-", "_") for value in PAIR_RE.findall(text)})
    timeframes = sorted(set(TIMEFRAME_RE.findall(text)))
    functions = FUNC_RE.findall(text)
    entry_functions = sorted({name for name in functions if any(k in name.lower() for k in ("entry", "signal", "setup", "qualif"))})
    exit_features = [
        key
        for key in ("take_profit", "stop_loss", "trailing", "breakeven", "partial", "timeout", "exit_worker")
        if key in text.lower() or (key == "exit_worker" and "exit_worker" in path)
    ]
    cost_features = [
        key
        for key in ("spread", "slippage", "latency", "commission")
        if key in text.lower()
    ]
    return {
        "pairs": pairs or ["USD_JPY"],
        "timeframes": timeframes or ["unknown"],
        "entry": entry_functions[:12] or ["implementation-defined"],
        "exit": exit_features or ["implementation-defined"],
        "cost_model": cost_features or ["not_explicit_in_strategy_file"],
    }


def _load_old_metrics(archive_root: Path) -> dict[str, list[dict[str, object]]]:
    metrics: dict[str, list[dict[str, object]]] = defaultdict(list)
    ledgers = archive_root / "logs" / "archive_legacy"
    for path in sorted(ledgers.glob("backtest_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        by_strategy = payload.get("by_strategy")
        if not isinstance(by_strategy, dict):
            continue
        for raw_name, values in by_strategy.items():
            if not isinstance(values, dict):
                continue
            strategy_id = _canonical(str(raw_name))
            metrics[strategy_id].append(
                {
                    "ledger": str(path),
                    "window": payload.get("date"),
                    "timeframe": payload.get("timeframe"),
                    "net_pnl_pips": values.get("profit_pips"),
                    "profit_factor": values.get("profit_factor"),
                    "max_drawdown_pips": values.get("max_dd_pips"),
                    "trades": values.get("trades"),
                    "win_rate": values.get("win_rate"),
                }
            )
    return metrics


def _evidence_files(roots: Iterable[Path]) -> Iterable[Path]:
    name_markers = (
        "gcloud",
        "cloud",
        "terraform",
        "ansible",
        "startup",
        "systemd",
        "service",
        "docker",
        "compose",
        "inventory",
        "manifest",
        "launchd",
        "cron",
        "runbook",
        "deploy",
        "snapshot",
        "image",
        "vm_",
        ".env",
    )
    always_scan_suffixes = {".tf", ".service", ".plist", ".env"}
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name
                for name in dirnames
                if name
                not in {
                    ".git",
                    ".gcloud",
                    ".venv",
                    "__pycache__",
                    "node_modules",
                    "site-packages",
                    "virtenv",
                    "market_data",
                    "ticks",
                }
            ]
            base = Path(directory)
            for filename in filenames:
                path = base / filename
                if path.suffix == ".pyc":
                    continue
                if "tests" in path.parts:
                    continue
                lower = str(path).lower()
                if (
                    "legacy-strategy-full-inventory-" in lower
                    or filename == "inventory-legacy-strategies.py"
                ):
                    continue
                marked = any(marker in lower for marker in name_markers)
                if path.suffix.lower() not in always_scan_suffixes and not marked:
                    continue
                if path in seen:
                    continue
                seen.add(path)
                yield path


def _runtime_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if name
            not in {
                ".git",
                ".gcloud",
                ".venv",
                "__pycache__",
                "node_modules",
                "site-packages",
                "virtenv",
                "market_data",
                "ticks",
            }
        ]
        base = Path(directory)
        for filename in filenames:
            if filename.endswith(".service") or filename.endswith(".out.log"):
                paths.append(base / filename)
    return paths


def _safe_vm_evidence(paths: Iterable[Path], strategy_ids: Iterable[str]) -> dict[str, object]:
    instances: set[str] = set()
    resources: set[str] = set()
    units: set[str] = set()
    secret_names: set[str] = set()
    env_names: set[str] = set()
    evidence_paths: set[str] = set()
    worker_links: dict[str, set[str]] = defaultdict(set)
    resource_evidence: dict[str, set[str]] = defaultdict(set)
    strategy_tokens = {sid: set(sid.split("_")) | {sid.replace("_", "-")} for sid in strategy_ids}

    for path in paths:
        text = _read_small_text(path)
        if not text:
            continue
        found_instances = (
            set(GCLOUD_INSTANCE_RE.findall(text))
            | set(GCLOUD_SSH_RE.findall(text))
            | set(NAME_ASSIGN_RE.findall(text))
        )
        found_resources = set(GCP_RESOURCE_RE.findall(text))
        found_units = set(UNIT_RE.findall(text))
        found_secrets = set(SECRET_PATH_RE.findall(text))
        found_env_names = set(SECRET_ENV_ASSIGN_RE.findall(text))
        if not (found_instances or found_resources or found_units or found_secrets or found_env_names):
            continue
        evidence_paths.add(str(path))
        instances.update(found_instances)
        resources.update(found_resources)
        units.update(found_units)
        secret_names.update(found_secrets)
        env_names.update(found_env_names)
        for identifier in found_instances | found_resources | found_units | found_secrets | found_env_names:
            resource_evidence[identifier].add(str(path))
        lower = text.lower()
        path_lower = str(path).lower()
        identifiers = {
            token.replace("-", "_")
            for token in re.findall(r"\b[a-z][a-z0-9_-]{3,80}\b", lower)
        }
        for strategy_id, tokens in strategy_tokens.items():
            if strategy_id in identifiers or strategy_id in path_lower.replace("-", "_"):
                worker_links[strategy_id].add(str(path))
                continue
            meaningful = {token for token in tokens if len(token) >= 5}
            if meaningful and sum(
                token in identifiers or token in path_lower for token in meaningful
            ) >= min(2, len(meaningful)):
                worker_links[strategy_id].add(str(path))

    normalized_worker_links = {
        key: sorted(value) for key, value in sorted(worker_links.items()) if value
    }
    instance_worker_links: dict[str, list[str]] = {}
    for instance in sorted(instances):
        instance_paths = resource_evidence.get(instance, set())
        linked = sorted(
            strategy_id
            for strategy_id, linked_paths in worker_links.items()
            if instance_paths.intersection(linked_paths)
        )
        if linked:
            instance_worker_links[instance] = linked
    return {
        "mode": "read_only_static_artifact_scan",
        "current_gcp_queried": False,
        "instances": sorted(instances),
        "disk_snapshot_image_resources": sorted(resources),
        "systemd_units": sorted(units),
        "secret_names_only": sorted(secret_names | env_names),
        "evidence_paths": sorted(evidence_paths),
        "worker_links": normalized_worker_links,
        "instance_worker_links": instance_worker_links,
        "resource_evidence": {
            key: sorted(value) for key, value in sorted(resource_evidence.items()) if value
        },
        "redaction": "values, command lines, file bodies, auth material, and current cloud state are excluded",
    }


def _runtime_evidence(runtime_paths: Iterable[Path], strategy_id: str, aliases: set[str]) -> list[str]:
    evidence: set[str] = set()
    tokens = {strategy_id, strategy_id.replace("_", "-")} | {a.lower() for a in aliases}
    for path in runtime_paths:
        lower = str(path).lower()
        if any(token and (token in lower or token.replace("_", "-") in lower) for token in tokens):
            evidence.add(str(path))
    return sorted(evidence)


def build_inventory(
    repo: Path, archive_root: Path, extra_roots: Iterable[Path] = ()
) -> tuple[list[dict[str, object]], dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    source_text: dict[str, list[tuple[str, str]]] = defaultdict(list)

    git_paths = _iter_git_paths(repo)
    repo_commits = _git_path_commits(repo)
    archive_commits = _git_path_commits(archive_root)
    for path in sorted(git_paths):
        strategy_id = _candidate_from_path(path)
        if not strategy_id:
            continue
        item = grouped.setdefault(strategy_id, {"paths": set(), "commits": set(), "aliases": set()})
        item["paths"].add(path)
        item["aliases"].add(Path(path).parent.name if "/workers/" in f"/{path}" else Path(path).stem)
        commit = repo_commits.get(path)
        if commit:
            item["commits"].add(commit)

    for path in _physical_sources(archive_root):
        rel = str(path.relative_to(archive_root))
        strategy_id = _candidate_from_path(rel)
        if not strategy_id:
            continue
        item = grouped.setdefault(strategy_id, {"paths": set(), "commits": set(), "aliases": set()})
        item["paths"].add(str(path))
        item["aliases"].add(path.parent.name if path.name == "worker.py" else path.stem)
        text = _read_small_text(path)
        if text:
            source_text[strategy_id].append((str(path), text))
        commit = archive_commits.get(rel)
        if commit:
            item["commits"].add(commit)

    metrics = _load_old_metrics(archive_root)
    runtime_paths = _runtime_files(archive_root)
    inventory: list[dict[str, object]] = []
    for strategy_id, raw in sorted(grouped.items()):
        aliases = set(raw["aliases"])
        contracts = [_static_contract(text, path) for path, text in source_text.get(strategy_id, [])]
        pairs = sorted({value for contract in contracts for value in contract["pairs"]}) or ["USD_JPY"]
        timeframes = sorted({value for contract in contracts for value in contract["timeframes"]}) or ["unknown"]
        entry = sorted({value for contract in contracts for value in contract["entry"]})[:20]
        exit_model = sorted({value for contract in contracts for value in contract["exit"]})[:20]
        cost_model = sorted({value for contract in contracts for value in contract["cost_model"]})[:20]
        runtime = _runtime_evidence(runtime_paths, strategy_id, aliases)
        old_metrics = metrics.get(strategy_id, [])
        evaluated = strategy_id in KNOWN_EVALUATED or bool(old_metrics)
        replay_compatible = strategy_id in REPLAY_COMPATIBLE
        reproducibility = (
            "offline_replay_ready"
            if replay_compatible
            else "implementation_recoverable"
            if source_text.get(strategy_id)
            else "evidence_only"
        )
        priority = (
            0
            if evaluated
            else 100
            + (30 if runtime else 0)
            + (25 if replay_compatible else 0)
            + (10 if source_text.get(strategy_id) else 0)
        )
        inventory.append(
            {
                "strategy_id": strategy_id,
                "duplicate_family": strategy_id,
                "aliases": sorted(aliases),
                "implementation_paths": sorted(raw["paths"]),
                "implementation_commits": sorted(raw["commits"]),
                "runtime_evidence": runtime,
                "pair": pairs,
                "timeframe": timeframes,
                "entry": entry or ["implementation-defined"],
                "exit": exit_model or ["implementation-defined"],
                "cost_model": cost_model or ["not_explicit_in_strategy_file"],
                "past_results": old_metrics,
                "reproducibility": reproducibility,
                "replay_compatible": replay_compatible,
                "evaluation_status": "evaluated" if evaluated else "unevaluated",
                "priority_score": priority,
            }
        )

    strategy_ids = [str(item["strategy_id"]) for item in inventory]
    vm_roots = [repo, archive_root]
    vm_roots.extend(path.resolve() for path in extra_roots if path.exists())
    vm = _safe_vm_evidence(
        _evidence_files(vm_roots),
        strategy_ids,
    )
    vm["source_roots"] = [str(path) for path in vm_roots]
    return inventory, vm


def _render_report(inventory: list[dict[str, object]], vm: dict[str, object]) -> str:
    evaluated = [item for item in inventory if item["evaluation_status"] == "evaluated"]
    unevaluated = [item for item in inventory if item["evaluation_status"] == "unevaluated"]
    replay_ready = sorted(
        (item for item in unevaluated if item["replay_compatible"]),
        key=lambda item: (-int(item["priority_score"]), str(item["strategy_id"])),
    )
    lines = [
        "# Legacy strategy full inventory",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        "- Authority: NONE; live_permission=false",
        "- Scope: local/Git/archive static read-only discovery",
        f"- Found: {len(inventory)} normalized strategy families",
        f"- Evaluated: {len(evaluated)}",
        f"- Unevaluated: {len(unevaluated)}",
        f"- Replay-ready unevaluated: {len(replay_ready)}",
        "- Notion/Slack corpus: connector unavailable in this execution; not counted as searched",
        "",
        "## Replay priority",
        "",
    ]
    for item in replay_ready:
        lines.append(
            f"- `{item['strategy_id']}`: score={item['priority_score']}, "
            f"runtime_evidence={len(item['runtime_evidence'])}"
        )
    lines.extend(
        [
            "",
            "## GCP/VM trace recovery",
            "",
            f"- Instance identifiers: {len(vm['instances'])}",
            f"- Disk/snapshot/image identifiers: {len(vm['disk_snapshot_image_resources'])}",
            f"- systemd units: {len(vm['systemd_units'])}",
            f"- Worker-linked evidence families: {len(vm['worker_links'])}",
            "- Current GCP was not queried or changed. Secret values and command bodies are excluded.",
            "",
            "## Normalized inventory",
            "",
            "| strategy_id | aliases | runtime | result | replay | reproducibility |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for item in inventory:
        lines.append(
            f"| `{item['strategy_id']}` | {len(item['aliases'])} | "
            f"{len(item['runtime_evidence'])} | {item['evaluation_status']} | "
            f"{'yes' if item['replay_compatible'] else 'no'} | {item['reproducibility']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--extra-root", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    archive_root = args.archive_root.resolve()
    out_dir = args.out_dir.resolve()
    inventory, vm = build_inventory(repo, archive_root, args.extra_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inventory.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "authority": "NONE",
                "live_permission": False,
                "current_gcp_queried": False,
                "inventory": inventory,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "vm_evidence.json").write_text(
        json.dumps(vm, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(_render_report(inventory, vm), encoding="utf-8")
    print(
        json.dumps(
            {
                "found": len(inventory),
                "evaluated": sum(item["evaluation_status"] == "evaluated" for item in inventory),
                "unevaluated": sum(item["evaluation_status"] == "unevaluated" for item in inventory),
                "replay_ready_unevaluated": sum(
                    item["evaluation_status"] == "unevaluated" and item["replay_compatible"]
                    for item in inventory
                ),
                "out_dir": str(out_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
