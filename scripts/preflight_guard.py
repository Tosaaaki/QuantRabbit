#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = REPO_ROOT / "logs" / "change_preflight_latest.json"
REQUIRED_FINDINGS = "docs/TRADE_FINDINGS.md"

PROTECTED_PREFIXES = (
    "execution/",
    "workers/",
    "strategies/",
    "analysis/",
)
PROTECTED_EXACT = {
    "config/strategy_exit_protections.yaml",
    "config/dynamic_alloc.json",
    "config/participation_alloc.json",
    "config/auto_canary_overrides.json",
    "config/pattern_book.json",
    "config/pattern_book_deep.json",
    "ops/env/local-v2-stack.env",
}
PROTECTED_PREFIX_MATCHES = (
    "ops/env/quant-",
    "ops/env/profiles/",
)
PROTECTED_SCRIPTS = {
    "scripts/participation_allocator.py",
    "scripts/dynamic_alloc_worker.py",
    "scripts/entry_path_aggregator.py",
    "scripts/loser_cluster_worker.py",
    "scripts/auto_canary_improver.py",
    "scripts/run_local_feedback_cycle.py",
    "scripts/publish_health_snapshot.py",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guard trading/risk commits unless change preflight and TRADE_FINDINGS are present."
    )
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--max-age-min", type=int, default=360)
    parser.add_argument("--paths", nargs="*", help="Optional file list for dry-run/testing.")
    return parser.parse_args()


def _staged_paths(paths_override: list[str] | None) -> list[str]:
    if paths_override is not None:
        return [path for path in paths_override if path]
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _is_protected(path: str) -> bool:
    if path in PROTECTED_EXACT or path in PROTECTED_SCRIPTS:
        return True
    if any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES):
        return True
    if any(path.startswith(prefix) for prefix in PROTECTED_PREFIX_MATCHES):
        return True
    return False


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_findings_lint() -> tuple[bool, str]:
    result = subprocess.run(
        ["python3", "scripts/trade_findings_lint.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "").strip()
    if result.returncode == 0:
        return True, output
    stderr = (result.stderr or "").strip()
    detail = output if output else stderr
    return False, detail


def main() -> int:
    args = _parse_args()

    if os.getenv("SKIP_PREFLIGHT_GUARD") == "1":
        print("preflight-guard: bypass via SKIP_PREFLIGHT_GUARD=1")
        return 0

    staged = _staged_paths(args.paths)
    protected = [path for path in staged if _is_protected(path)]
    if not protected:
        print("preflight-guard: skip (no protected trading/risk paths staged)")
        return 0

    if REQUIRED_FINDINGS not in staged:
        print("preflight-guard: blocked")
        print(f"- staged protected paths: {', '.join(protected[:8])}")
        print(f"- required staged file missing: {REQUIRED_FINDINGS}")
        print('- run `scripts/change_preflight.sh "<query>"` and update docs/TRADE_FINDINGS.md before commit')
        return 1

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print("preflight-guard: blocked")
        print(f"- missing artifact: {artifact_path}")
        print('- run `scripts/change_preflight.sh "<query>"` before commit')
        return 1

    try:
        artifact = _load_json(artifact_path)
    except Exception as exc:
        print("preflight-guard: blocked")
        print(f"- invalid artifact json: {artifact_path} ({exc})")
        return 1

    artifact_age_sec = max(0.0, time.time() - artifact_path.stat().st_mtime)
    max_age_sec = max(60, args.max_age_min * 60)
    if artifact_age_sec > max_age_sec:
        print("preflight-guard: blocked")
        print(f"- stale artifact: age_sec={artifact_age_sec:.0f} > max_age_sec={max_age_sec}")
        print(f"- artifact query: {artifact.get('query') or 'n/a'}")
        print('- rerun `scripts/change_preflight.sh "<query>"` and commit again')
        return 1

    query = str(artifact.get("query") or "").strip()
    if not query:
        print("preflight-guard: blocked")
        print("- artifact query is empty")
        print('- rerun `scripts/change_preflight.sh "<query>"` with a non-empty query')
        return 1

    lint_ok, lint_detail = _run_findings_lint()
    if not lint_ok:
        print("preflight-guard: blocked")
        print("- TRADE_FINDINGS lint failed")
        if lint_detail:
            print(lint_detail)
        return 1

    review = artifact.get("review") if isinstance(artifact.get("review"), dict) else {}
    market = artifact.get("market") if isinstance(artifact.get("market"), dict) else {}
    print("preflight-guard: ok")
    print(f"- protected paths: {', '.join(protected[:8])}")
    print(f"- artifact query: {query}")
    print(f"- artifact age_sec: {artifact_age_sec:.0f}")
    if market:
        print(
            "- market: "
            f"mid={market.get('mid', 'n/a')} spread={market.get('spread_pips', 'n/a')} "
            f"fills_15m={market.get('fills_15m', 'n/a')} rejects_30m={market.get('rejects_30m', 'n/a')} "
            f"status={artifact.get('preflight_status', 'n/a')}"
        )
    print("- findings_lint: ok")
    print(f"- review matches: {len(review.get('entries') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
