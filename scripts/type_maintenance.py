#!/usr/bin/env python3
"""Type maintenance helper for QuantRabbit.

Workflow:
- make type-fix : Ruff ANN ルールを利用して欠損型ヒントを自動補完
- make type-check: Ruff/pyright/mypy で型検証
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "logs" / "type_audit_report.json"
TARGET_PATHS = [
    "analysis",
    "execution",
    "utils",
    "addons",
    "market_data",
    "indicators",
    "scripts",
    "strategies",
    "analytics",
    "workers",
    "tests",
    "main.py",
]
ANN_RULES = "ANN001,ANN201,ANN202,ANN204,ANN205,ANN206,ANN401"


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def run_command(cmd: List[str], *, cwd: Path, label: str) -> Dict[str, Any]:
    process = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    return {
        "label": label,
        "returncode": process.returncode,
        "stdout": process.stdout.strip(),
        "stderr": process.stderr.strip(),
        "command": " ".join(cmd),
    }


def count_ruff_issues(output: str) -> int:
    pattern = re.compile(r"^.+?:\d+:\d+: [A-Z]+\d+")
    return sum(1 for line in output.splitlines() if pattern.match(line))


def run_ruff(paths: List[str], *, fix: bool = False) -> Dict[str, Any]:
    if not module_available("ruff"):
        return {
            "label": "ruff",
            "returncode": 127,
            "stdout": "",
            "stderr": "ruff module not found. Install with: pip install -r requirements-dev.txt",
            "command": f"ruff {'--fix ' if fix else ''}check --select {ANN_RULES} ...",
            "issues": 0,
            "applied": False,
        }

    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--select",
        ANN_RULES,
        *paths,
    ]
    if fix:
        cmd.append("--fix")
    result = run_command(cmd, cwd=ROOT, label="ruff")
    result["issues"] = count_ruff_issues(result["stdout"] + "\n" + result["stderr"])
    result["applied"] = fix
    return result


def run_mypy(paths: List[str]) -> Dict[str, Any]:
    if not module_available("mypy"):
        return {
            "label": "mypy",
            "returncode": 127,
            "stdout": "",
            "stderr": "mypy module not found. Install with: pip install -r requirements-dev.txt",
            "command": f"mypy --config-file mypy.ini {' '.join(paths)}",
            "issues": 0,
            "applied": False,
        }

    cmd = [sys.executable, "-m", "mypy", "--config-file", str(ROOT / "mypy.ini"), *paths]
    result = run_command(cmd, cwd=ROOT, label="mypy")
    result["issues"] = (result["stderr"].count("error: ") + result["stdout"].count("error: "))
    result["applied"] = False
    return result


def ensure_report_dir() -> None:
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)


def build_report(
    *,
    args: argparse.Namespace,
    ruff_fix: Dict[str, Any],
    ruff_check: Dict[str, Any],
    mypy: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "apply": args.apply,
        "target_paths": args.paths,
        "results": {
            "ruff_fix": ruff_fix,
            "ruff_check": ruff_check,
            "mypy": mypy,
        },
        "status": {
            "ruff_check_ok": ruff_check["returncode"] == 0,
            "mypy_ok": mypy["returncode"] == 0,
        },
    }


def print_result(name: str, result: Dict[str, Any], *, stream: str = "stdout") -> None:
    output = (result.get(stream) or "").strip()
    if output:
        print(output)
    if result.get("issues", 0):
        print(f"[{name}] issues={result['issues']} returncode={result['returncode']}")
    else:
        print(f"[{name}] OK ({result['returncode']})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantRabbit type maintenance workflow")
    parser.add_argument(
        "--paths",
        nargs="*",
        default=TARGET_PATHS,
        help="Paths to analyze. Default is core modules and scripts.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Enable Ruff autofix (adds missing function annotations with Any when possible).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [p for p in args.paths if (ROOT / p).exists()]
    if not paths:
        print("No valid paths were found.", file=sys.stderr)
        return 1
    ruff_fix: Dict[str, Any] = {}

    if args.apply:
        ruff_fix = run_ruff(paths, fix=True)
        print_result("ruff-fix", ruff_fix)
        if ruff_fix["returncode"] == 0 or ruff_fix["returncode"] == 1:
            print("ruff autofix completed. Re-running annotation check...")
        else:
            print(f"ruff autofix failed: {ruff_fix['stderr']}", file=sys.stderr)

    ruff_check = run_ruff(paths, fix=False)
    print_result("ruff-check", ruff_check)
    mypy = run_mypy(paths)
    print_result("mypy", mypy, stream="stderr" if mypy["returncode"] else "stdout")

    ensure_report_dir()
    report = build_report(args=args, ruff_fix=ruff_fix if args.apply else {}, ruff_check=ruff_check, mypy=mypy)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report: {REPORT_PATH}")

    if ruff_check["returncode"] != 0:
        print("Type check failed: ruff", file=sys.stderr)
        return 1
    if mypy["returncode"] != 0:
        print("Type check failed: mypy", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
