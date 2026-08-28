"""Preinstall and install OANDA-only user LaunchAgents."""
from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
from pathlib import Path

from oanda_launchd_runtime import (
    LABELS,
    PACKAGE_ROOT,
    RUNTIME_SOURCE_HASHES,
    SERVICE_ATTESTATION_HASH,
    SERVICE_ROOT,
    SHARED_RUNTIME_HASH,
    runtime_source_hashes,
)
from shadow_runtime import real_dir, secure_read, valid_target

PLIST_ROOT = PACKAGE_ROOT / "oanda_launchagents"
USER_LAUNCH_AGENTS = Path.home() / "Library" / "LaunchAgents"
BANNED = ("massive", "coinbase", "crypto", "/orders", "/trades", "/positions", "POST", "PUT", "PATCH", "DELETE")


def plist_paths() -> list[Path]:
    return [PLIST_ROOT / f"{label}.plist" for label in LABELS.values()]


def preinstall() -> dict[str, int | str]:
    if runtime_source_hashes() != RUNTIME_SOURCE_HASHES:
        raise RuntimeError("RUNTIME_SOURCE_DRIFT")
    labels = set()
    for path in plist_paths():
        subprocess.run(["plutil", "-lint", str(path)], check=True, capture_output=True, text=True)
        raw = secure_read(path)
        lowered = raw.decode("utf-8", "strict").lower()
        if any(term.lower() in lowered for term in BANNED):
            raise RuntimeError(f"BANNED_PLIST_CAPABILITY:{path.name}")
        data = plistlib.loads(raw)
        label = data["Label"]
        if label in labels or label not in LABELS.values():
            raise RuntimeError("PLIST_LABEL_MISMATCH")
        labels.add(label)
        args = data["ProgramArguments"]
        if args[0] != "/Users/tossaki/.pyenv/versions/3.10.14/bin/python3":
            raise RuntimeError("PYTHON_PATH_MISMATCH")
        if args[1] != str(PACKAGE_ROOT / "oanda_launchd_runtime.py"):
            raise RuntimeError("SERVICE_PATH_MISMATCH")
        if data.get("WorkingDirectory") != str(PACKAGE_ROOT):
            raise RuntimeError("WORKING_DIRECTORY_MISMATCH")
        if "EnvironmentVariables" in data:
            raise RuntimeError("PLIST_ENVIRONMENT_FORBIDDEN")
    if labels != set(LABELS.values()):
        raise RuntimeError("PLIST_SET_INCOMPLETE")
    return {
        "plists": len(labels),
        "lint_failures": 0,
        "candidate_runtime_hash": SHARED_RUNTIME_HASH,
        "service_attestation_hash": SERVICE_ATTESTATION_HASH,
    }


def install() -> dict[str, int | str]:
    result = preinstall()
    real_dir(USER_LAUNCH_AGENTS)
    for subdir in ("feed", "bot", "llm", "watchdog", "logs", "triggers"):
        real_dir(SERVICE_ROOT / subdir)
    installed = 0
    for source in plist_paths():
        target = USER_LAUNCH_AGENTS / source.name
        valid_target(target)
        temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
        valid_target(temporary)
        temporary.write_bytes(secure_read(source))
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        installed += 1
    return {**result, "installed": installed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preinstall", "install"))
    args = parser.parse_args(argv)
    result = preinstall() if args.action == "preinstall" else install()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
