#!/usr/bin/env python3
"""Deduplicate redundant EnvironmentFile lines in systemd drop-ins.

Safety:
- Only removes duplicate `EnvironmentFile=` entries that point to the exact
  same file path already seen earlier in systemd load order.
- Never edits the fragment unit file; edits only `/etc/systemd/system/*.d/*.conf`.
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
from collections import defaultdict
from pathlib import Path


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _list_services(patterns: list[str]) -> list[str]:
    rc, out, _ = _run(
        ["systemctl", "list-unit-files", "--type=service", "--no-pager", "--no-legend"]
    )
    if rc != 0:
        return []
    names: list[str] = []
    for line in out.splitlines():
        parts = line.split()
        if not parts:
            continue
        name = parts[0].strip()
        if any(fnmatch.fnmatch(name, pat) for pat in patterns):
            names.append(name)
    return sorted(set(names))


def _show_value(service: str, prop: str) -> str:
    rc, out, _ = _run(["systemctl", "show", service, f"--property={prop}", "--value"])
    if rc != 0:
        return ""
    return out.strip()


def _normalize_envfile(raw_value: str) -> str:
    value = raw_value.strip().strip('"').strip("'")
    if value.startswith("-"):
        value = value[1:]
    return value.strip().strip('"').strip("'")


def _extract_envfile_path(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("EnvironmentFile="):
        return None
    raw = stripped.split("=", 1)[1]
    norm = _normalize_envfile(raw)
    if not norm:
        return None
    return norm


def _dropin_files(service: str) -> list[Path]:
    raw = _show_value(service, "DropInPaths")
    files: list[Path] = []
    for token in raw.split():
        p = Path(token.strip())
        if p.exists() and p.is_file():
            files.append(p)
    return files


def _fragment_file(service: str) -> Path | None:
    raw = _show_value(service, "FragmentPath")
    if not raw:
        return None
    p = Path(raw)
    if p.exists() and p.is_file():
        return p
    return None


def _service_line_order(service: str) -> list[Path]:
    order: list[Path] = []
    fragment = _fragment_file(service)
    if fragment is not None:
        order.append(fragment)
    order.extend(_dropin_files(service))
    return order


def _is_editable_dropin(path: Path) -> bool:
    return str(path).startswith("/etc/systemd/system/") and ".service.d/" in str(path)


def dedupe_service(
    service: str,
    *,
    apply: bool,
    remove_envfiles: set[str],
) -> tuple[int, int, int, list[str]]:
    seen_paths: set[str] = set()
    duplicates_to_remove: dict[Path, list[int]] = defaultdict(list)
    explicit_to_remove: dict[Path, list[int]] = defaultdict(list)
    notes: list[str] = []

    for unit_file in _service_line_order(service):
        try:
            lines = unit_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError as exc:
            notes.append(f"read_error file={unit_file} err={exc}")
            continue

        for idx, line in enumerate(lines, start=1):
            env_path = _extract_envfile_path(line)
            if env_path is None:
                continue
            if env_path in remove_envfiles:
                if _is_editable_dropin(unit_file):
                    explicit_to_remove[unit_file].append(idx)
                    continue
                notes.append(
                    f"remove_target_in_non_editable file={unit_file} line={idx} env={env_path}"
                )
            if env_path in seen_paths:
                if _is_editable_dropin(unit_file):
                    duplicates_to_remove[unit_file].append(idx)
                else:
                    notes.append(
                        f"duplicate_in_non_editable file={unit_file} line={idx} env={env_path}"
                    )
            else:
                seen_paths.add(env_path)

    duplicate_count = sum(len(v) for v in duplicates_to_remove.values())
    explicit_remove_count = sum(len(v) for v in explicit_to_remove.values())
    if not apply or (duplicate_count == 0 and explicit_remove_count == 0):
        return duplicate_count, explicit_remove_count, 0, notes

    changed_files = 0
    targets = set(duplicates_to_remove) | set(explicit_to_remove)
    for path in targets:
        remove_set = set(duplicates_to_remove.get(path, []))
        remove_set.update(explicit_to_remove.get(path, []))
        try:
            original_text = path.read_text(encoding="utf-8", errors="ignore")
            original = original_text.splitlines()
        except OSError as exc:
            notes.append(f"read_error file={path} err={exc}")
            continue
        updated = [line for i, line in enumerate(original, start=1) if i not in remove_set]
        if updated == original:
            continue
        text = "\n".join(updated)
        if original and original_text.endswith("\n"):
            text += "\n"
        try:
            path.write_text(text, encoding="utf-8")
            changed_files += 1
        except OSError as exc:
            notes.append(f"write_error file={path} err={exc}")

    return duplicate_count, explicit_remove_count, changed_files, notes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deduplicate redundant EnvironmentFile lines in systemd drop-ins."
    )
    parser.add_argument(
        "--services",
        nargs="+",
        default=["quant-*.service"],
        help="Service name globs (default: quant-*.service)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to drop-in files (default: dry-run).",
    )
    parser.add_argument(
        "--remove-envfile",
        action="append",
        default=[],
        help=(
            "Remove matching EnvironmentFile entries from editable drop-ins. "
            "Can be specified multiple times."
        ),
    )
    args = parser.parse_args()

    services = _list_services(args.services)
    if not services:
        print("No matching services found.")
        return 0

    remove_envfiles = {_normalize_envfile(item) for item in args.remove_envfile if item.strip()}
    total_duplicates = 0
    total_explicit_removed = 0
    total_changed_files = 0
    services_with_duplicates = 0

    for service in services:
        duplicate_count, explicit_remove_count, changed_files, notes = dedupe_service(
            service,
            apply=args.apply,
            remove_envfiles=remove_envfiles,
        )
        if duplicate_count > 0 or explicit_remove_count > 0:
            services_with_duplicates += 1
            total_duplicates += duplicate_count
            total_explicit_removed += explicit_remove_count
            total_changed_files += changed_files
            print(
                f"[{service}] duplicate_envfile_lines={duplicate_count} "
                f"remove_target_lines={explicit_remove_count} "
                f"changed_dropins={changed_files}"
            )
            for note in notes:
                print(f"  note: {note}")

    mode = "apply" if args.apply else "dry-run"
    print(
        f"mode={mode} services_scanned={len(services)} "
        f"services_with_duplicates={services_with_duplicates} "
        f"duplicate_lines={total_duplicates} "
        f"remove_target_lines={total_explicit_removed} "
        f"changed_dropins={total_changed_files}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
