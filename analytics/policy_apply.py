from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from analytics.policy_diff import apply_policy_diff, default_policy_snapshot
from utils.yaml_merge import deep_update

DEFAULT_OVERLAY_PATH = Path(os.getenv("POLICY_OVERLAY_PATH", "logs/policy_overlay.json"))
DEFAULT_HISTORY_DIR = Path(os.getenv("POLICY_HISTORY_DIR", "logs/policy_history"))
DEFAULT_LATEST_PATH = Path(os.getenv("POLICY_LATEST_PATH", "logs/policy_latest.json"))

REENTRY_CONFIG_PATH = Path(os.getenv("REENTRY_CONFIG_PATH", "config/worker_reentry.yaml"))
_TUNING_RUNTIME_DIR = os.getenv("TUNING_RUNTIME_DIR", "logs/tuning")
TUNING_OVERRIDES_PATH = Path(
    os.getenv("TUNING_OVERRIDES_PATH", f"{_TUNING_RUNTIME_DIR}/tuning_overrides.yaml")
)
TUNING_PRESETS_PATH = Path(os.getenv("TUNING_PRESETS_PATH", "config/tuning_presets.yaml"))
TUNING_OVERLAY_PATH = Path(os.getenv("TUNING_OVERLAY_PATH", f"{_TUNING_RUNTIME_DIR}/tuning_overlay.yaml"))


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))


def load_policy_snapshot(path: Path) -> Dict[str, Any]:
    payload = _read_json(path)
    if payload:
        return payload
    return default_policy_snapshot()


def save_policy_snapshot(path: Path, snapshot: Dict[str, Any]) -> None:
    _write_json(path, snapshot)


def _apply_reentry_overrides(overrides: Dict[str, Any], path: Path) -> bool:
    if yaml is None:
        logging.warning("[POLICY_APPLY] PyYAML missing; reentry overrides skipped.")
        return False
    if not overrides:
        return False
    base: Dict[str, Any] = {}
    if path.exists():
        try:
            base = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            base = {}
    merged = deep_update(base, overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    return True


def _apply_tuning_overrides(overrides: Dict[str, Any]) -> bool:
    if yaml is None:
        logging.warning("[POLICY_APPLY] PyYAML missing; tuning overrides skipped.")
        return False
    if not overrides:
        return False
    base: Dict[str, Any] = {}
    if TUNING_OVERRIDES_PATH.exists():
        try:
            base = yaml.safe_load(TUNING_OVERRIDES_PATH.read_text(encoding="utf-8")) or {}
        except Exception:
            base = {}
    merged = deep_update(base, overrides)
    TUNING_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TUNING_OVERRIDES_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)

    # regenerate overlay from presets + overrides
    try:
        presets = yaml.safe_load(TUNING_PRESETS_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        presets = {}
    overlay = deep_update(presets, merged)
    TUNING_OVERLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TUNING_OVERLAY_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(overlay, f, sort_keys=False)
    return True


def apply_policy_diff_to_paths(
    policy_diff: Dict[str, Any],
    *,
    overlay_path: Path = DEFAULT_OVERLAY_PATH,
    history_dir: Path = DEFAULT_HISTORY_DIR,
    latest_path: Path = DEFAULT_LATEST_PATH,
    apply_reentry: bool = True,
    apply_tuning: bool = True,
) -> Tuple[Dict[str, Any], bool, Dict[str, bool]]:
    base_snapshot = load_policy_snapshot(overlay_path)
    updated, changed = apply_policy_diff(base_snapshot, policy_diff)
    if changed:
        save_policy_snapshot(overlay_path, updated)
    save_policy_snapshot(latest_path, updated)

    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"{policy_diff.get('policy_id','policy')}.json"
    _write_json(history_path, policy_diff)

    applied_flags = {"reentry": False, "tuning": False}
    if apply_reentry and isinstance(policy_diff.get("reentry_overrides"), dict):
        applied_flags["reentry"] = _apply_reentry_overrides(policy_diff["reentry_overrides"], REENTRY_CONFIG_PATH)
    if apply_tuning and isinstance(policy_diff.get("tuning_overrides"), dict):
        applied_flags["tuning"] = _apply_tuning_overrides(policy_diff["tuning_overrides"])
    return updated, changed, applied_flags
