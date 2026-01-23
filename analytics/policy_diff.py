from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

POCKET_KEYS = ("macro", "micro", "scalp")

ROOT_ALLOWED_KEYS = {
    "policy_id",
    "generated_at",
    "source",
    "no_change",
    "reason",
    "patch",
    "notes",
    "metrics_window",
    "slo_guard",
    "reentry_overrides",
    "tuning_overrides",
}

PATCH_ALLOWED_KEYS = {
    "air_score",
    "uncertainty",
    "event_lock",
    "range_mode",
    "notes",
    "pockets",
}

POCKET_ALLOWED_KEYS = {
    "enabled",
    "bias",
    "confidence",
    "units_cap",
    "entry_gates",
    "exit_profile",
    "be_profile",
    "partial_profile",
    "strategies",
    "pending_orders",
}

ENTRY_GATE_KEYS = {"allow_new", "require_retest", "spread_ok", "drift_ok"}
EXIT_PROFILE_KEYS = {"reverse_threshold", "time_stop"}
BE_PROFILE_KEYS = {"enabled", "trigger_pips", "cooldown_sec", "lock_ratio", "min_lock_pips"}
PARTIAL_PROFILE_KEYS = {"thresholds_pips", "fractions", "min_units"}


POLICY_DIFF_SCHEMA = {
    "type": "object",
    "required": ["policy_id", "generated_at", "source", "no_change"],
    "properties": {
        "policy_id": {"type": "string"},
        "generated_at": {"type": "string"},
        "source": {"type": "string"},
        "no_change": {"type": "boolean"},
        "reason": {"type": "string"},
        "patch": {"type": "object"},
        "notes": {"type": "object"},
        "metrics_window": {"type": "object"},
        "slo_guard": {"type": "object"},
        "reentry_overrides": {"type": "object"},
        "tuning_overrides": {"type": "object"},
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _boolish(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _deep_merge(base: Any, patch: Any) -> Any:
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    merged: Dict[str, Any] = copy.deepcopy(base) if isinstance(base, dict) else {}
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged.get(key), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def default_policy_snapshot() -> Dict[str, Any]:
    from analysis.policy_bus import PolicySnapshot

    return PolicySnapshot(version=0, generated_ts=time.time()).to_dict()


def normalize_policy_diff(diff: Dict[str, Any], *, source: str | None = None) -> Dict[str, Any]:
    data = dict(diff or {})
    if source and not data.get("source"):
        data["source"] = source
    if not data.get("policy_id"):
        data["policy_id"] = f"policy-{int(time.time())}"
    if not data.get("generated_at"):
        data["generated_at"] = utc_now_iso()
    if "no_change" not in data:
        data["no_change"] = False
    data["no_change"] = _boolish(data.get("no_change"), default=False)
    if data.get("no_change") and not data.get("patch"):
        data.pop("patch", None)
    return data


def validate_policy_diff(diff: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(diff, dict):
        return ["policy_diff is not an object"]
    for key in diff.keys():
        if key not in ROOT_ALLOWED_KEYS:
            errors.append(f"unknown_root_key:{key}")
    for req in ("policy_id", "generated_at", "source", "no_change"):
        if req not in diff:
            errors.append(f"missing:{req}")
    if "patch" in diff and not isinstance(diff.get("patch"), dict):
        errors.append("patch_not_object")
    if not diff.get("no_change") and "patch" not in diff:
        errors.append("patch_required_when_no_change_false")

    patch = diff.get("patch") if isinstance(diff.get("patch"), dict) else {}
    for key in patch.keys():
        if key not in PATCH_ALLOWED_KEYS:
            errors.append(f"unknown_patch_key:{key}")
    pockets = patch.get("pockets")
    if pockets is not None and not isinstance(pockets, dict):
        errors.append("pockets_not_object")
    if isinstance(pockets, dict):
        for pocket, config in pockets.items():
            if pocket not in POCKET_KEYS:
                errors.append(f"unknown_pocket:{pocket}")
                continue
            if not isinstance(config, dict):
                errors.append(f"pocket_not_object:{pocket}")
                continue
            for key in config.keys():
                if key not in POCKET_ALLOWED_KEYS:
                    errors.append(f"unknown_pocket_key:{pocket}:{key}")
            for k, allowed in (
                ("entry_gates", ENTRY_GATE_KEYS),
                ("exit_profile", EXIT_PROFILE_KEYS),
                ("be_profile", BE_PROFILE_KEYS),
                ("partial_profile", PARTIAL_PROFILE_KEYS),
            ):
                payload = config.get(k)
                if payload is None:
                    continue
                if not isinstance(payload, dict):
                    errors.append(f"{k}_not_object:{pocket}")
                    continue
                for item in payload.keys():
                    if item not in allowed:
                        errors.append(f"unknown_{k}_key:{pocket}:{item}")

    reentry = diff.get("reentry_overrides")
    if reentry is not None and not isinstance(reentry, dict):
        errors.append("reentry_overrides_not_object")
    tuning = diff.get("tuning_overrides")
    if tuning is not None and not isinstance(tuning, dict):
        errors.append("tuning_overrides_not_object")
    return errors


def apply_policy_diff(
    base_snapshot: Dict[str, Any],
    policy_diff: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    if policy_diff.get("no_change"):
        return base_snapshot, False
    patch = policy_diff.get("patch") or {}
    updated = _deep_merge(base_snapshot, patch)
    try:
        updated["version"] = int(base_snapshot.get("version", 0)) + 1
    except Exception:
        updated["version"] = 1
    updated["generated_ts"] = time.time()
    notes = updated.get("notes")
    if not isinstance(notes, dict):
        notes = {}
    notes["policy_id"] = policy_diff.get("policy_id")
    notes["policy_source"] = policy_diff.get("source")
    notes["policy_reason"] = policy_diff.get("reason")
    notes["policy_applied_at"] = utc_now_iso()
    updated["notes"] = notes
    if policy_diff.get("policy_id"):
        updated["policy_id"] = policy_diff.get("policy_id")
    return updated, True


def policy_hash(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AggregateRow:
    key: Tuple[str, ...]
    trade_count: int
    wins: int
    gross_profit: float
    gross_loss: float
    total_pips: float
    avg_hold_minutes: float

    def as_summary(self) -> Dict[str, Any]:
        trade_count = max(0, int(self.trade_count))
        win_rate = (self.wins / trade_count) if trade_count else 0.0
        pf = (
            self.gross_profit / self.gross_loss
            if self.gross_loss > 1e-9
            else (None if self.gross_profit > 0 else 0.0)
        )
        avg_hold = self.avg_hold_minutes if trade_count else 0.0
        return {
            "trade_count": trade_count,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(pf, 4) if isinstance(pf, (int, float)) else pf,
            "total_pips": round(self.total_pips, 3),
            "avg_hold_minutes": round(avg_hold, 2),
        }


def aggregate_rows(
    rows: Iterable[Dict[str, Any]],
    keys: Tuple[str, ...],
) -> List[AggregateRow]:
    buckets: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(k) or "unknown") for k in keys)
        bucket = buckets.setdefault(
            key,
            {"trade_count": 0, "wins": 0, "gross_profit": 0.0, "gross_loss": 0.0, "total_pips": 0.0, "hold": 0.0},
        )
        trade_count = int(row.get("trade_count") or 0)
        bucket["trade_count"] += trade_count
        bucket["wins"] += int(row.get("wins") or 0)
        bucket["gross_profit"] += float(row.get("gross_profit") or 0.0)
        bucket["gross_loss"] += float(row.get("gross_loss") or 0.0)
        bucket["total_pips"] += float(row.get("total_pips") or 0.0)
        bucket["hold"] += float(row.get("avg_hold_minutes") or 0.0) * trade_count
    out: List[AggregateRow] = []
    for key, agg in buckets.items():
        trade_count = max(0, int(agg["trade_count"]))
        avg_hold = agg["hold"] / trade_count if trade_count else 0.0
        out.append(
            AggregateRow(
                key=key,
                trade_count=trade_count,
                wins=int(agg["wins"]),
                gross_profit=float(agg["gross_profit"]),
                gross_loss=float(agg["gross_loss"]),
                total_pips=float(agg["total_pips"]),
                avg_hold_minutes=float(avg_hold),
            )
        )
    return out
