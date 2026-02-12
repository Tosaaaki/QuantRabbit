"""
workers.common.pattern_gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pattern-based entry gate backed by logs/patterns.db.

This gate consumes the auto-updated pattern book outputs and applies:
- block on persistent bad patterns (e.g. quality=avoid)
- dynamic unit scaling from suggested_multiplier

Default behavior is conservative:
- no entry_thesis or generic pattern id => no-op
- insufficient samples => no-op
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from analysis.pattern_book import build_pattern_id

LOG = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _env_set(name: str, default_csv: str = "") -> set[str]:
    raw = os.getenv(name, default_csv)
    return {item.strip().lower() for item in str(raw).split(",") if item.strip()}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _normalize_side(side: Optional[str], units: int) -> str:
    if units > 0:
        return "long"
    if units < 0:
        return "short"
    key = str(side or "").strip().lower()
    if key in {"buy", "long", "open_long"}:
        return "long"
    if key in {"sell", "short", "open_short"}:
        return "short"
    return "unknown"


def _strategy_base(strategy_tag: Optional[str]) -> str:
    raw = str(strategy_tag or "").strip().lower()
    if not raw:
        return ""
    return raw.split("-", 1)[0].strip()


def _parse_pattern_tokens(pattern_id: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in str(pattern_id or "").split("|"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip().lower()
        value = value.strip().lower()
        if key:
            out[key] = value
    return out


def _is_generic_pattern_id(pattern_id: str) -> bool:
    tokens = _parse_pattern_tokens(pattern_id)
    keys = ("sg", "mtf", "hz", "ex", "rg", "pt")
    informative = 0
    for key in keys:
        value = str(tokens.get(key, "")).strip().lower()
        if value and value not in {"na", "none", "unknown"}:
            informative += 1
    return informative == 0


_ENABLED = _env_bool("ORDER_PATTERN_GATE_ENABLED", True)
_TTL_SEC = max(2.0, _env_float("ORDER_PATTERN_GATE_TTL_SEC", 20.0))
_DB_PATH = Path(os.getenv("ORDER_PATTERN_GATE_DB_PATH", "logs/patterns.db"))
_JSON_PATH = Path(os.getenv("ORDER_PATTERN_GATE_JSON_PATH", "config/pattern_book_deep.json"))
_GLOBAL_OPT_IN = _env_bool("ORDER_PATTERN_GATE_GLOBAL_OPT_IN", False)

_POCKET_ALLOWLIST = _env_set("ORDER_PATTERN_GATE_POCKET_ALLOWLIST", "micro,macro,scalp,scalp_fast")
_POCKET_BLOCKLIST = _env_set("ORDER_PATTERN_GATE_POCKET_BLOCKLIST", "")
_STRATEGY_ALLOWLIST = _env_set("ORDER_PATTERN_GATE_STRATEGY_ALLOWLIST", "")
_STRATEGY_BLOCKLIST = _env_set("ORDER_PATTERN_GATE_STRATEGY_BLOCKLIST", "")

_BLOCK_QUALITIES = _env_set("ORDER_PATTERN_GATE_BLOCK_QUALITIES", "avoid")
_REDUCE_QUALITIES = _env_set("ORDER_PATTERN_GATE_REDUCE_QUALITIES", "weak")

_SCALE_MIN_TRADES = max(1, _env_int("ORDER_PATTERN_GATE_SCALE_MIN_TRADES", 30))
_BLOCK_MIN_TRADES = max(_SCALE_MIN_TRADES, _env_int("ORDER_PATTERN_GATE_BLOCK_MIN_TRADES", 90))

_BLOCK_MAX_PVALUE = max(0.0, min(1.0, _env_float("ORDER_PATTERN_GATE_BLOCK_MAX_PVALUE", 0.35)))
_BLOCK_MAX_SCORE = _env_float("ORDER_PATTERN_GATE_BLOCK_MAX_SCORE", -0.9)

_ALLOW_BOOST = _env_bool("ORDER_PATTERN_GATE_ALLOW_BOOST", True)
_SCALE_MIN = max(0.1, _env_float("ORDER_PATTERN_GATE_SCALE_MIN", 0.70))
_SCALE_MAX = max(1.0, _env_float("ORDER_PATTERN_GATE_SCALE_MAX", 1.20))
_MIN_SCALE_DELTA = max(0.0, _env_float("ORDER_PATTERN_GATE_MIN_SCALE_DELTA", 0.03))
_REDUCE_FALLBACK_SCALE = max(0.2, min(1.0, _env_float("ORDER_PATTERN_GATE_REDUCE_FALLBACK_SCALE", 0.88)))

_DRIFT_PENALTY_ENABLED = _env_bool("ORDER_PATTERN_GATE_DRIFT_PENALTY_ENABLED", True)
_DRIFT_MIN_TRADES = max(1, _env_int("ORDER_PATTERN_GATE_DRIFT_MIN_TRADES", 40))
_DRIFT_DETERIORATION_SCALE = max(0.2, min(1.0, _env_float("ORDER_PATTERN_GATE_DRIFT_DETERIORATION_SCALE", 0.82)))
_DRIFT_SOFT_DETERIORATION_SCALE = max(
    0.2, min(1.0, _env_float("ORDER_PATTERN_GATE_DRIFT_SOFT_DETERIORATION_SCALE", 0.90))
)


@dataclass(frozen=True)
class PatternGateDecision:
    allowed: bool
    scale: float
    reason: str
    action: str
    pattern_id: str
    quality: str
    trades: int
    suggested_multiplier: float
    robust_score: float
    p_value: float
    source: str
    drift_state: Optional[str] = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "allowed": bool(self.allowed),
            "scale": round(float(self.scale), 4),
            "reason": str(self.reason),
            "action": str(self.action),
            "pattern_id": str(self.pattern_id),
            "quality": str(self.quality),
            "trades": int(self.trades),
            "suggested_multiplier": round(float(self.suggested_multiplier), 4),
            "robust_score": round(float(self.robust_score), 4),
            "p_value": round(float(self.p_value), 6),
            "source": str(self.source),
            "drift_state": str(self.drift_state) if self.drift_state else None,
        }


_CACHE_TS = 0.0
_CACHE_ROWS: dict[str, dict[str, Any]] = {}
_CACHE_DRIFT: dict[str, dict[str, Any]] = {}
_CACHE_SOURCE = "none"


def _should_use(strategy_tag: Optional[str], pocket: Optional[str]) -> bool:
    if not _ENABLED:
        return False
    pk = str(pocket or "").strip().lower()
    if not pk:
        return False
    if _POCKET_ALLOWLIST and pk not in _POCKET_ALLOWLIST:
        return False
    if _POCKET_BLOCKLIST and pk in _POCKET_BLOCKLIST:
        return False
    tag = str(strategy_tag or "").strip().lower()
    base = _strategy_base(strategy_tag)
    if _STRATEGY_BLOCKLIST and (tag in _STRATEGY_BLOCKLIST or base in _STRATEGY_BLOCKLIST):
        return False
    if _STRATEGY_ALLOWLIST and tag not in _STRATEGY_ALLOWLIST and base not in _STRATEGY_ALLOWLIST:
        return False
    return True


def _entry_opt_in(entry_thesis: Optional[dict], meta: Optional[dict]) -> bool:
    if _GLOBAL_OPT_IN:
        return True
    if isinstance(entry_thesis, dict):
        for key in ("pattern_gate_opt_in", "use_pattern_gate", "pattern_gate_enabled"):
            if bool(entry_thesis.get(key)):
                return True
    if isinstance(meta, dict):
        for key in ("pattern_gate_opt_in", "use_pattern_gate", "pattern_gate_enabled"):
            if bool(meta.get(key)):
                return True
    return False


def _load_from_db() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str]:
    if not _DB_PATH.exists():
        return {}, {}, "db_missing"

    rows: dict[str, dict[str, Any]] = {}
    drift: dict[str, dict[str, Any]] = {}
    try:
        con = sqlite3.connect(_DB_PATH, timeout=2.5)
        con.row_factory = sqlite3.Row
        try:
            for row in con.execute(
                """
                SELECT
                  pattern_id, trades, quality, suggested_multiplier,
                  robust_score, p_value, win_rate, avg_pips, profit_factor
                FROM pattern_scores
                """
            ).fetchall():
                pid = str(row["pattern_id"] or "")
                if not pid:
                    continue
                rows[pid] = {
                    "pattern_id": pid,
                    "trades": _safe_int(row["trades"]),
                    "quality": str(row["quality"] or "neutral").strip().lower(),
                    "suggested_multiplier": _safe_float(row["suggested_multiplier"], 1.0),
                    "robust_score": _safe_float(row["robust_score"], 0.0),
                    "p_value": _safe_float(row["p_value"], 1.0),
                    "win_rate": _safe_float(row["win_rate"], 0.0),
                    "avg_pips": _safe_float(row["avg_pips"], 0.0),
                    "profit_factor": _safe_float(row["profit_factor"], 0.0),
                }
        except sqlite3.OperationalError:
            rows = {}
        try:
            for row in con.execute(
                """
                SELECT pattern_id, drift_state, delta_avg_pips, delta_win_rate, p_value
                FROM pattern_drift
                """
            ).fetchall():
                pid = str(row["pattern_id"] or "")
                if not pid:
                    continue
                drift[pid] = {
                    "drift_state": str(row["drift_state"] or "").strip().lower(),
                    "delta_avg_pips": _safe_float(row["delta_avg_pips"], 0.0),
                    "delta_win_rate": _safe_float(row["delta_win_rate"], 0.0),
                    "p_value": _safe_float(row["p_value"], 1.0),
                }
        except sqlite3.OperationalError:
            drift = {}

        # Fallback to legacy action table if deep score table is not present yet.
        if not rows:
            try:
                for row in con.execute(
                    """
                    SELECT pattern_id, action, lot_multiplier, reason, trades
                    FROM pattern_actions
                    """
                ).fetchall():
                    pid = str(row["pattern_id"] or "")
                    if not pid:
                        continue
                    action = str(row["action"] or "neutral").strip().lower()
                    quality = {
                        "block": "avoid",
                        "reduce": "weak",
                        "boost": "candidate",
                        "neutral": "neutral",
                        "learn_only": "learn_only",
                    }.get(action, "neutral")
                    rows[pid] = {
                        "pattern_id": pid,
                        "trades": _safe_int(row["trades"]),
                        "quality": quality,
                        "suggested_multiplier": _safe_float(row["lot_multiplier"], 1.0),
                        "robust_score": 0.0,
                        "p_value": 1.0,
                    }
            except sqlite3.OperationalError:
                rows = {}
    except Exception as exc:
        LOG.debug("[PATTERN_GATE] db load failed: %s", exc)
        return {}, {}, "db_error"
    finally:
        try:
            con.close()
        except Exception:
            pass
    return rows, drift, "db"


def _load_from_json() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str]:
    if not _JSON_PATH.exists():
        return {}, {}, "json_missing"
    try:
        payload = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.debug("[PATTERN_GATE] json load failed: %s", exc)
        return {}, {}, "json_error"

    rows: dict[str, dict[str, Any]] = {}
    drift: dict[str, dict[str, Any]] = {}
    for bucket, quality in (("top_robust", "robust"), ("top_weak", "weak")):
        recs = payload.get(bucket)
        if not isinstance(recs, list):
            continue
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            pid = str(rec.get("pattern_id") or "")
            if not pid:
                continue
            rows[pid] = {
                "pattern_id": pid,
                "trades": _safe_int(rec.get("trades"), 0),
                "quality": str(rec.get("quality") or quality).strip().lower(),
                "suggested_multiplier": _safe_float(rec.get("suggested_multiplier"), 1.0),
                "robust_score": _safe_float(rec.get("robust_score"), 0.0),
                "p_value": _safe_float(rec.get("p_value"), 1.0),
            }
    drift_recs = payload.get("drift_alerts")
    if isinstance(drift_recs, list):
        for rec in drift_recs:
            if not isinstance(rec, dict):
                continue
            pid = str(rec.get("pattern_id") or "")
            if not pid:
                continue
            drift[pid] = {
                "drift_state": str(rec.get("drift_state") or "").strip().lower(),
                "delta_avg_pips": _safe_float(rec.get("delta_avg_pips"), 0.0),
                "delta_win_rate": _safe_float(rec.get("delta_win_rate"), 0.0),
                "p_value": _safe_float(rec.get("p_value"), 1.0),
            }
    return rows, drift, "json"


def _load_cache() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str]:
    global _CACHE_TS, _CACHE_ROWS, _CACHE_DRIFT, _CACHE_SOURCE
    now = time.monotonic()
    if _CACHE_ROWS and (now - _CACHE_TS) <= _TTL_SEC:
        return _CACHE_ROWS, _CACHE_DRIFT, _CACHE_SOURCE

    rows, drift, source = _load_from_db()
    if not rows:
        rows, drift, source = _load_from_json()

    _CACHE_TS = now
    _CACHE_ROWS = rows
    _CACHE_DRIFT = drift
    _CACHE_SOURCE = source
    return _CACHE_ROWS, _CACHE_DRIFT, _CACHE_SOURCE


def _resolve_scale(row: dict[str, Any], drift_row: Optional[dict[str, Any]]) -> float:
    quality = str(row.get("quality") or "").strip().lower()
    trades = _safe_int(row.get("trades"), 0)
    mult = _safe_float(row.get("suggested_multiplier"), 1.0)
    mult = max(0.05, mult)

    if not _ALLOW_BOOST and mult > 1.0:
        mult = 1.0
    if quality in _REDUCE_QUALITIES and mult >= 1.0:
        mult = min(mult, _REDUCE_FALLBACK_SCALE)
    if trades < _SCALE_MIN_TRADES:
        mult = 1.0

    if _DRIFT_PENALTY_ENABLED and trades >= _DRIFT_MIN_TRADES and isinstance(drift_row, dict):
        state = str(drift_row.get("drift_state") or "").strip().lower()
        if state == "deterioration":
            mult = min(mult, _DRIFT_DETERIORATION_SCALE)
        elif state == "soft_deterioration":
            mult = min(mult, _DRIFT_SOFT_DETERIORATION_SCALE)

    mult = max(_SCALE_MIN, min(_SCALE_MAX, mult))
    if abs(mult - 1.0) < _MIN_SCALE_DELTA:
        return 1.0
    return mult


def decide(
    *,
    strategy_tag: Optional[str],
    pocket: Optional[str],
    side: Optional[str],
    units: int,
    entry_thesis: Optional[dict],
    meta: Optional[dict] = None,
) -> Optional[PatternGateDecision]:
    if not _should_use(strategy_tag, pocket):
        return None
    if not _entry_opt_in(entry_thesis, meta):
        return None
    if not isinstance(entry_thesis, dict):
        return None
    if not units:
        return None

    side_key = _normalize_side(side, units)
    if side_key == "unknown":
        return None

    strategy_fallback = str(strategy_tag or entry_thesis.get("strategy_tag") or "").strip()
    if not strategy_fallback:
        return None
    pattern_id = build_pattern_id(
        entry_thesis=entry_thesis,
        units=units,
        pocket=str(pocket or ""),
        strategy_tag_fallback=strategy_fallback,
    )
    if not pattern_id or _is_generic_pattern_id(pattern_id):
        return None

    rows, drift_map, source = _load_cache()
    row = rows.get(pattern_id)
    if not row:
        return None

    quality = str(row.get("quality") or "neutral").strip().lower()
    trades = _safe_int(row.get("trades"), 0)
    suggested = _safe_float(row.get("suggested_multiplier"), 1.0)
    robust_score = _safe_float(row.get("robust_score"), 0.0)
    p_value = _safe_float(row.get("p_value"), 1.0)
    drift_row = drift_map.get(pattern_id)
    drift_state = (
        str(drift_row.get("drift_state")).strip().lower()
        if isinstance(drift_row, dict) and drift_row.get("drift_state")
        else None
    )

    if (
        quality in _BLOCK_QUALITIES
        and trades >= _BLOCK_MIN_TRADES
        and robust_score <= _BLOCK_MAX_SCORE
        and p_value <= _BLOCK_MAX_PVALUE
    ):
        return PatternGateDecision(
            allowed=False,
            scale=0.0,
            reason="pattern_avoid",
            action="block",
            pattern_id=pattern_id,
            quality=quality,
            trades=trades,
            suggested_multiplier=suggested,
            robust_score=robust_score,
            p_value=p_value,
            source=source,
            drift_state=drift_state,
        )

    scale = _resolve_scale(row, drift_row)
    if scale == 1.0:
        return PatternGateDecision(
            allowed=True,
            scale=1.0,
            reason="pattern_neutral",
            action="pass",
            pattern_id=pattern_id,
            quality=quality,
            trades=trades,
            suggested_multiplier=suggested,
            robust_score=robust_score,
            p_value=p_value,
            source=source,
            drift_state=drift_state,
        )

    reason = "pattern_reduce" if scale < 1.0 else "pattern_boost"
    return PatternGateDecision(
        allowed=True,
        scale=scale,
        reason=reason,
        action="scale",
        pattern_id=pattern_id,
        quality=quality,
        trades=trades,
        suggested_multiplier=suggested,
        robust_score=robust_score,
        p_value=p_value,
        source=source,
        drift_state=drift_state,
    )
