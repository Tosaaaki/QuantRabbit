"""
LLM-backed decision brain for per-strategy entry gating.
Each strategy_tag+pocket pair keeps its own memory and decision cache.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from utils.ollama_client import call_ollama_chat_json
from utils.vertex_client import call_vertex_text
from utils.metrics_logger import log_metric

LOG = logging.getLogger(__name__)

_DB_PATH = pathlib.Path(os.getenv("BRAIN_DB_PATH", "logs/brain_state.db"))
_TRADES_DB_PATH = pathlib.Path(os.getenv("BRAIN_TRADES_DB_PATH", "logs/trades.db"))
_DB_TIMEOUT = float(os.getenv("BRAIN_DB_TIMEOUT_SEC", "3.0"))

_ENABLED = os.getenv("BRAIN_ENABLED", "0").strip().lower() not in {"", "0", "false", "no", "off"}
_ALLOWLIST = {item.strip().lower() for item in os.getenv("BRAIN_STRATEGY_ALLOWLIST", "").split(",") if item.strip()}
_BLOCKLIST = {item.strip().lower() for item in os.getenv("BRAIN_STRATEGY_BLOCKLIST", "").split(",") if item.strip()}
_POCKET_ALLOWLIST = {item.strip().lower() for item in os.getenv("BRAIN_POCKET_ALLOWLIST", "").split(",") if item.strip()}
_SAMPLE_RATE = max(0.0, min(1.0, float(os.getenv("BRAIN_SAMPLE_RATE", "1.0") or 1.0)))

_TTL_SEC = max(5.0, float(os.getenv("BRAIN_TTL_SEC", "90") or 90.0))
_MEMORY_TTL_H = max(1.0, float(os.getenv("BRAIN_MEMORY_TTL_H", "72") or 72.0))
_MAX_CONTEXT_CHARS = max(200, int(float(os.getenv("BRAIN_MAX_CONTEXT_CHARS", "1200") or 1200)))
_MIN_SCALE = max(0.05, float(os.getenv("BRAIN_MIN_SCALE", "0.2") or 0.2))
_BACKEND = (os.getenv("BRAIN_BACKEND", "vertex") or "vertex").strip().lower()
if _BACKEND not in {"vertex", "ollama"}:
    _BACKEND = "vertex"

_VERTEX_MODEL = (
    os.getenv("BRAIN_VERTEX_MODEL", "")
    or os.getenv("VERTEX_DECIDER_MODEL")
    or os.getenv("VERTEX_MODEL")
    or "gemini-2.0-flash"
)
_OLLAMA_MODEL = (os.getenv("BRAIN_OLLAMA_MODEL", "gpt-oss:20b") or "gpt-oss:20b").strip()
_OLLAMA_URL = (
    os.getenv("BRAIN_OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
    or "http://127.0.0.1:11434/api/chat"
).strip()

_TEMP = max(0.0, min(1.0, float(os.getenv("BRAIN_TEMPERATURE", "0.2") or 0.2)))
_MAX_TOKENS = max(64, int(float(os.getenv("BRAIN_MAX_TOKENS", "256") or 256)))
_TIMEOUT_SEC = max(2.0, float(os.getenv("BRAIN_TIMEOUT_SEC", "6") or 6))
_FAIL_POLICY = (os.getenv("BRAIN_FAIL_POLICY", "allow") or "allow").strip().lower()
if _FAIL_POLICY not in {"allow", "reduce", "block"}:
    _FAIL_POLICY = "allow"

_PERSONA_ENABLED = os.getenv("BRAIN_PERSONA_ENABLED", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
_PERSONA_MODE = (os.getenv("BRAIN_PERSONA_MODE", "auto") or "auto").strip().lower()
_PERSONA_DEFAULT = os.getenv(
    "BRAIN_PERSONA_DEFAULT",
    "Seasoned discretionary FX trader focused on risk-first entries.",
)
_RAW_PERSONA_OVERRIDES = os.getenv("BRAIN_PERSONA_OVERRIDES", "")

_POCKET_TRAITS = {
    "macro": "Patient macro trader; values regime alignment and avoids noise.",
    "micro": "Tactical micro trader; values precise timing and quick invalidation.",
    "scalp": "High-speed scalper; avoids hesitation and rejects low edge setups.",
    "scalp_fast": "Ultra-fast scalp; only highest edge entries, minimal tolerance.",
}

_PERSONA_PRESETS = {
    "trend": {
        "name": "Trend Hunter",
        "traits": "Prefers strong directional alignment and momentum follow-through.",
        "bias": "Allow only when trend evidence is clear; otherwise reduce or block.",
        "avoid": "Avoid mean-reversion entries or weak trend conditions.",
    },
    "momentum": {
        "name": "Momentum Specialist",
        "traits": "Enters on acceleration and continuation, avoids fading moves.",
        "bias": "Allow only when acceleration/impulse is present.",
        "avoid": "Avoid chop, fading, or low-volatility setups.",
    },
    "range": {
        "name": "Range Fader",
        "traits": "Trades mean reversion; seeks stretched moves back to mean.",
        "bias": "Allow only when range/mean-revert signals are present.",
        "avoid": "Avoid breakout or strong trend continuation entries.",
    },
    "pullback": {
        "name": "Pullback Sniper",
        "traits": "Waits for retrace to high-probability zones in trend.",
        "bias": "Allow only when pullback quality is clear; otherwise reduce.",
        "avoid": "Avoid chasing late entries or shallow/noisy pullbacks.",
    },
    "scalp": {
        "name": "Precision Scalper",
        "traits": "Seeks tight execution; rejects anything ambiguous.",
        "bias": "Prefer reduce/block if spread or context is uncertain.",
        "avoid": "Avoid slow or wide-spread entries.",
    },
    "neutral": {
        "name": "Balanced Trader",
        "traits": "Risk-first, selective, avoids low edge entries.",
        "bias": "If uncertain, reduce or block.",
        "avoid": "Avoid conflicting signals.",
    },
}

_COST_INPUT_PER_1K = float(os.getenv("BRAIN_COST_INPUT_PER_1K", "0") or 0.0)
_COST_OUTPUT_PER_1K = float(os.getenv("BRAIN_COST_OUTPUT_PER_1K", "0") or 0.0)

_PROMPT_PROFILE_PATH = pathlib.Path(
    os.getenv("BRAIN_PROMPT_PROFILE_PATH", "config/brain_prompt_profile.json")
)
_PROMPT_PROFILE_TTL_SEC = max(
    5.0, float(os.getenv("BRAIN_PROMPT_PROFILE_TTL_SEC", "30") or 30.0)
)
_PROMPT_EXTRA_RULES_MAX = max(
    1, int(float(os.getenv("BRAIN_PROMPT_EXTRA_RULES_MAX", "8") or 8))
)
_PROMPT_EXTRA_RULE_LEN = max(
    24, int(float(os.getenv("BRAIN_PROMPT_EXTRA_RULE_LEN", "160") or 160))
)

_PROMPT_AUTOTUNE_ENABLED = os.getenv("BRAIN_PROMPT_AUTO_TUNE_ENABLED", "0").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
_PROMPT_AUTOTUNE_INTERVAL_SEC = max(
    60.0, float(os.getenv("BRAIN_PROMPT_AUTO_TUNE_INTERVAL_SEC", "1800") or 1800.0)
)
_PROMPT_AUTOTUNE_MIN_DECISIONS = max(
    10, int(float(os.getenv("BRAIN_PROMPT_AUTO_TUNE_MIN_DECISIONS", "120") or 120))
)
_PROMPT_AUTOTUNE_LOOKBACK_HOURS = max(
    1.0, float(os.getenv("BRAIN_PROMPT_AUTO_TUNE_LOOKBACK_HOURS", "24") or 24.0)
)
_PROMPT_AUTOTUNE_TIMEOUT_SEC = max(
    1.0, float(os.getenv("BRAIN_PROMPT_AUTO_TUNE_TIMEOUT_SEC", "5") or 5.0)
)
_PROMPT_AUTOTUNE_TEMP = max(
    0.0, min(1.0, float(os.getenv("BRAIN_PROMPT_AUTO_TUNE_TEMP", "0.1") or 0.1))
)
_PROMPT_AUTOTUNE_MAX_TOKENS = max(
    128, int(float(os.getenv("BRAIN_PROMPT_AUTO_TUNE_MAX_TOKENS", "512") or 512))
)
_PROMPT_AUTOTUNE_MODEL = (
    os.getenv("BRAIN_PROMPT_AUTO_TUNE_MODEL", "")
    or os.getenv("BRAIN_OLLAMA_MODEL", "")
    or _OLLAMA_MODEL
).strip()
_PROMPT_AUTOTUNE_URL = (
    os.getenv("BRAIN_PROMPT_AUTO_TUNE_URL", "")
    or os.getenv("BRAIN_OLLAMA_URL", "")
    or _OLLAMA_URL
).strip()

_CACHE: dict[tuple[str, str], tuple[float, "BrainDecision"]] = {}
_PROMPT_PROFILE_CACHE: tuple[float, dict[str, Any]] = (0.0, {})
_PROMPT_PROFILE_LOCK = threading.Lock()
_PROMPT_AUTOTUNE_LOCK = threading.Lock()
_LAST_PROMPT_AUTOTUNE_TS = 0.0


@dataclass(frozen=True)
class BrainDecision:
    allowed: bool
    scale: float
    reason: str
    action: str
    memory: Optional[str] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_schema() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_DB_PATH, timeout=_DB_TIMEOUT)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS brain_memory (
                strategy_tag TEXT NOT NULL,
                pocket TEXT NOT NULL,
                memory TEXT,
                last_action TEXT,
                last_scale REAL,
                last_reason TEXT,
                updated_at TEXT,
                last_ts REAL,
                PRIMARY KEY (strategy_tag, pocket)
            )
            """
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_brain_memory_updated ON brain_memory(updated_at)"
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS brain_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                ts_epoch REAL NOT NULL,
                strategy_tag TEXT NOT NULL,
                pocket TEXT NOT NULL,
                side TEXT,
                units INTEGER,
                sl_price REAL,
                tp_price REAL,
                confidence REAL,
                client_order_id TEXT,
                backend TEXT,
                source TEXT,
                llm_ok INTEGER NOT NULL,
                latency_ms REAL,
                action TEXT,
                allowed INTEGER,
                scale REAL,
                reason TEXT,
                memory_before TEXT,
                memory_after TEXT,
                profile_version TEXT,
                context_json TEXT,
                response_json TEXT,
                error TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS brain_prompt_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                ts_epoch REAL NOT NULL,
                lookback_hours REAL NOT NULL,
                decision_count INTEGER NOT NULL,
                model TEXT,
                url TEXT,
                applied INTEGER NOT NULL,
                profile_version TEXT,
                summary_json TEXT,
                response_json TEXT,
                error TEXT
            )
            """
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_brain_decisions_ts ON brain_decisions(ts_epoch)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_brain_decisions_strategy ON brain_decisions(strategy_tag, pocket, ts_epoch)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_brain_decisions_client ON brain_decisions(client_order_id, ts_epoch)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_brain_prompt_runs_ts ON brain_prompt_runs(ts_epoch)"
        )
        con.commit()
    finally:
        try:
            con.close()
        except Exception:
            pass


def _should_use(strategy_tag: Optional[str], pocket: Optional[str]) -> bool:
    if not _ENABLED:
        return False
    if not strategy_tag or not pocket:
        return False
    if _BLOCKLIST:
        tag_key = strategy_tag.strip().lower()
        base = tag_key.split("-", 1)[0]
        if tag_key in _BLOCKLIST or base in _BLOCKLIST:
            return False
    if _ALLOWLIST:
        tag_key = strategy_tag.strip().lower()
        base = tag_key.split("-", 1)[0]
        if tag_key not in _ALLOWLIST and base not in _ALLOWLIST:
            return False
    if _POCKET_ALLOWLIST:
        if pocket.strip().lower() not in _POCKET_ALLOWLIST:
            return False
    if _SAMPLE_RATE < 1.0 and random.random() > _SAMPLE_RATE:
        return False
    return True


def _load_memory(strategy_tag: str, pocket: str) -> Optional[str]:
    if not _DB_PATH.exists():
        return None
    cutoff = time.time() - (_MEMORY_TTL_H * 3600.0)
    con = sqlite3.connect(_DB_PATH, timeout=_DB_TIMEOUT)
    try:
        row = con.execute(
            """
            SELECT memory, last_ts
            FROM brain_memory
            WHERE strategy_tag = ? AND pocket = ?
            """,
            (strategy_tag, pocket),
        ).fetchone()
    except Exception:
        row = None
    finally:
        try:
            con.close()
        except Exception:
            pass
    if not row:
        return None
    memory, last_ts = row
    try:
        ts = float(last_ts or 0.0)
    except Exception:
        ts = 0.0
    if ts <= 0 or ts < cutoff:
        return None
    if not memory:
        return None
    return str(memory)


def _save_memory(
    strategy_tag: str,
    pocket: str,
    *,
    memory: Optional[str],
    decision: BrainDecision,
) -> None:
    _ensure_schema()
    con = sqlite3.connect(_DB_PATH, timeout=_DB_TIMEOUT)
    try:
        con.execute(
            """
            INSERT INTO brain_memory(strategy_tag, pocket, memory, last_action, last_scale, last_reason, updated_at, last_ts)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(strategy_tag, pocket) DO UPDATE SET
                memory=excluded.memory,
                last_action=excluded.last_action,
                last_scale=excluded.last_scale,
                last_reason=excluded.last_reason,
                updated_at=excluded.updated_at,
                last_ts=excluded.last_ts
            """,
            (
                strategy_tag,
                pocket,
                memory,
                decision.action,
                float(decision.scale),
                decision.reason,
                _now_iso(),
                time.time(),
            ),
        )
        con.commit()
    except Exception:
        pass
    finally:
        try:
            con.close()
        except Exception:
            pass


def _stringify(obj: Any, limit: int) -> str:
    if obj is None:
        return ""
    try:
        text = json.dumps(obj, ensure_ascii=True, default=str)
    except Exception:
        text = str(obj)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _coerce_prompt_profile(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    rules: list[str] = []
    for item in raw.get("extra_rules", []) or []:
        text = str(item or "").strip()
        if not text:
            continue
        if len(text) > _PROMPT_EXTRA_RULE_LEN:
            text = text[:_PROMPT_EXTRA_RULE_LEN].rstrip() + "..."
        rules.append(text)
        if len(rules) >= _PROMPT_EXTRA_RULES_MAX:
            break
    profile: dict[str, Any] = {
        "version": str(raw.get("version") or "").strip() or "v1",
        "updated_at": str(raw.get("updated_at") or "").strip() or _now_iso(),
        "extra_rules": rules,
    }
    focus = str(raw.get("focus") or "").strip()
    if focus:
        profile["focus"] = focus[:_PROMPT_EXTRA_RULE_LEN]
    risk_bias = str(raw.get("risk_bias") or "").strip()
    if risk_bias:
        profile["risk_bias"] = risk_bias[:_PROMPT_EXTRA_RULE_LEN]
    return profile


def _profile_version(profile: Optional[dict[str, Any]]) -> str:
    if not isinstance(profile, dict):
        return "none"
    text = str(profile.get("version") or "").strip()
    return text or "none"


def _load_prompt_profile(force: bool = False) -> dict[str, Any]:
    global _PROMPT_PROFILE_CACHE
    now = time.monotonic()
    with _PROMPT_PROFILE_LOCK:
        cached_ts, cached = _PROMPT_PROFILE_CACHE
        if (
            not force
            and cached
            and now - cached_ts <= _PROMPT_PROFILE_TTL_SEC
        ):
            return dict(cached)
        profile: dict[str, Any] = {}
        if _PROMPT_PROFILE_PATH.exists():
            try:
                raw = json.loads(_PROMPT_PROFILE_PATH.read_text(encoding="utf-8"))
                profile = _coerce_prompt_profile(raw)
            except Exception:
                profile = {}
        _PROMPT_PROFILE_CACHE = (now, profile)
        return dict(profile)


def _format_prompt_profile(profile: dict[str, Any]) -> str:
    parts: list[str] = []
    rules = profile.get("extra_rules") or []
    if isinstance(rules, list) and rules:
        parts.append("Adaptive rules (recent live outcomes):")
        for item in rules:
            text = str(item or "").strip()
            if text:
                parts.append(f"- {text}")
    focus = str(profile.get("focus") or "").strip()
    if focus:
        parts.append(f"Focus: {focus}")
    risk_bias = str(profile.get("risk_bias") or "").strip()
    if risk_bias:
        parts.append(f"Risk bias: {risk_bias}")
    if not parts:
        return ""
    return "\n".join(parts)


def _log_decision_row(
    *,
    strategy_tag: str,
    pocket: str,
    side: str,
    units: int,
    sl_price: Optional[float],
    tp_price: Optional[float],
    confidence: Optional[int],
    client_order_id: Optional[str],
    backend: str,
    source: str,
    llm_ok: bool,
    latency_ms: float,
    decision: BrainDecision,
    memory_before: Optional[str],
    memory_after: Optional[str],
    profile_version: str,
    context: Optional[dict[str, Any]],
    payload: Optional[dict[str, Any]],
    error: Optional[str] = None,
) -> None:
    _ensure_schema()
    con = sqlite3.connect(_DB_PATH, timeout=_DB_TIMEOUT)
    try:
        con.execute(
            """
            INSERT INTO brain_decisions(
                ts, ts_epoch, strategy_tag, pocket, side, units, sl_price, tp_price,
                confidence, client_order_id, backend, source, llm_ok, latency_ms,
                action, allowed, scale, reason, memory_before, memory_after,
                profile_version, context_json, response_json, error
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                _now_iso(),
                float(time.time()),
                strategy_tag,
                pocket,
                side,
                int(units),
                sl_price,
                tp_price,
                float(confidence) if confidence is not None else None,
                str(client_order_id or "").strip() or None,
                backend,
                source,
                1 if llm_ok else 0,
                float(latency_ms),
                decision.action,
                1 if decision.allowed else 0,
                float(decision.scale),
                decision.reason,
                memory_before,
                memory_after,
                profile_version,
                _stringify(context or {}, 4000),
                _stringify(payload or {}, 4000),
                str(error or "").strip() or None,
            ),
        )
        con.commit()
    except Exception:
        pass
    finally:
        try:
            con.close()
        except Exception:
            pass


def _collect_autotune_summary(lookback_hours: float) -> dict[str, Any]:
    cutoff_epoch = time.time() - (max(1.0, lookback_hours) * 3600.0)
    summary: dict[str, Any] = {
        "lookback_hours": float(lookback_hours),
        "decision_count": 0,
        "action_counts": {},
        "source_counts": {},
        "strategy_action_counts": [],
        "reason_counts": [],
        "filled_trade_outcome": {},
    }
    if not _DB_PATH.exists():
        return summary
    con = sqlite3.connect(_DB_PATH, timeout=_DB_TIMEOUT)
    try:
        row = con.execute(
            "SELECT COUNT(*) FROM brain_decisions WHERE ts_epoch >= ?",
            (cutoff_epoch,),
        ).fetchone()
        summary["decision_count"] = int(row[0] or 0) if row else 0
        for action, count in con.execute(
            """
            SELECT COALESCE(action, 'UNKNOWN') AS action, COUNT(*) AS c
            FROM brain_decisions
            WHERE ts_epoch >= ?
            GROUP BY action
            ORDER BY c DESC
            """,
            (cutoff_epoch,),
        ).fetchall():
            summary["action_counts"][str(action)] = int(count or 0)
        for source, count in con.execute(
            """
            SELECT COALESCE(source, 'unknown') AS source, COUNT(*) AS c
            FROM brain_decisions
            WHERE ts_epoch >= ?
            GROUP BY source
            ORDER BY c DESC
            """,
            (cutoff_epoch,),
        ).fetchall():
            summary["source_counts"][str(source)] = int(count or 0)
        for strat, pocket, action, count in con.execute(
            """
            SELECT strategy_tag, pocket, COALESCE(action, 'UNKNOWN') AS action, COUNT(*) AS c
            FROM brain_decisions
            WHERE ts_epoch >= ?
            GROUP BY strategy_tag, pocket, action
            ORDER BY c DESC
            LIMIT 30
            """,
            (cutoff_epoch,),
        ).fetchall():
            summary["strategy_action_counts"].append(
                {
                    "strategy_tag": str(strat or ""),
                    "pocket": str(pocket or ""),
                    "action": str(action or "UNKNOWN"),
                    "count": int(count or 0),
                }
            )
        for action, reason, count in con.execute(
            """
            SELECT COALESCE(action, 'UNKNOWN') AS action, COALESCE(reason, '') AS reason, COUNT(*) AS c
            FROM brain_decisions
            WHERE ts_epoch >= ?
            GROUP BY action, reason
            ORDER BY c DESC
            LIMIT 40
            """,
            (cutoff_epoch,),
        ).fetchall():
            summary["reason_counts"].append(
                {
                    "action": str(action or "UNKNOWN"),
                    "reason": str(reason or ""),
                    "count": int(count or 0),
                }
            )
    except Exception:
        return summary
    finally:
        try:
            con.close()
        except Exception:
            pass

    if not _TRADES_DB_PATH.exists():
        return summary

    uri = f"file:{_TRADES_DB_PATH}?mode=ro"
    tcon = sqlite3.connect(uri, uri=True, timeout=_DB_TIMEOUT)
    try:
        cur = tcon.execute(
            """
            SELECT
              d.action,
              COUNT(*) AS trades,
              SUM(CASE WHEN t.realized_pl > 0 THEN 1 ELSE 0 END) AS wins,
              AVG(t.pl_pips) AS avg_pips,
              AVG(t.realized_pl) AS avg_realized,
              SUM(CASE WHEN t.realized_pl > 0 THEN t.realized_pl ELSE 0 END) AS gross_win,
              SUM(CASE WHEN t.realized_pl < 0 THEN -t.realized_pl ELSE 0 END) AS gross_loss
            FROM brain_decisions d
            JOIN trades t
              ON t.client_order_id = d.client_order_id
            WHERE d.ts_epoch >= ?
              AND d.action IN ('ALLOW','REDUCE')
              AND t.close_time IS NOT NULL
            GROUP BY d.action
            """,
            (cutoff_epoch,),
        )
        for action, trades, wins, avg_pips, avg_realized, gross_win, gross_loss in cur.fetchall():
            trades_n = int(trades or 0)
            wins_n = int(wins or 0)
            gross_win_f = float(gross_win or 0.0)
            gross_loss_f = float(gross_loss or 0.0)
            pf = (
                gross_win_f / gross_loss_f
                if gross_loss_f > 1e-9
                else (gross_win_f if gross_win_f > 0 else 0.0)
            )
            summary["filled_trade_outcome"][str(action)] = {
                "trades": trades_n,
                "wins": wins_n,
                "win_rate": round(wins_n / trades_n, 4) if trades_n > 0 else 0.0,
                "avg_pips": round(float(avg_pips or 0.0), 4),
                "avg_realized": round(float(avg_realized or 0.0), 6),
                "profit_factor": round(float(pf), 4),
            }
    except Exception:
        pass
    finally:
        try:
            tcon.close()
        except Exception:
            pass
    return summary


def _record_prompt_run(
    *,
    lookback_hours: float,
    decision_count: int,
    applied: bool,
    profile_version: str,
    summary: dict[str, Any],
    response: Optional[dict[str, Any]],
    error: Optional[str] = None,
) -> None:
    _ensure_schema()
    con = sqlite3.connect(_DB_PATH, timeout=_DB_TIMEOUT)
    try:
        con.execute(
            """
            INSERT INTO brain_prompt_runs(
                ts, ts_epoch, lookback_hours, decision_count, model, url, applied,
                profile_version, summary_json, response_json, error
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                _now_iso(),
                float(time.time()),
                float(lookback_hours),
                int(decision_count),
                _PROMPT_AUTOTUNE_MODEL,
                _PROMPT_AUTOTUNE_URL,
                1 if applied else 0,
                profile_version,
                _stringify(summary, 10000),
                _stringify(response or {}, 10000),
                str(error or "").strip() or None,
            ),
        )
        con.commit()
    except Exception:
        pass
    finally:
        try:
            con.close()
        except Exception:
            pass


def _write_prompt_profile(profile_payload: dict[str, Any]) -> dict[str, Any]:
    global _PROMPT_PROFILE_CACHE
    _PROMPT_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged = _load_prompt_profile(force=True)
    merged.update(_coerce_prompt_profile(profile_payload))
    merged["updated_at"] = _now_iso()
    version = str(merged.get("version") or "").strip() or "v1"
    merged["version"] = version
    tmp_path = _PROMPT_PROFILE_PATH.with_suffix(_PROMPT_PROFILE_PATH.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(merged, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(_PROMPT_PROFILE_PATH)
    with _PROMPT_PROFILE_LOCK:
        _PROMPT_PROFILE_CACHE = (time.monotonic(), dict(merged))
    return merged


def _maybe_autotune_prompt_profile() -> None:
    global _LAST_PROMPT_AUTOTUNE_TS
    if not _PROMPT_AUTOTUNE_ENABLED:
        return
    now_epoch = time.time()
    if now_epoch - _LAST_PROMPT_AUTOTUNE_TS < _PROMPT_AUTOTUNE_INTERVAL_SEC:
        return
    if not _PROMPT_AUTOTUNE_LOCK.acquire(blocking=False):
        return
    try:
        now_epoch = time.time()
        if now_epoch - _LAST_PROMPT_AUTOTUNE_TS < _PROMPT_AUTOTUNE_INTERVAL_SEC:
            return
        # Throttle checks even when summary is insufficient to avoid scanning every decision.
        _LAST_PROMPT_AUTOTUNE_TS = now_epoch
        summary = _collect_autotune_summary(_PROMPT_AUTOTUNE_LOOKBACK_HOURS)
        decision_count = int(summary.get("decision_count") or 0)
        if decision_count < _PROMPT_AUTOTUNE_MIN_DECISIONS:
            return
        current_profile = _load_prompt_profile(force=True)
        prompt = (
            "You optimize an FX decision gate prompt for USD/JPY scalping.\n"
            "Given recent live decision statistics and outcomes, update the prompt profile.\n"
            "Return JSON only with schema:\n"
            "{\n"
            '  "version": "v2",\n'
            '  "focus": "one short sentence",\n'
            '  "risk_bias": "one short sentence",\n'
            '  "extra_rules": ["rule1", "rule2"]\n'
            "}\n"
            "Constraints:\n"
            "- extra_rules must be imperative, concrete, and <=160 chars each.\n"
            "- keep 3-8 rules.\n"
            "- prioritize protecting downside and reducing low-edge entries.\n"
            "- do not include markdown.\n\n"
            f"Current profile: {_stringify(current_profile, 6000)}\n"
            f"Recent summary: {_stringify(summary, 9000)}\n"
        )
        response = call_ollama_chat_json(
            prompt,
            model=_PROMPT_AUTOTUNE_MODEL,
            url=_PROMPT_AUTOTUNE_URL,
            timeout_sec=_PROMPT_AUTOTUNE_TIMEOUT_SEC,
            temperature=_PROMPT_AUTOTUNE_TEMP,
            max_tokens=_PROMPT_AUTOTUNE_MAX_TOKENS,
        )
        if not isinstance(response, dict):
            _record_prompt_run(
                lookback_hours=_PROMPT_AUTOTUNE_LOOKBACK_HOURS,
                decision_count=decision_count,
                applied=False,
                profile_version=_profile_version(current_profile),
                summary=summary,
                response=None,
                error="no_response",
            )
            return
        merged = _write_prompt_profile(response)
        _record_prompt_run(
            lookback_hours=_PROMPT_AUTOTUNE_LOOKBACK_HOURS,
            decision_count=decision_count,
            applied=True,
            profile_version=_profile_version(merged),
            summary=summary,
            response=response,
            error=None,
        )
        try:
            log_metric(
                "brain_prompt_autotune_applied",
                1.0,
                tags={
                    "version": _profile_version(merged),
                    "decisions": str(decision_count),
                },
            )
        except Exception:
            pass
    except Exception as exc:
        _record_prompt_run(
            lookback_hours=_PROMPT_AUTOTUNE_LOOKBACK_HOURS,
            decision_count=0,
            applied=False,
            profile_version="none",
            summary={},
            response=None,
            error=f"exception:{exc}",
        )
    finally:
        try:
            _PROMPT_AUTOTUNE_LOCK.release()
        except Exception:
            pass


def _maybe_autotune_prompt_profile_async() -> None:
    if not _PROMPT_AUTOTUNE_ENABLED:
        return
    if time.time() - _LAST_PROMPT_AUTOTUNE_TS < _PROMPT_AUTOTUNE_INTERVAL_SEC:
        return
    thread = threading.Thread(
        target=_maybe_autotune_prompt_profile,
        name="brain-prompt-autotune",
        daemon=True,
    )
    thread.start()


def _load_persona_overrides() -> dict[str, Any]:
    if not _RAW_PERSONA_OVERRIDES:
        return {}
    try:
        data = json.loads(_RAW_PERSONA_OVERRIDES)
    except Exception:
        return {}
    if isinstance(data, dict):
        return {str(k).strip().lower(): v for k, v in data.items()}
    return {}


_PERSONA_OVERRIDES = _load_persona_overrides()


def _normalize_tag(tag: str) -> str:
    text = str(tag or "").strip().lower()
    if not text:
        return ""
    base = text.split("-", 1)[0].split("_", 1)[0]
    return base or text


def _select_persona_key(tag: str) -> str:
    text = str(tag or "").strip().lower()
    if not text:
        return "neutral"
    tokens = [t for t in text.replace("-", "_").split("_") if t]
    token_set = set(tokens)
    if "range" in token_set or "revert" in token_set or "reversion" in token_set or "fade" in token_set:
        return "range"
    if "bbrsi" in token_set or "bb" in token_set or "rsi" in token_set:
        return "range"
    if "vwap" in token_set or "levelreactor" in token_set or "magnet" in token_set:
        return "range"
    if "pullback" in token_set or "retest" in token_set:
        return "pullback"
    if "trend" in token_set or "donchian" in token_set or "breakout" in token_set:
        return "trend"
    if "momentum" in token_set or "impulse" in token_set or "burst" in token_set or "squeeze" in token_set:
        return "momentum"
    if "scalp" in token_set:
        return "scalp"
    return "neutral"


def _persona_text(strategy_tag: str, pocket: str) -> str:
    if not _PERSONA_ENABLED:
        return _PERSONA_DEFAULT
    if _PERSONA_MODE == "uniform":
        return _PERSONA_DEFAULT
    tag_key = _normalize_tag(strategy_tag)
    # Overrides can be string or dict with fields.
    override = _PERSONA_OVERRIDES.get(tag_key) or _PERSONA_OVERRIDES.get(strategy_tag.strip().lower())
    if override:
        if isinstance(override, str):
            return override
        if isinstance(override, dict):
            name = str(override.get("name") or "").strip()
            traits = str(override.get("traits") or "").strip()
            bias = str(override.get("bias") or "").strip()
            avoid = str(override.get("avoid") or "").strip()
            parts = [p for p in (name, traits, bias, avoid) if p]
            if parts:
                return " ".join(parts)
    key = _select_persona_key(strategy_tag)
    profile = _PERSONA_PRESETS.get(key, _PERSONA_PRESETS["neutral"])
    pocket_trait = _POCKET_TRAITS.get(pocket.strip().lower(), "")
    parts = [
        profile.get("name", ""),
        profile.get("traits", ""),
        profile.get("bias", ""),
        profile.get("avoid", ""),
        pocket_trait,
    ]
    return " ".join([p for p in parts if p])


def _build_prompt(context: dict[str, Any]) -> str:
    strategy_tag = str(context.get("strategy_tag") or "")
    pocket = str(context.get("pocket") or "")
    persona = _persona_text(strategy_tag, pocket)
    profile = _load_prompt_profile()
    profile_text = _format_prompt_profile(profile)
    ctx_text = _stringify(context, _MAX_CONTEXT_CHARS)
    profile_block = f"{profile_text}\n\n" if profile_text else ""
    return (
        "You are the decision brain for an automated USD/JPY trading worker. "
        "Decide whether to allow, reduce, or block a single entry candidate. "
        "Respond with JSON only.\n\n"
        f"Persona: {persona}\n\n"
        f"{profile_block}"
        "Rules:\n"
        "- action must be one of: ALLOW, REDUCE, BLOCK.\n"
        "- scale must be between 0.2 and 1.0 (never > 1.0). Use 1.0 for ALLOW.\n"
        "- Prefer blocking on uncertainty or missing context.\n"
        "- memory_update must be a short summary (<=200 chars) or empty string to keep memory.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "action": "ALLOW|REDUCE|BLOCK",\n'
        '  "scale": 0.2,\n'
        '  "reason": "short reason",\n'
        '  "memory_update": "short memory or empty"\n'
        "}\n\n"
        "Context:\n"
        f"{ctx_text}\n"
    )


def _parse_response(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # try to extract JSON object from text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


def _llm_failure_decision(*, memory: Optional[str], allow_reason: str) -> BrainDecision:
    if _FAIL_POLICY == "block":
        return BrainDecision(False, 0.0, "no_llm_block", "BLOCK", memory=memory)
    if _FAIL_POLICY == "reduce":
        fail_scale = min(0.5, max(_MIN_SCALE, 0.5))
        return BrainDecision(True, float(fail_scale), "no_llm_reduce", "REDUCE", memory=memory)
    return BrainDecision(True, 1.0, allow_reason, "ALLOW", memory=memory)


def decide(
    *,
    strategy_tag: Optional[str],
    pocket: Optional[str],
    side: str,
    units: int,
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None,
    entry_thesis: Optional[dict] = None,
    meta: Optional[dict] = None,
    confidence: Optional[int] = None,
    client_order_id: Optional[str] = None,
) -> BrainDecision:
    if not _should_use(strategy_tag, pocket):
        return BrainDecision(True, 1.0, "disabled", "ALLOW")
    tag = str(strategy_tag).strip()
    pocket_key = str(pocket).strip().lower()
    cache_key = (tag, pocket_key)
    profile_version = _profile_version(_load_prompt_profile())
    memory_before = _load_memory(tag, pocket_key)
    context = {
        "ts": _now_iso(),
        "strategy_tag": tag,
        "pocket": pocket_key,
        "side": side,
        "units": int(units),
        "sl_price": sl_price,
        "tp_price": tp_price,
        "confidence": confidence,
        "memory": memory_before or "",
        "entry_thesis": entry_thesis or {},
        "meta": meta or {},
    }
    now = time.monotonic()
    cached = _CACHE.get(cache_key)
    if cached and now - cached[0] <= _TTL_SEC:
        decision = cached[1]
        _log_decision_row(
            strategy_tag=tag,
            pocket=pocket_key,
            side=side,
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            confidence=confidence,
            client_order_id=client_order_id,
            backend=_BACKEND,
            source="cache",
            llm_ok=True,
            latency_ms=0.0,
            decision=decision,
            memory_before=memory_before,
            memory_after=decision.memory,
            profile_version=profile_version,
            context=context,
            payload={"cached": True},
            error=None,
        )
        try:
            _maybe_autotune_prompt_profile_async()
        except Exception:
            pass
        return decision

    prompt = _build_prompt(context)
    call_start = time.monotonic()
    vertex_resp = None
    payload: Optional[dict[str, Any]] = None
    llm_text_available = False
    if _BACKEND == "ollama":
        payload = call_ollama_chat_json(
            prompt,
            model=_OLLAMA_MODEL,
            url=_OLLAMA_URL,
            timeout_sec=_TIMEOUT_SEC,
            temperature=_TEMP,
            max_tokens=_MAX_TOKENS,
        )
        llm_text_available = bool(payload)
    else:
        vertex_resp = call_vertex_text(
            prompt,
            model=_VERTEX_MODEL,
            temperature=_TEMP,
            max_tokens=_MAX_TOKENS,
            timeout_sec=_TIMEOUT_SEC,
            response_mime_type="application/json",
        )
        llm_text_available = bool(vertex_resp and vertex_resp.text)
        if llm_text_available:
            payload = _parse_response(vertex_resp.text)
    call_ms = max(0.0, (time.monotonic() - call_start) * 1000.0)
    try:
        log_metric(
            "brain_latency_ms",
            call_ms,
            tags={
                "strategy": tag,
                "pocket": pocket_key,
                "backend": _BACKEND,
                "ok": bool(payload),
            },
        )
    except Exception:
        pass
    if payload is None:
        allow_reason = "bad_response" if llm_text_available else "no_llm"
        decision = _llm_failure_decision(memory=memory_before, allow_reason=allow_reason)
        try:
            _save_memory(tag, pocket_key, memory=decision.memory, decision=decision)
        except Exception:
            pass
        _CACHE[cache_key] = (now, decision)
        _log_decision_row(
            strategy_tag=tag,
            pocket=pocket_key,
            side=side,
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            confidence=confidence,
            client_order_id=client_order_id,
            backend=_BACKEND,
            source="llm_fail",
            llm_ok=False,
            latency_ms=call_ms,
            decision=decision,
            memory_before=memory_before,
            memory_after=decision.memory,
            profile_version=profile_version,
            context=context,
            payload=payload,
            error=allow_reason,
        )
        try:
            _maybe_autotune_prompt_profile_async()
        except Exception:
            pass
        return decision

    if vertex_resp is not None:
        try:
            log_metric(
                "brain_tokens_prompt",
                float(vertex_resp.prompt_tokens or 0),
                tags={"strategy": tag, "pocket": pocket_key, "backend": _BACKEND},
            )
            log_metric(
                "brain_tokens_output",
                float(vertex_resp.output_tokens or 0),
                tags={"strategy": tag, "pocket": pocket_key, "backend": _BACKEND},
            )
            log_metric(
                "brain_tokens_total",
                float(vertex_resp.total_tokens or 0),
                tags={"strategy": tag, "pocket": pocket_key, "backend": _BACKEND},
            )
            if _COST_INPUT_PER_1K > 0 or _COST_OUTPUT_PER_1K > 0:
                est_cost = (
                    (float(vertex_resp.prompt_tokens or 0) / 1000.0) * _COST_INPUT_PER_1K
                    + (float(vertex_resp.output_tokens or 0) / 1000.0) * _COST_OUTPUT_PER_1K
                )
                log_metric(
                    "brain_cost_est",
                    float(est_cost),
                    tags={"strategy": tag, "pocket": pocket_key, "backend": _BACKEND},
                )
        except Exception:
            pass

    action = str(payload.get("action") or "ALLOW").strip().upper()
    reason = str(payload.get("reason") or "").strip()
    if not reason:
        reason = "llm_decision"
    try:
        scale = float(payload.get("scale", 1.0))
    except Exception:
        scale = 1.0
    scale = max(_MIN_SCALE, min(scale, 1.0))

    if action not in {"ALLOW", "REDUCE", "BLOCK"}:
        action = "ALLOW"
    if action == "BLOCK":
        allowed = False
        scale = 0.0
    elif action == "REDUCE":
        allowed = True
    else:
        allowed = True
        if scale < 1.0:
            action = "REDUCE"

    memory_update = str(payload.get("memory_update") or "").strip()
    memory_after = memory_before
    if memory_update:
        if len(memory_update) > 200:
            memory_update = memory_update[:200] + "..."
        memory_after = memory_update
    elif memory_before:
        memory_after = memory_before

    decision = BrainDecision(allowed, scale, reason, action, memory=memory_after)
    if memory_update:
        _save_memory(tag, pocket_key, memory=memory_after, decision=decision)
    else:
        # still persist decision metadata for observability
        _save_memory(tag, pocket_key, memory=memory_after, decision=decision)

    _CACHE[cache_key] = (now, decision)
    _log_decision_row(
        strategy_tag=tag,
        pocket=pocket_key,
        side=side,
        units=units,
        sl_price=sl_price,
        tp_price=tp_price,
        confidence=confidence,
        client_order_id=client_order_id,
        backend=_BACKEND,
        source="llm",
        llm_ok=True,
        latency_ms=call_ms,
        decision=decision,
        memory_before=memory_before,
        memory_after=memory_after,
        profile_version=profile_version,
        context=context,
        payload=payload,
        error=None,
    )
    try:
        _maybe_autotune_prompt_profile_async()
    except Exception:
        pass
    return decision


__all__ = ["BrainDecision", "decide"]
