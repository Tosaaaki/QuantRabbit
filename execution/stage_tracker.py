"""
execution.stage_tracker
~~~~~~~~~~~~~~~~~~~~~~~
ステージ進行や再エントリーのクールダウンを SQLite で管理する補助モジュール。
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
import math

_DB_PATH = Path("logs/stage_state.db")
_TRADES_DB = Path("logs/trades.db")
_POCKET_SIZE_FLOOR = {
    "macro": float(os.getenv("SIZE_FLOOR_MACRO", "0.6")),
    "micro": float(os.getenv("SIZE_FLOOR_MICRO", "0.4")),
    "scalp": float(os.getenv("SIZE_FLOOR_SCALP", "0.35")),
}
_STAGE_REVERSE_JST_START = int(os.getenv("STAGE_REVERSE_JST_START", "10")) % 24
_STAGE_REVERSE_JST_END = int(os.getenv("STAGE_REVERSE_JST_END", "12")) % 24
_STAGE_REVERSE_LOSS_FLOOR = float(os.getenv("STAGE_REVERSE_LOSS_FLOOR", "0.3"))
_STAGE_REVERSE_FLIP_GUARD_SEC = int(os.getenv("STAGE_REVERSE_FLIP_GUARD_SEC", "90"))
# Loss cluster window default (minutes)
_LOSS_WINDOW_MINUTES = int(os.getenv("LOSS_CLUSTER_WINDOW_MIN", "480"))
# Loss judgment floor (JPY). Trades with |pl_jpy| <= this are treated as flat.
_MIN_LOSS_JPY = float(os.getenv("LOSS_CLUSTER_MIN_JPY", "50"))
# Clamp detection for scalp pocket (loss bundles)
_CLAMP_WINDOW_SEC = int(os.getenv("SCALP_CLAMP_WINDOW_SEC", "120"))
_CLAMP_WINDOW_COUNT = int(os.getenv("SCALP_CLAMP_WINDOW_COUNT", "5"))
_CLAMP_WINDOW_JPY = float(os.getenv("SCALP_CLAMP_WINDOW_JPY", "5000"))
_CLAMP_WINDOW_PIPS = float(os.getenv("SCALP_CLAMP_WINDOW_PIPS", "50"))
_CLAMP_DECAY_MIN = int(os.getenv("SCALP_CLAMP_DECAY_MIN", "120"))
_CLAMP_LEVEL_COOLDOWN = {
    1: int(os.getenv("SCALP_CLAMP_COOLDOWN_L1_SEC", "600")),
    2: int(os.getenv("SCALP_CLAMP_COOLDOWN_L2_SEC", "1800")),
    3: int(os.getenv("SCALP_CLAMP_COOLDOWN_L3_SEC", "3600")),
}
_CLAMP_IMPULSE_STOP = {
    1: 0,
    2: int(os.getenv("SCALP_CLAMP_IMPULSE_STOP_L2_SEC", "1800")),
    3: int(os.getenv("SCALP_CLAMP_IMPULSE_STOP_L3_SEC", "3600")),
}
_CLAMP_SCALP_FACTORS = {1: 0.6, 2: 0.4, 3: 0.1}
_CLAMP_IMPULSE_THIN_SCALE = float(os.getenv("SCALP_CLAMP_IMPULSE_THIN_SCALE", "0.2"))
AGGRESSIVE_TRADING = os.getenv("AGGRESSIVE_TRADING", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# trades.db 読み取り用 PRAGMA（多プロセス同時アクセス時のロック回避）
_TRADES_BUSY_TIMEOUT_MS = int(os.getenv("STAGE_TRADES_BUSY_TIMEOUT_MS", "10000"))
_TRADES_JOURNAL_MODE = os.getenv("STAGE_TRADES_JOURNAL_MODE", "WAL")
_TRADES_SYNCHRONOUS = os.getenv("STAGE_TRADES_SYNCHRONOUS", "NORMAL")
_TRADES_TEMP_STORE = os.getenv("STAGE_TRADES_TEMP_STORE", "MEMORY")
_TRADES_URI_TMPL = "file:{path}?mode=ro"
_STRATEGY_ALIAS_BASE = {
    "mlr": "MicroLevelReactor",
    "trendma": "TrendMA",
    "donchian": "Donchian55",
    "h1momentum": "H1Momentum",
    "bbrsi": "BB_RSI",
    "bb_rsi": "BB_RSI",
    "techfusion": "TechFusion",
    "macrotechfusion": "MacroTechFusion",
    "micropullbackfib": "MicroPullbackFib",
    "scalpreversalnwave": "ScalpReversalNWave",
    "rangecompressionbreak": "RangeCompressionBreak",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_utc(value: Optional[datetime]) -> datetime:
    if value is None:
        return _utcnow()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _base_strategy_tag(tag: Optional[str]) -> str:
    if not tag:
        return ""
    text = str(tag).strip()
    if not text:
        return ""
    base = text.split("-", 1)[0].strip() or text
    alias = _STRATEGY_ALIAS_BASE.get(base.lower())
    return alias or base


def _in_jst_reverse_window(now: Optional[datetime] = None) -> bool:
    current = _coerce_utc(now)
    jst = current + timedelta(hours=9)
    hour = jst.hour
    if _STAGE_REVERSE_JST_START <= _STAGE_REVERSE_JST_END:
        return _STAGE_REVERSE_JST_START <= hour < _STAGE_REVERSE_JST_END
    return hour >= _STAGE_REVERSE_JST_START or hour < _STAGE_REVERSE_JST_END


def _ensure_row_factory(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row


@dataclass(slots=True)
class CooldownInfo:
    pocket: str
    direction: str
    reason: str
    cooldown_until: datetime


class StageTracker:
    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or _DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self._path, timeout=5.0, check_same_thread=False)
        _ensure_row_factory(self._con)
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS stage_cooldown (
                pocket TEXT NOT NULL,
                direction TEXT NOT NULL,
                reason TEXT,
                cooldown_until TEXT NOT NULL,
                PRIMARY KEY (pocket, direction)
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS stage_state (
                pocket TEXT NOT NULL,
                direction TEXT NOT NULL,
                stage INTEGER DEFAULT 0,
                updated_at TEXT,
                PRIMARY KEY (pocket, direction)
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS stage_history (
                pocket TEXT NOT NULL,
                direction TEXT NOT NULL,
                last_trade_id INTEGER DEFAULT 0,
                lose_streak INTEGER DEFAULT 0,
                win_streak INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (pocket, direction)
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_cooldown (
                strategy TEXT PRIMARY KEY,
                reason TEXT,
                cooldown_until TEXT NOT NULL
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_history (
                strategy TEXT PRIMARY KEY,
                last_trade_id INTEGER DEFAULT 0,
                lose_streak INTEGER DEFAULT 0,
                win_streak INTEGER DEFAULT 0,
                updated_at TEXT
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_reentry_state (
                strategy TEXT NOT NULL,
                direction TEXT NOT NULL,
                last_trade_id INTEGER DEFAULT 0,
                last_close_time TEXT,
                last_close_price REAL,
                last_result TEXT,
                last_pl_pips REAL,
                updated_at TEXT,
                PRIMARY KEY (strategy, direction)
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS clamp_guard_state (
                id INTEGER PRIMARY KEY CHECK (id=1),
                level INTEGER DEFAULT 0,
                event_count INTEGER DEFAULT 0,
                clamp_score REAL DEFAULT 0,
                last_trade_id INTEGER DEFAULT 0,
                last_event_at TEXT,
                clamp_until TEXT,
                impulse_stop_until TEXT
            )
            """
        )
        try:
            self._con.execute("ALTER TABLE clamp_guard_state ADD COLUMN clamp_score REAL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS hold_violation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                pocket TEXT NOT NULL,
                direction TEXT NOT NULL,
                required_sec REAL,
                actual_sec REAL,
                reason TEXT
            )
            """
        )
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS pocket_loss_window (
                pocket TEXT NOT NULL,
                trade_id INTEGER NOT NULL,
                closed_at TEXT NOT NULL,
                loss_jpy REAL NOT NULL,
                loss_pips REAL NOT NULL,
                PRIMARY KEY (pocket, trade_id)
            )
            """
        )
        self._con.commit()
        self._loss_window_minutes = int(
            os.getenv("LOSS_CLUSTER_WINDOW_MIN", str(_LOSS_WINDOW_MINUTES)) or _LOSS_WINDOW_MINUTES
        )
        self._loss_decay_minutes = int(os.getenv("LOSS_STREAK_DECAY_MINUTES", "120") or 120)
        self._cooldown_disabled = (
            AGGRESSIVE_TRADING
            or str(os.getenv("DISABLE_ALL_COOLDOWNS", "")).strip().lower() in {"1", "true", "yes"}
        )
        self._cluster_cooldown_disabled = (
            AGGRESSIVE_TRADING
            or str(os.getenv("DISABLE_CLUSTER_COOLDOWN", "")).strip().lower() in {"1", "true", "yes"}
        )
        # デフォルトで scalp の損失クラスタークールダウンを無効化（リクエスト対応）
        self._scalp_cluster_cooldown_disabled = (
            AGGRESSIVE_TRADING
            or str(os.getenv("DISABLE_SCALP_CLUSTER_COOLDOWN", "0")).strip().lower()
            in {"1", "true", "yes"}
        )
        self._cluster_last_trade_id: Dict[str, int] = {}
        for row in self._con.execute(
            "SELECT pocket, MAX(trade_id) AS last_id FROM pocket_loss_window GROUP BY pocket"
        ):
            pocket = row["pocket"]
            if pocket:
                self._cluster_last_trade_id[pocket] = int(row["last_id"] or 0)
        self._recent_profile: Dict[str, Dict[str, float]] = {}
        self._weight_hint: Dict[str, float] = {}
        self._clamp_state = self._load_clamp_state()
        # ts, trade_id, pl_jpy, pl_pips, direction
        self._clamp_recent: Deque[Tuple[datetime, int, float, float, str]] = deque()
        # 再起動後に過去のクラスターを再検知しないよう、前回イベントの trade_id を初期値にする
        try:
            self._clamp_last_scanned = int(self._clamp_state.get("last_trade_id", 0) or 0)
        except Exception:
            self._clamp_last_scanned = 0

    def _load_clamp_state(self) -> Dict[str, object]:
        row = self._con.execute(
            "SELECT level, event_count, clamp_score, last_trade_id, last_event_at, clamp_until, impulse_stop_until FROM clamp_guard_state WHERE id=1"
        ).fetchone()
        state = {
            "level": 0,
            "event_count": 0,
            "clamp_score": 0.0,
            "last_trade_id": 0,
            "last_event_at": None,
            "clamp_until": None,
            "impulse_stop_until": None,
        }
        if not row:
            self._con.execute(
                """
                INSERT OR IGNORE INTO clamp_guard_state(id, level, event_count, clamp_score, last_trade_id, last_event_at, clamp_until, impulse_stop_until)
                VALUES (1, 0, 0, 0.0, 0, NULL, NULL, NULL)
                """
            )
            self._con.commit()
            return state
        try:
            state["level"] = int(row["level"] or 0)
            state["event_count"] = int(row["event_count"] or 0)
            state["clamp_score"] = float(row["clamp_score"] or 0.0)
            state["last_trade_id"] = int(row["last_trade_id"] or 0)
        except Exception:
            state["level"] = 0
            state["event_count"] = 0
            state["clamp_score"] = 0.0
            state["last_trade_id"] = 0
        # Safety cap to avoid runaway scores on reload
        try:
            if state["clamp_score"] > 3.0:
                state["clamp_score"] = 3.0
        except Exception:
            state["clamp_score"] = 0.0
        for key in ("last_event_at", "clamp_until", "impulse_stop_until"):
            raw = row[key] if row else None
            try:
                state[key] = _parse_timestamp(raw) if raw else None
            except Exception:
                state[key] = None
        return state

    def _persist_clamp_state(self) -> None:
        st = self._clamp_state
        self._con.execute(
            """
            INSERT INTO clamp_guard_state(id, level, event_count, clamp_score, last_trade_id, last_event_at, clamp_until, impulse_stop_until)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                level=excluded.level,
                event_count=excluded.event_count,
                clamp_score=excluded.clamp_score,
                last_trade_id=excluded.last_trade_id,
                last_event_at=excluded.last_event_at,
                clamp_until=excluded.clamp_until,
                impulse_stop_until=excluded.impulse_stop_until
            """,
            (
                int(st.get("level", 0) or 0),
                int(st.get("event_count", 0) or 0),
                float(st.get("clamp_score", 0.0) or 0.0),
                int(st.get("last_trade_id", 0) or 0),
                st.get("last_event_at").isoformat() if st.get("last_event_at") else None,
                st.get("clamp_until").isoformat() if st.get("clamp_until") else None,
                st.get("impulse_stop_until").isoformat()
                if st.get("impulse_stop_until")
                else None,
            ),
        )
        self._con.commit()

    def _decay_clamp_guard(self, now: datetime) -> None:
        level = int(self._clamp_state.get("level", 0) or 0)
        score = float(self._clamp_state.get("clamp_score", 0.0) or 0.0)
        last_evt = self._clamp_state.get("last_event_at")
        if (level <= 0 and score <= 0.0) or not last_evt:
            return
        try:
            elapsed = max(0.0, (now - last_evt).total_seconds())
        except Exception:
            return
        # Exponential decay with half-life = _CLAMP_DECAY_MIN minutes
        half_life_sec = max(1.0, float(_CLAMP_DECAY_MIN) * 60.0)
        decay_factor = 0.5 ** (elapsed / half_life_sec)
        decayed_score = max(0.0, min(3.0, score * decay_factor))
        self._clamp_state["clamp_score"] = decayed_score
        if decayed_score <= 0.5:
            new_level = 0
        elif decayed_score >= 3.0:
            new_level = 3
        elif decayed_score >= 2.0:
            new_level = 2
        else:
            new_level = 1
        self._clamp_state["level"] = new_level
        if new_level == 0:
            self._clamp_state["clamp_until"] = None
            self._clamp_state["impulse_stop_until"] = None
        self._persist_clamp_state()

    def _bump_clamp_level(
        self,
        event_time: datetime,
        trade_id: int,
        *,
        loss_jpy: float,
        loss_pips: float,
        count: int,
        severity: int = 1,
        cooldown_scale: float = 1.0,
    ) -> int:
        severity = max(1, severity)
        current_level = int(self._clamp_state.get("level", 0) or 0)
        current_score = float(self._clamp_state.get("clamp_score", 0.0) or 0.0)
        new_score = min(3.0, current_score + float(severity))
        self._clamp_state["clamp_score"] = new_score
        if new_score >= 3.0:
            new_level = 3
        elif new_score >= 2.0:
            new_level = 2
        else:
            new_level = max(1, 1 if current_level == 0 else current_level + 1)
        self._clamp_state["level"] = new_level
        self._clamp_state["event_count"] = int(self._clamp_state.get("event_count", 0) or 0) + 1
        self._clamp_state["last_trade_id"] = max(int(trade_id), int(self._clamp_state.get("last_trade_id", 0) or 0))
        self._clamp_state["last_event_at"] = event_time
        base_cd = _CLAMP_LEVEL_COOLDOWN.get(new_level, _CLAMP_LEVEL_COOLDOWN[1])
        scaled_cd = int(max(1, base_cd * cooldown_scale))
        clamp_until = event_time + timedelta(seconds=scaled_cd)
        prev_clamp = self._clamp_state.get("clamp_until")
        if prev_clamp:
            clamp_until = max(prev_clamp, clamp_until)
        self._clamp_state["clamp_until"] = clamp_until
        stop_dur = _CLAMP_IMPULSE_STOP.get(new_level, 0)
        if stop_dur > 0:
            scaled_stop = int(max(1, stop_dur * cooldown_scale))
            stop_until = event_time + timedelta(seconds=scaled_stop)
            prev_stop = self._clamp_state.get("impulse_stop_until")
            if not prev_stop or stop_until > prev_stop:
                self._clamp_state["impulse_stop_until"] = stop_until
        self._persist_clamp_state()
        logging.info(
            "[CLAMP] detected scalp loss bundle level=%s score=%.2f severity=%s count=%s loss_jpy=%.1f loss_pips=%.1f window=%ss cd=%ss stop=%ss",
            new_level,
            new_score,
            severity,
            count,
            loss_jpy,
            loss_pips,
            _CLAMP_WINDOW_SEC,
            scaled_cd,
            stop_dur,
        )
        return new_level

    def _detect_clamp_events(
        self,
        rows: List[sqlite3.Row],
        now_dt: datetime,
        *,
        clamp_jpy: float,
        clamp_pips: float,
        n_min: int,
        nav_val: float = 0.0,
        cooldown_scale: float = 1.0,
    ) -> None:
        self._decay_clamp_guard(now_dt)
        if not rows:
            return
        window = self._clamp_recent
        last_scanned = int(self._clamp_last_scanned or 0)
        max_seen = last_scanned
        for row in rows:
            try:
                trade_id = int(row["id"])
            except Exception:
                continue
            if trade_id <= last_scanned:
                continue
            pocket = row["pocket"] or ""
            if pocket != "scalp":
                continue
            try:
                close_reason = str(row["close_reason"] or "")
            except Exception:
                close_reason = ""
            if close_reason != "MARKET_ORDER_TRADE_CLOSE":
                continue
            try:
                pl_jpy = float(row["realized_pl"] or 0.0)
                pl_pips = float(row["pl_pips"] or 0.0)
            except Exception:
                continue
            if pl_jpy >= -_MIN_LOSS_JPY:
                continue
            try:
                units_val = int(row["units"] or 0)
            except Exception:
                units_val = 0
            direction = "long" if units_val > 0 else "short" if units_val < 0 else ""
            ts_raw = row["close_time"]
            try:
                ts = _parse_timestamp(ts_raw) if ts_raw else now_dt
            except Exception:
                ts = now_dt
            window.append((ts, trade_id, pl_jpy, pl_pips, direction))
            while window and (ts - window[0][0]).total_seconds() > _CLAMP_WINDOW_SEC:
                window.popleft()
            max_seen = max(max_seen, trade_id)
            if not window:
                continue
            count = len(window)
            loss_sum_jpy = sum(item[2] for item in window)
            loss_sum_pips = sum(item[3] for item in window)
            if (
                count >= n_min
                and (loss_sum_jpy <= -clamp_jpy or loss_sum_pips <= -clamp_pips)
            ):
                severity = 1
                dir_counts: Dict[str, int] = {"long": 0, "short": 0}
                for item in window:
                    if len(item) >= 5 and item[4]:
                        dir_counts[item[4]] = dir_counts.get(item[4], 0) + 1
                main_dir = None
                if dir_counts["long"] > dir_counts["short"]:
                    main_dir = "long"
                elif dir_counts["short"] > dir_counts["long"]:
                    main_dir = "short"
                nav_thresh = 0.0
                try:
                    nav_thresh = float(nav_val or 0.0) * 0.005
                except Exception:
                    nav_thresh = 0.0
                if nav_thresh > 0 and abs(loss_sum_jpy) >= nav_thresh:
                    severity = 2
                new_level = self._bump_clamp_level(
                    ts,
                    trade_id,
                    loss_jpy=loss_sum_jpy,
                    loss_pips=loss_sum_pips,
                    count=count,
                    severity=severity,
                    cooldown_scale=cooldown_scale,
                )
                # Direction-aware re-entryガード: 同方向のみ一定時間ブロック
                if main_dir:
                    cd = _CLAMP_LEVEL_COOLDOWN.get(new_level, _CLAMP_LEVEL_COOLDOWN[1])
                    for pocket_name in ("scalp", "micro", "macro"):
                        try:
                            self.ensure_cooldown(
                                pocket_name,
                                main_dir,
                                reason=f"clamp_bundle_L{new_level}",
                                seconds=cd,
                                now=ts,
                            )
                        except Exception:
                            logging.exception("[CLAMP] failed to set cooldown pocket=%s dir=%s", pocket_name, main_dir)
                    logging.info(
                        "[CLAMP] cooldown dir=%s level=%s sec=%s pockets=scalp|micro|macro",
                        main_dir,
                        new_level,
                        cd,
                    )
                window.clear()
                last_scanned = trade_id
                max_seen = max(max_seen, trade_id)
        if max_seen > self._clamp_last_scanned:
            self._clamp_last_scanned = max_seen
        # Keep only recent trades in memory to allow cross-iteration detection
        self._clamp_recent = deque(
            [
                item
                for item in window
                if (now_dt - item[0]).total_seconds() <= _CLAMP_WINDOW_SEC
            ]
        )

    def get_clamp_state(self, now: Optional[datetime] = None) -> Dict[str, object]:
        current = _coerce_utc(now)
        self._decay_clamp_guard(current)
        state = dict(self._clamp_state)
        clamp_until = state.get("clamp_until")
        active_level = 0
        if clamp_until and clamp_until > current:
            active_level = int(state.get("level", 0) or 0)
        scalp_scale = 1.0
        if active_level > 0:
            scalp_scale = _CLAMP_SCALP_FACTORS.get(active_level, 1.0)
        impulse_stop_until = state.get("impulse_stop_until")
        impulse_thin_active = False
        if state.get("level", 0) >= 2:
            if impulse_stop_until and current >= impulse_stop_until:
                last_evt = state.get("last_event_at") or current
                thin_until = last_evt + timedelta(minutes=_CLAMP_DECAY_MIN)
                impulse_thin_active = current < thin_until
        return {
            "level": int(state.get("level", 0) or 0),
            "event_count": int(state.get("event_count", 0) or 0),
            "clamp_score": float(state.get("clamp_score", 0.0) or 0.0),
            "last_event_at": state.get("last_event_at"),
            "clamp_until": clamp_until,
            "scalp_conf_scale": float(scalp_scale),
            "impulse_stop_until": impulse_stop_until,
            "impulse_thin_active": impulse_thin_active,
            "impulse_thin_scale": _CLAMP_IMPULSE_THIN_SCALE if impulse_thin_active else 1.0,
        }

    def close(self) -> None:
        self._con.close()

    def clear_expired(self, now: datetime) -> None:
        if self._cooldown_disabled:
            return
        now = _coerce_utc(now)
        ts = now.isoformat()
        self._con.execute(
            "DELETE FROM stage_cooldown WHERE cooldown_until <= ?", (ts,)
        )
        self._con.execute(
            "DELETE FROM strategy_cooldown WHERE cooldown_until <= ?",
            (ts,),
        )
        self._con.commit()

    def set_stage(
        self,
        pocket: str,
        direction: str,
        stage: int,
        *,
        now: Optional[datetime] = None,
    ) -> None:
        ts = _coerce_utc(now).isoformat()
        self._con.execute(
            """
            INSERT INTO stage_state(pocket, direction, stage, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(pocket, direction)
            DO UPDATE SET stage=excluded.stage,
                          updated_at=excluded.updated_at
            """,
            (pocket, direction, max(0, stage), ts),
        )
        self._con.commit()

    def reset_stage(self, pocket: str, direction: str, *, now: Optional[datetime] = None) -> None:
        self.set_stage(pocket, direction, 0, now=now)

    def get_stage(self, pocket: str, direction: str) -> int:
        row = self._con.execute(
            "SELECT stage FROM stage_state WHERE pocket=? AND direction=?", (pocket, direction)
        ).fetchone()
        if not row:
            return 0
        return int(row[0] or 0)

    def set_cooldown(
        self,
        pocket: str,
        direction: str,
        *,
        reason: str,
        seconds: int,
        now: Optional[datetime] = None,
    ) -> None:
        if self._cooldown_disabled:
            return
        base = _coerce_utc(now)
        cooldown_until = base + timedelta(seconds=max(1, seconds))
        self._con.execute(
            """
            INSERT INTO stage_cooldown(pocket, direction, reason, cooldown_until)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(pocket, direction)
            DO UPDATE SET reason=excluded.reason,
                          cooldown_until=excluded.cooldown_until
            """,
            (pocket, direction, reason, cooldown_until.isoformat()),
        )
        self._con.commit()

    def ensure_cooldown(
        self,
        pocket: str,
        direction: str,
        *,
        reason: str,
        seconds: int,
        now: Optional[datetime] = None,
    ) -> bool:
        """
        Ensure the cooldown for (pocket, direction) is at least the requested duration.
        Returns True if the cooldown was extended or set, False if existing cooldown was longer.
        """
        if self._cooldown_disabled:
            return False
        current = _coerce_utc(now)
        desired_until = current + timedelta(seconds=max(1, seconds))
        info = self.get_cooldown(pocket, direction, now=current)
        if info and info.cooldown_until >= desired_until:
            return False
        self.set_cooldown(
            pocket,
            direction,
            reason=reason,
            seconds=seconds,
            now=current,
        )
        return True

    def clear_cooldown(self, pocket: str, direction: str) -> bool:
        cur = self._con.execute(
            "DELETE FROM stage_cooldown WHERE pocket=? AND direction=?",
            (pocket, direction),
        )
        self._con.commit()
        return cur.rowcount > 0

    def expire_stages_if_flat(
        self,
        positions: Dict[str, Dict],
        *,
        now: Optional[datetime] = None,
        grace_seconds: int = 180,
    ) -> int:
        """
        Reset stale stage states when the direction has been flat for a while.
        Returns the number of resets performed.
        """
        now = _coerce_utc(now)
        threshold = now - timedelta(seconds=max(1, grace_seconds))
        resets = 0
        for pocket, direction, stage, updated_at in self._con.execute(
            "SELECT pocket, direction, stage, updated_at FROM stage_state WHERE stage>0"
        ):
            info = positions.get(pocket) or {}
            units = int(info.get("long_units" if direction == "long" else "short_units", 0) or 0)
            if units != 0:
                continue
            try:
                updated_dt = _parse_timestamp(updated_at) if updated_at else None
            except Exception:
                updated_dt = None
            if updated_dt and updated_dt > threshold:
                continue
            self.reset_stage(pocket, direction, now=now)
            resets += 1
        return resets

    def log_hold_violation(
        self,
        pocket: str,
        direction: str,
        *,
        required_sec: float,
        actual_sec: float,
        reason: str = "hold_violation",
        cooldown_seconds: int = 90,
        now: Optional[datetime] = None,
    ) -> None:
        ts = _coerce_utc(now).isoformat()
        self._con.execute(
            """
            INSERT INTO hold_violation_log(ts, pocket, direction, required_sec, actual_sec, reason)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ts, pocket, direction, float(required_sec), float(actual_sec), reason),
        )
        self._con.commit()
        if cooldown_seconds > 0 and not self._cooldown_disabled:
            current = _coerce_utc(now)
            self.ensure_cooldown(
                pocket,
                direction,
                reason=reason,
                seconds=cooldown_seconds,
                now=current,
            )

    def set_strategy_cooldown(
        self,
        strategy: str,
        *,
        reason: str,
        seconds: int,
        now: Optional[datetime] = None,
    ) -> None:
        if self._cooldown_disabled:
            return
        if not strategy:
            return
        base = _coerce_utc(now)
        cooldown_until = base + timedelta(seconds=max(1, seconds))
        self._con.execute(
            """
            INSERT INTO strategy_cooldown(strategy, reason, cooldown_until)
            VALUES (?, ?, ?)
            ON CONFLICT(strategy)
            DO UPDATE SET reason=excluded.reason,
                          cooldown_until=excluded.cooldown_until
            """,
            (strategy, reason, cooldown_until.isoformat()),
        )
        self._con.commit()

    def is_strategy_blocked(
        self, strategy: str, now: Optional[datetime] = None
    ) -> Tuple[bool, Optional[int], Optional[str]]:
        if self._cooldown_disabled:
            return False, None, None
        if not strategy:
            return False, None, None
        row = self._con.execute(
            "SELECT reason, cooldown_until FROM strategy_cooldown WHERE strategy=?",
            (strategy,),
        ).fetchone()
        if not row:
            return False, None, None
        limit = _parse_timestamp(row[1])
        current = _coerce_utc(now)
        if current >= limit:
            self._con.execute(
                "DELETE FROM strategy_cooldown WHERE strategy=?",
                (strategy,),
            )
            self._con.commit()
            return False, None, None
        remaining = int((limit - current).total_seconds())
        return True, max(1, remaining), row[0] or ""

    def get_cooldown(
        self, pocket: str, direction: str, now: Optional[datetime] = None
    ) -> Optional[CooldownInfo]:
        if self._cooldown_disabled:
            return None
        row = self._con.execute(
            "SELECT reason, cooldown_until FROM stage_cooldown WHERE pocket=? AND direction=?",
            (pocket, direction),
        ).fetchone()
        if not row:
            return None
        limit = _parse_timestamp(row[1])
        current = _coerce_utc(now)
        if current >= limit:
            self._con.execute(
                "DELETE FROM stage_cooldown WHERE pocket=? AND direction=?",
                (pocket, direction),
            )
            self._con.commit()
            return None
        # For historical compatibility, expose cooldown_until as a naive UTC datetime.
        # Internally we normalize comparisons to UTC-aware datetimes via _coerce_utc/_parse_timestamp.
        limit_out = limit.replace(tzinfo=None)
        return CooldownInfo(
            pocket=pocket,
            direction=direction,
            reason=row["reason"] or "",
            cooldown_until=limit_out,
        )

    def is_blocked(
        self, pocket: str, direction: str, now: Optional[datetime] = None
    ) -> Tuple[bool, Optional[int], Optional[str]]:
        if self._cooldown_disabled:
            return False, None, None
        info = self.get_cooldown(pocket, direction, now)
        if not info:
            return False, None, None
        current = _coerce_utc(now)
        remaining = int((info.cooldown_until - current).total_seconds())
        return True, max(1, remaining), info.reason

    def update_loss_streaks(
        self,
        *,
        trades_db: Path | None = None,
        now: Optional[datetime] = None,
        cooldown_seconds: int = 900,
        cooldown_map: Optional[Dict[str, int]] = None,
        strategy_loss_threshold: int = 3,
        range_active: bool = False,
        atr_pips: Optional[float] = None,
        vol_5m: Optional[float] = None,
        adx_m1: Optional[float] = None,
        momentum: Optional[float] = None,
        nav: Optional[float] = None,
        open_scalp_positions: int = 0,
        atr_m5_pips: Optional[float] = None,
    ) -> None:
        trades_path = trades_db or _TRADES_DB
        if not trades_path.exists():
            return
        # trades.db は読み取り専用接続でロックを避ける
        uri = _TRADES_URI_TMPL.format(path=trades_path)
        conn = sqlite3.connect(
            uri,
            uri=True,
            timeout=_TRADES_BUSY_TIMEOUT_MS / 1000.0,
            isolation_level=None,
        )
        try:
            conn.execute(f"PRAGMA journal_mode={_TRADES_JOURNAL_MODE}")
        except sqlite3.Error:
            pass
        try:
            conn.execute(f"PRAGMA busy_timeout={_TRADES_BUSY_TIMEOUT_MS}")
        except sqlite3.Error:
            pass
        try:
            conn.execute(f"PRAGMA synchronous={_TRADES_SYNCHRONOUS}")
        except sqlite3.Error:
            pass
        try:
            conn.execute(f"PRAGMA temp_store={_TRADES_TEMP_STORE}")
        except sqlite3.Error:
            pass
        try:
            conn.execute("PRAGMA query_only=ON")
        except sqlite3.Error:
            pass
        _ensure_row_factory(conn)
        rows = conn.execute(
            "SELECT id, pocket, units, pl_pips, realized_pl, strategy_tag, close_time, close_price, close_reason FROM trades WHERE close_time IS NOT NULL ORDER BY id ASC"
        ).fetchall()
        conn.close()
        if not rows:
            return
        rows = list(rows)

        now_dt = _coerce_utc(now)
        self._decay_loss_streaks(now_dt)
        clamp_jpy = _CLAMP_WINDOW_JPY
        clamp_pips = _CLAMP_WINDOW_PIPS
        cooldown_scale = 1.0
        try:
            def _clampf(val: float, lo: float, hi: float) -> float:
                return max(lo, min(hi, val))

            if vol_5m is not None:
                cooldown_scale *= max(0.7, min(1.3, float(vol_5m) / 1.0))
            if range_active:
                cooldown_scale *= 0.7
            if adx_m1 is not None and momentum is not None:
                if adx_m1 >= 30.0 and abs(momentum) > 0.0:
                    cooldown_scale *= 1.2
            cooldown_scale = max(0.5, min(1.5, cooldown_scale))
            vol_norm = _clampf(float(vol_5m or 1.0) / 1.2, 0.0, 2.0)
            atr_norm = _clampf(float(atr_pips or 2.0) / 2.0, 0.0, 2.0)
            scale = _clampf(0.6 + 0.25 * vol_norm + 0.25 * atr_norm, 0.6, 1.6)
            clamp_jpy = _CLAMP_WINDOW_JPY * scale
            clamp_pips = _CLAMP_WINDOW_PIPS * scale
        except Exception:
            clamp_jpy = _CLAMP_WINDOW_JPY
            clamp_pips = _CLAMP_WINDOW_PIPS
            cooldown_scale = 1.0
        # Scale loss thresholds by NAV and current exposure context
        nav_val = float(nav or 0.0)
        nav_pct_main = 0.0025  # 0.25% NAV
        nav_pct_floor = 0.0005  # 0.05% NAV
        nav_jpy = nav_val * nav_pct_main if nav_val > 0 else 0.0
        nav_floor_jpy = nav_val * nav_pct_floor if nav_val > 0 else 0.0
        clamp_jpy = max(clamp_jpy, nav_jpy, nav_floor_jpy)
        # pips threshold: ATR(M5) weighted to reduce noise on high vol
        atr_m5_val = float(atr_m5_pips or 0.0)
        if atr_m5_val > 0:
            clamp_pips = max(clamp_pips, atr_m5_val * 8.0)
        clamp_pips = max(clamp_pips, _CLAMP_WINDOW_PIPS)
        # Count threshold scales with current open scalp positions
        n_min = max(3, int(math.ceil(max(0, open_scalp_positions) * 0.25)))
        n_min = max(n_min, 3)
        self._detect_clamp_events(
            rows,
            now_dt,
            clamp_jpy=clamp_jpy,
            clamp_pips=clamp_pips,
            n_min=n_min,
            nav_val=nav_val,
            cooldown_scale=cooldown_scale,
        )

        existing: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
        for row in self._con.execute(
            "SELECT pocket, direction, last_trade_id, lose_streak, win_streak FROM stage_history"
        ):
            existing[(row[0], row[1])] = (
                int(row[2] or 0),
                int(row[3] or 0),
                int(row[4] or 0),
            )

        strategy_existing: Dict[str, Tuple[int, int, int]] = {}
        for row in self._con.execute(
            "SELECT strategy, last_trade_id, lose_streak, win_streak FROM strategy_history"
        ):
            strategy_existing[row[0]] = (
                int(row[1] or 0),
                int(row[2] or 0),
                int(row[3] or 0),
            )
        reentry_existing: Dict[Tuple[str, str], Tuple[int, Optional[datetime]]] = {}
        for row in self._con.execute(
            "SELECT strategy, direction, last_trade_id, last_close_time FROM strategy_reentry_state"
        ):
            key = (row["strategy"], row["direction"])
            last_close_dt: Optional[datetime] = None
            raw_last_close = row["last_close_time"]
            if raw_last_close:
                try:
                    last_close_dt = _parse_timestamp(str(raw_last_close))
                except Exception:
                    last_close_dt = None
            reentry_existing[key] = (int(row["last_trade_id"] or 0), last_close_dt)

        ts = now_dt.isoformat()
        reverse_window = _in_jst_reverse_window(now_dt)
        new_losses: List[Tuple[str, int, datetime, float, float]] = []
        for row in rows:
            pocket = row["pocket"] or ""
            units = int(row["units"] or 0)
            if not pocket or units == 0:
                continue
            direction = "long" if units > 0 else "short"
            key = (pocket, direction)
            last_id, lose_streak, win_streak = existing.get(key, (0, 0, 0))
            try:
                trade_id = int(row["id"])
            except (TypeError, ValueError):
                continue
            if trade_id <= last_id:
                continue
            pl_jpy = float(row["realized_pl"] or 0.0)
            if pl_jpy < -_MIN_LOSS_JPY:
                lose_streak += 1
                win_streak = 0
                try:
                    pl_pips = float(row["pl_pips"] or 0.0)
                except Exception:
                    pl_pips = 0.0
                new_losses.append((pocket, trade_id, now_dt, abs(pl_jpy), abs(pl_pips)))
            elif pl_jpy > _MIN_LOSS_JPY:
                win_streak += 1
                lose_streak = 0
            else:
                lose_streak = 0
                win_streak = 0

            self._con.execute(
                """
                INSERT INTO stage_history(pocket, direction, last_trade_id, lose_streak, win_streak, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(pocket, direction)
                DO UPDATE SET last_trade_id=excluded.last_trade_id,
                              lose_streak=excluded.lose_streak,
                              win_streak=excluded.win_streak,
                              updated_at=excluded.updated_at
                """,
                (pocket, direction, trade_id, lose_streak, win_streak, ts),
            )
            existing[key] = (trade_id, lose_streak, win_streak)
            if lose_streak >= 3:
                pocket_cooldown = (cooldown_map or {}).get(pocket, cooldown_seconds)
                if pocket != "scalp":
                    self.set_cooldown(
                        pocket,
                        direction,
                        reason="loss_streak",
                        seconds=pocket_cooldown,
                        now=now_dt,
                    )
                    opp_dir = "short" if direction == "long" else "long"
                    if pocket == "micro":
                        opp_seconds = max(90, pocket_cooldown // 3)
                    else:
                        opp_seconds = max(240, pocket_cooldown // 2)
                    if pocket == "macro" and reverse_window:
                        reduced = max(
                            _STAGE_REVERSE_FLIP_GUARD_SEC,
                            int(opp_seconds * 0.4),
                        )
                        opp_seconds = reduced
                    self.set_cooldown(
                        pocket,
                        opp_dir,
                        reason="flip_guard",
                        seconds=opp_seconds,
                        now=now_dt,
                    )

            strategy_tag = row["strategy_tag"] if "strategy_tag" in row.keys() else None
            if strategy_tag:
                base_strategy = _base_strategy_tag(strategy_tag)
                strat_last, strat_lose, strat_win = strategy_existing.get(strategy_tag, (0, 0, 0))
                if trade_id > strat_last:
                    if pl_jpy < -1.0:
                        strat_lose += 1
                        strat_win = 0
                    elif pl_jpy > 1.0:
                        strat_win += 1
                        strat_lose = 0
                    else:
                        strat_win = 0
                        strat_lose = 0

                    self._con.execute(
                        """
                        INSERT INTO strategy_history(strategy, last_trade_id, lose_streak, win_streak, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(strategy)
                        DO UPDATE SET last_trade_id=excluded.last_trade_id,
                                      lose_streak=excluded.lose_streak,
                                      win_streak=excluded.win_streak,
                                      updated_at=excluded.updated_at
                        """,
                        (strategy_tag, trade_id, strat_lose, strat_win, ts),
                    )
                    strategy_existing[strategy_tag] = (trade_id, strat_lose, strat_win)
                    if strat_lose >= strategy_loss_threshold:
                        strat_seconds = (cooldown_map or {}).get(pocket, cooldown_seconds)
                        self.set_strategy_cooldown(
                            strategy_tag,
                            reason="loss_streak",
                            seconds=strat_seconds,
                            now=now_dt,
                        )
                        strat_lose = 0
                        self._con.execute(
                            "UPDATE strategy_history SET lose_streak=? WHERE strategy=?",
                            (strat_lose, strategy_tag),
                        )
                if base_strategy:
                    reentry_key = (base_strategy, direction)
                    last_reentry_id, last_reentry_close = reentry_existing.get(reentry_key, (0, None))
                    close_time = row["close_time"] if "close_time" in row.keys() else None
                    close_time_dt: Optional[datetime] = None
                    if close_time:
                        try:
                            close_time_dt = _parse_timestamp(str(close_time))
                        except Exception:
                            close_time_dt = None
                    should_update_reentry = trade_id > last_reentry_id
                    if (
                        not should_update_reentry
                        and close_time_dt is not None
                        and last_reentry_close is not None
                        and close_time_dt > (last_reentry_close + timedelta(seconds=1))
                    ):
                        # trades.db row ids can regress after restore/compaction; fall back to close_time recency.
                        should_update_reentry = True
                    if should_update_reentry:
                        try:
                            close_price = (
                                float(row["close_price"])
                                if row["close_price"] is not None
                                else None
                            )
                        except Exception:
                            close_price = None
                        try:
                            pl_pips = float(row["pl_pips"] or 0.0)
                        except Exception:
                            pl_pips = 0.0
                        if pl_pips > 0:
                            result = "win"
                        elif pl_pips < 0:
                            result = "loss"
                        else:
                            result = "flat"
                        self._con.execute(
                            """
                            INSERT INTO strategy_reentry_state(
                                strategy,
                                direction,
                                last_trade_id,
                                last_close_time,
                                last_close_price,
                                last_result,
                                last_pl_pips,
                                updated_at
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(strategy, direction)
                            DO UPDATE SET last_trade_id=excluded.last_trade_id,
                                          last_close_time=excluded.last_close_time,
                                          last_close_price=excluded.last_close_price,
                                          last_result=excluded.last_result,
                                          last_pl_pips=excluded.last_pl_pips,
                                          updated_at=excluded.updated_at
                            """,
                            (
                                base_strategy,
                                direction,
                                trade_id,
                                close_time,
                                close_price,
                                result,
                                pl_pips,
                                ts,
                            ),
                        )
                        reentry_existing[reentry_key] = (
                            trade_id,
                            close_time_dt or last_reentry_close,
                        )
        self._con.commit()

        fallback_cd = cooldown_seconds
        cooldown_lookup: Dict[str, int] = dict(cooldown_map or {})
        if cooldown_lookup:
            fallback_cd = int(min(cooldown_lookup.values()))

        if new_losses:
            payload = []
            for pocket, trade_id, closed_at, loss_jpy, loss_pips in new_losses:
                last_cluster = self._cluster_last_trade_id.get(pocket, 0)
                if trade_id <= last_cluster:
                    continue
                payload.append(
                    (pocket, trade_id, closed_at.isoformat(), loss_jpy, loss_pips)
                )
                self._cluster_last_trade_id[pocket] = max(last_cluster, trade_id)
            if payload:
                self._con.executemany(
                    """
                    INSERT OR IGNORE INTO pocket_loss_window(pocket, trade_id, closed_at, loss_jpy, loss_pips)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    payload,
                )

        cutoff = (now_dt - timedelta(minutes=self._loss_window_minutes)).isoformat()
        self._con.execute(
            "DELETE FROM pocket_loss_window WHERE closed_at < ?", (cutoff,)
        )

        cluster_stats: Dict[str, Dict[str, float]] = {}
        for row in self._con.execute(
            "SELECT pocket, COUNT(*) AS cnt, SUM(loss_jpy) AS total_jpy, SUM(loss_pips) AS total_pips FROM pocket_loss_window GROUP BY pocket"
        ):
            pocket = row["pocket"]
            if not pocket:
                continue
            cluster_stats[pocket] = {
                "count": float(row["cnt"] or 0.0),
                "loss_jpy": float(row["total_jpy"] or 0.0),
                "loss_pips": float(row["total_pips"] or 0.0),
                "cooldown": int(cooldown_lookup.get(pocket, fallback_cd)),
            }

        self._weight_hint = {}
        if cluster_stats:
            self._apply_loss_clusters(
                cluster_stats,
                range_active=range_active,
                atr_pips=atr_pips,
                vol_5m=vol_5m,
                now=now_dt,
            )

        self._recent_profile = self._build_recent_profile(rows, now_dt)
        self._con.commit()

    def _apply_loss_clusters(
        self,
        cluster_stats: Dict[str, Dict[str, float]],
        *,
        range_active: bool = False,
        atr_pips: Optional[float] = None,
        vol_5m: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> None:
        if not cluster_stats:
            return
        now_dt = _coerce_utc(now)
        atr_val = float(atr_pips or 0.0)
        vol_val = float(vol_5m or 0.0)

        if self._cooldown_disabled or self._cluster_cooldown_disabled:
            self._drop_cluster_cooldowns(cluster_stats.keys())
            return

        for pocket, stats in cluster_stats.items():
            count = int(stats.get("count", 0) or 0)
            loss_jpy = float(stats.get("loss_jpy", 0.0) or 0.0)
            loss_pips = float(stats.get("loss_pips", 0.0) or 0.0)
            base_cd = int(stats.get("cooldown", 0) or 0)
            if base_cd <= 0:
                base_cd = 900
            # エントリー頻度を確保するため、micro/scalpのクラスターCDを短縮
            if pocket in {"micro", "scalp"}:
                base_cd = min(base_cd, 180)
            elif pocket == "macro":
                base_cd = min(base_cd, 300)

            if pocket == "scalp" and self._scalp_cluster_cooldown_disabled:
                self._drop_cluster_cooldowns({pocket})
                continue

            severity = 0.0
            if count >= 3:
                severity += 0.6
            elif count == 2:
                severity += 0.3
            if loss_pips >= 40 or loss_jpy >= 5000:
                severity += 0.6
            elif loss_pips >= 20 or loss_jpy >= 2500:
                severity += 0.4
            elif loss_pips >= 10 or loss_jpy >= 1200:
                severity += 0.2
            if range_active:
                severity += 0.2
            if atr_val >= 2.8:
                severity += 0.1
            if vol_val >= 1.6:
                severity += 0.1

            if severity <= 0:
                self._weight_hint.pop(pocket, None)
                continue

            base_seconds = base_cd
            seconds = int(base_seconds * (1.0 + severity))
            seconds = max(seconds, base_seconds)
            seconds = min(seconds, 4 * base_seconds)

            # 低ボラ・小損のときはクールダウンを短縮し、再開までの待ち時間を減らす
            low_vol_soft = atr_val <= 1.4 and vol_val <= 1.0
            small_losses = loss_pips <= 12 and loss_jpy <= 2000
            if pocket in {"micro", "scalp"} and low_vol_soft and small_losses:
                seconds = max(int(seconds * 0.55), int(base_seconds * 0.4))
            elif pocket == "macro" and low_vol_soft and small_losses:
                seconds = max(int(seconds * 0.7), int(base_seconds * 0.5))

            reason = f"loss_cluster_{count or 1}"
            self.ensure_cooldown(
                pocket,
                "long",
                reason=reason,
                seconds=seconds,
                now=now_dt,
            )
            self.ensure_cooldown(
                pocket,
                "short",
                reason=reason,
                seconds=seconds,
                now=now_dt,
            )
            self._weight_hint[pocket] = max(0.3, round(1.0 - min(severity * 0.25, 0.6), 3))
            logging.info(
                "[STAGE] cluster cooldown pocket=%s count=%s loss_pips=%.1f jpy=%.1f sec=%s atr=%.2f vol=%.2f",
                pocket,
                count,
                loss_pips,
                loss_jpy,
                seconds,
                atr_val,
                vol_val,
            )

    def _drop_cluster_cooldowns(self, pockets) -> None:
        items = [p for p in pockets if p]
        if not items:
            return
        placeholders = ",".join("?" for _ in items)
        self._con.execute(
            f"DELETE FROM stage_cooldown WHERE pocket IN ({placeholders}) AND reason LIKE 'loss_cluster_%'",
            tuple(items),
        )
        self._con.commit()

    def _build_recent_profile(
        self, rows: List[sqlite3.Row], now_dt: datetime
    ) -> Dict[str, Dict[str, float]]:
        """直近の勝ち負け分布をシンプルに集計する。"""
        profile: Dict[str, Dict[str, float]] = {}
        window_sec = 6 * 3600  # last 6h
        threshold_ts = now_dt - timedelta(seconds=window_sec)
        for row in rows:
            try:
                trade_id = int(row["id"])
                pocket = str(row["pocket"] or "")
                units = int(row["units"] or 0)
                pl_jpy = float(row["realized_pl"] or 0.0)
                pl_pips = float(row["pl_pips"] or 0.0)
            except Exception:
                continue
            if not pocket or units == 0:
                continue
            # trades table doesn't include close_time here; approximate recency by id ordering
            # and assume IDs are roughly monotonic in time; this is a lightweight sanity metric.
            key = pocket
            stats = profile.setdefault(
                key,
                {
                    "win_count": 0.0,
                    "loss_count": 0.0,
                    "win_jpy": 0.0,
                    "loss_jpy": 0.0,
                    "win_pips": 0.0,
                    "loss_pips": 0.0,
                },
            )
            if pl_jpy > _MIN_LOSS_JPY:
                stats["win_count"] += 1.0
                stats["win_jpy"] += pl_jpy
                stats["win_pips"] += pl_pips
            elif pl_jpy < -_MIN_LOSS_JPY:
                stats["loss_count"] += 1.0
                stats["loss_jpy"] += abs(pl_jpy)
                stats["loss_pips"] += abs(pl_pips)
        for stats in profile.values():
            stats["sample_size"] = float(stats.get("win_count", 0.0) + stats.get("loss_count", 0.0))
        return profile

    def _decay_loss_streaks(self, now: datetime) -> None:
        current = _coerce_utc(now)
        try:
            threshold = current - timedelta(minutes=self._loss_decay_minutes)
        except Exception:
            return
        rows = self._con.execute(
            "SELECT pocket, direction, lose_streak, updated_at FROM stage_history"
        ).fetchall()
        changed = False
        for row in rows:
            lose_streak = int(row["lose_streak"] or 0)
            if lose_streak <= 0:
                continue
            try:
                updated_at = datetime.fromisoformat(row["updated_at"])
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
                else:
                    updated_at = updated_at.astimezone(timezone.utc)
            except Exception:
                updated_at = threshold - timedelta(minutes=1)
            if updated_at > threshold:
                continue
            new_lose = max(0, lose_streak - 1)
            self._con.execute(
                """
                UPDATE stage_history
                SET lose_streak=?, updated_at=?
                WHERE pocket=? AND direction=?
                """,
                (
                    new_lose,
                    now.isoformat(),
                    row["pocket"],
                    row["direction"],
                ),
            )
            changed = True
        if changed:
            self._con.commit()

    def get_loss_profile(self, pocket: str, direction: str) -> Tuple[int, int]:
        row = self._con.execute(
            "SELECT lose_streak, win_streak FROM stage_history WHERE pocket=? AND direction=?",
            (pocket, direction),
        ).fetchone()
        if not row:
            return 0, 0
        return int(row["lose_streak"] or 0), int(row["win_streak"] or 0)

    def size_multiplier(self, pocket: str, direction: str) -> float:
        """連敗時はより強くサイズを縮小し、連勝時はゆるやかに抑制。

        例:
          - 1連敗: 0.75x
          - 2連敗: 0.6x
          - 3連敗以上: 0.5x（Pocket別の下限あり）
          - 2連勝: 0.9x, 3連勝: 0.8x
        """
        lose_streak, win_streak = self.get_loss_profile(pocket, direction)
        factor = 1.0
        if lose_streak >= 3:
            factor *= 0.5
        elif lose_streak == 2:
            factor *= 0.6
        elif lose_streak == 1:
            factor *= 0.75

        if win_streak >= 3:
            factor *= 0.8
        elif win_streak >= 2:
            factor *= 0.9

        reverse_brake = pocket == "macro" and lose_streak > 0 and _in_jst_reverse_window()
        if reverse_brake:
            factor = min(factor, _STAGE_REVERSE_LOSS_FLOOR)

        floor = max(0.3, _POCKET_SIZE_FLOOR.get(pocket, 0.4))
        if reverse_brake:
            floor = min(floor, _STAGE_REVERSE_LOSS_FLOOR)
        return max(floor, round(factor, 3))

    @property
    def recent_profiles(self) -> Dict[str, Dict[str, float]]:
        return {pocket: dict(stats) for pocket, stats in self._recent_profile.items()}

    def get_recent_profiles(self) -> Dict[str, Dict[str, float]]:
        # Backward compat helper for callers expecting a method.
        return self.recent_profiles

    @property
    def weight_hints(self) -> Dict[str, float]:
        return dict(self._weight_hint)

    def set_weight_hint(self, pocket: str, value: Optional[float]) -> None:
        if value is None:
            self._weight_hint.pop(pocket, None)
        else:
            self._weight_hint[pocket] = float(value)
