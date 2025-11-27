"""
execution.stage_tracker
~~~~~~~~~~~~~~~~~~~~~~~
ステージ進行や再エントリーのクールダウンを SQLite で管理する補助モジュール。
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

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
        self._con.commit()
        self._loss_window_minutes = int(
            os.getenv("LOSS_CLUSTER_WINDOW_MIN", str(_LOSS_WINDOW_MINUTES)) or _LOSS_WINDOW_MINUTES
        )
        self._loss_decay_minutes = int(os.getenv("LOSS_STREAK_DECAY_MINUTES", "120") or 120)
        self._cluster_last_trade_id: Dict[str, int] = {}
        for row in self._con.execute(
            "SELECT pocket, MAX(trade_id) AS last_id FROM pocket_loss_window GROUP BY pocket"
        ):
            pocket = row["pocket"]
            if pocket:
                self._cluster_last_trade_id[pocket] = int(row["last_id"] or 0)
        self._recent_profile: Dict[str, Dict[str, float]] = {}
        self._weight_hint: Dict[str, float] = {}

    def close(self) -> None:
        self._con.close()

    def clear_expired(self, now: datetime) -> None:
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
        if cooldown_seconds > 0:
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
        return CooldownInfo(
            pocket=pocket, direction=direction, reason=row["reason"] or "", cooldown_until=limit
        )

    def is_blocked(
        self, pocket: str, direction: str, now: Optional[datetime] = None
    ) -> Tuple[bool, Optional[int], Optional[str]]:
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
    ) -> None:
        trades_path = trades_db or _TRADES_DB
        if not trades_path.exists():
            return
        conn = sqlite3.connect(trades_path)
        _ensure_row_factory(conn)
        rows = conn.execute(
            "SELECT id, pocket, units, pl_pips, realized_pl, strategy_tag FROM trades WHERE close_time IS NOT NULL ORDER BY id ASC"
        ).fetchall()
        conn.close()
        if not rows:
            return
        rows = list(rows)

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

        now_dt = _coerce_utc(now)
        ts = now_dt.isoformat()
        reverse_window = _in_jst_reverse_window(now)
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
                pocket_cooldown = (
                    (cooldown_map or {}).get(pocket, cooldown_seconds)
                )
                if pocket != "scalp":
                    self.set_cooldown(
                        pocket,
                        direction,
                        reason="loss_streak",
                        seconds=pocket_cooldown,
                        now=now,
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
                        now=now,
                    )

            strategy_tag = row["strategy_tag"] if "strategy_tag" in row.keys() else None
            if strategy_tag:
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
                            now=now,
                        )
                        strat_lose = 0
                        self._con.execute(
                            "UPDATE strategy_history SET lose_streak=? WHERE strategy=?",
                            (strat_lose, strategy_tag),
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

    def _decay_loss_streaks(self, now: datetime) -> None:
        try:
            threshold = now - timedelta(minutes=self._loss_decay_minutes)
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
                if updated_at.tzinfo is not None:
                    # Normalize to naive UTC for consistent comparison
                    updated_at = updated_at.astimezone(timezone.utc).replace(tzinfo=None)
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
