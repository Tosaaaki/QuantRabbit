"""
execution.stage_tracker
~~~~~~~~~~~~~~~~~~~~~~~
ステージ進行や再エントリーのクールダウンを SQLite で管理する補助モジュール。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

_DB_PATH = Path("logs/stage_state.db")
_TRADES_DB = Path("logs/trades.db")


@dataclass(slots=True)
class CooldownInfo:
    pocket: str
    direction: str
    reason: str
    cooldown_until: datetime


class StageTracker:
    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or _DB_PATH
        self._con = sqlite3.connect(self._path)
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
        self._con.commit()

    def close(self) -> None:
        self._con.close()

    def clear_expired(self, now: datetime) -> None:
        ts = now.isoformat()
        self._con.execute(
            "DELETE FROM stage_cooldown WHERE cooldown_until <= ?", (ts,)
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
        ts = (now or datetime.utcnow()).isoformat()
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
        base = now or datetime.utcnow()
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

    def get_cooldown(
        self, pocket: str, direction: str, now: Optional[datetime] = None
    ) -> Optional[CooldownInfo]:
        row = self._con.execute(
            "SELECT reason, cooldown_until FROM stage_cooldown WHERE pocket=? AND direction=?",
            (pocket, direction),
        ).fetchone()
        if not row:
            return None
        limit = datetime.fromisoformat(row[1])
        current = now or datetime.utcnow()
        if current >= limit:
            self._con.execute(
                "DELETE FROM stage_cooldown WHERE pocket=? AND direction=?",
                (pocket, direction),
            )
            self._con.commit()
            return None
        return CooldownInfo(
            pocket=pocket, direction=direction, reason=row[0] or "", cooldown_until=limit
        )

    def is_blocked(
        self, pocket: str, direction: str, now: Optional[datetime] = None
    ) -> Tuple[bool, Optional[int], Optional[str]]:
        info = self.get_cooldown(pocket, direction, now)
        if not info:
            return False, None, None
        remaining = int((info.cooldown_until - (now or datetime.utcnow())).total_seconds())
        return True, max(1, remaining), info.reason

    def update_loss_streaks(
        self,
        *,
        trades_db: Path | None = None,
        now: Optional[datetime] = None,
        cooldown_seconds: int = 900,
        cooldown_map: Optional[Dict[str, int]] = None,
    ) -> None:
        trades_path = trades_db or _TRADES_DB
        if not trades_path.exists():
            return
        conn = sqlite3.connect(trades_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, pocket, units, pl_pips FROM trades WHERE close_time IS NOT NULL ORDER BY id ASC"
        ).fetchall()
        conn.close()
        if not rows:
            return

        existing: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
        for row in self._con.execute("SELECT pocket, direction, last_trade_id, lose_streak, win_streak FROM stage_history"):
            existing[(row[0], row[1])] = (int(row[2] or 0), int(row[3] or 0), int(row[4] or 0))

        ts = (now or datetime.utcnow()).isoformat()
        for row in rows:
            pocket = row["pocket"] or ""
            units = int(row["units"] or 0)
            if not pocket or units == 0:
                continue
            direction = "long" if units > 0 else "short"
            key = (pocket, direction)
            last_id, lose_streak, win_streak = existing.get(key, (0, 0, 0))
            try:
                trade_id = int(row["id"]) if row["id"] is not None else None
            except (TypeError, ValueError):  # noqa: PERF203
                trade_id = None
            if not trade_id:
                continue
            if trade_id <= last_id:
                continue
            pl_pips = float(row["pl_pips"] or 0.0)
            if pl_pips < -0.05:
                lose_streak += 1
                win_streak = 0
            elif pl_pips > 0.05:
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
                self.set_cooldown(
                    pocket,
                    direction,
                    reason="loss_streak",
                    seconds=pocket_cooldown,
                    now=now,
                )
        self._con.commit()

    def get_loss_profile(self, pocket: str, direction: str) -> Tuple[int, int]:
        row = self._con.execute(
            "SELECT lose_streak, win_streak FROM stage_history WHERE pocket=? AND direction=?",
            (pocket, direction),
        ).fetchone()
        if not row:
            return 0, 0
        return int(row[0] or 0), int(row[1] or 0)

    def size_multiplier(self, pocket: str, direction: str) -> float:
        lose_streak, win_streak = self.get_loss_profile(pocket, direction)
        factor = 1.0
        if lose_streak > 0:
            factor *= max(0.4, 1.0 - lose_streak * 0.25)
        if win_streak >= 2:
            factor *= 0.8
        if win_streak >= 3:
            factor *= 0.7
        return max(0.3, round(factor, 3))
