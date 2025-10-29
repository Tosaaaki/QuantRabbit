"""
execution.stage_tracker
~~~~~~~~~~~~~~~~~~~~~~~
ステージ進行や再エントリーのクールダウンを SQLite で管理する補助モジュール。
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_DB_PATH = Path("logs/stage_state.db")
_TRADES_DB = Path("logs/trades.db")
_LOSS_WINDOW_MINUTES = 12
_MIN_LOSS_JPY = 1.0


def _ensure_row_factory(con: sqlite3.Connection) -> None:
    con.row_factory = sqlite3.Row


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
        self._con = sqlite3.connect(self._path)
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
            CREATE TABLE IF NOT EXISTS pocket_loss_window (
                pocket TEXT NOT NULL,
                trade_id INTEGER PRIMARY KEY,
                closed_at TEXT NOT NULL,
                loss_jpy REAL NOT NULL,
                loss_pips REAL NOT NULL
            )
            """
        )
        self._con.commit()
        self._loss_window_minutes = _LOSS_WINDOW_MINUTES
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

    def clear_cooldown(
        self,
        pocket: str,
        direction: str,
        *,
        reason: Optional[str] = None,
    ) -> None:
        if reason:
            self._con.execute(
                "DELETE FROM stage_cooldown WHERE pocket=? AND direction=? AND reason=?",
                (pocket, direction, reason),
            )
        else:
            self._con.execute(
                "DELETE FROM stage_cooldown WHERE pocket=? AND direction=?",
                (pocket, direction),
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
        limit = datetime.fromisoformat(row["cooldown_until"])
        current = now or datetime.utcnow()
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
        remaining = int((info.cooldown_until - (now or datetime.utcnow())).total_seconds())
        return True, max(1, remaining), info.reason

    def update_loss_streaks(
        self,
        *,
        trades_db: Path | None = None,
        now: Optional[datetime] = None,
        cooldown_seconds: int = 900,
        cooldown_map: Optional[Dict[str, int]] = None,
        range_active: bool = False,
        atr_pips: Optional[float] = None,
        vol_5m: Optional[float] = None,
    ) -> None:
        trades_path = trades_db or _TRADES_DB
        if not trades_path.exists():
            return
        conn = sqlite3.connect(trades_path)
        _ensure_row_factory(conn)
        rows = conn.execute(
            "SELECT id, pocket, units, pl_pips, realized_pl, close_time FROM trades WHERE close_time IS NOT NULL ORDER BY id ASC"
        ).fetchall()
        conn.close()
        if not rows:
            return
        rows = list(rows)

        existing: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
        for row in self._con.execute(
            "SELECT pocket, direction, last_trade_id, lose_streak, win_streak FROM stage_history"
        ):
            existing[(row["pocket"], row["direction"])] = (
                int(row["last_trade_id"] or 0),
                int(row["lose_streak"] or 0),
                int(row["win_streak"] or 0),
            )

        # 正規化: now が timezone-aware なら UTC に変換して tzinfo を外す
        if now is not None:
            if isinstance(now, datetime) and now.tzinfo is not None:
                now_dt = now.astimezone(timezone.utc).replace(tzinfo=None)
            else:
                now_dt = now
        else:
            now_dt = datetime.utcnow()
        ts = now_dt.isoformat()
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
            if pl_jpy < -_MIN_LOSS_JPY:
                close_dt = self._parse_iso(row["close_time"], now_dt)
                loss_jpy = abs(pl_jpy)
                loss_pips = abs(float(row["pl_pips"] or 0.0))
                new_losses.append((pocket, trade_id, close_dt, loss_jpy, loss_pips))

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

    def get_loss_profile(self, pocket: str, direction: str) -> Tuple[int, int]:
        row = self._con.execute(
            "SELECT lose_streak, win_streak FROM stage_history WHERE pocket=? AND direction=?",
            (pocket, direction),
        ).fetchone()
        if not row:
            return 0, 0
        return int(row["lose_streak"] or 0), int(row["win_streak"] or 0)

    def size_multiplier(self, pocket: str, direction: str) -> float:
        """連敗での縮小と、直近パフォーマンスに基づく緩やかな拡大を両立させる。"""
        lose_streak, win_streak = self.get_loss_profile(pocket, direction)
        factor = 1.0
        if lose_streak >= 4:
            factor *= 0.5
        elif lose_streak == 3:
            factor *= 0.6
        elif lose_streak == 2:
            factor *= 0.8
        elif lose_streak == 1:
            factor *= 0.95

        # ストリークが続く場合の軽い調整（縮小し過ぎないよう控えめにする）
        if win_streak >= 4:
            factor *= 1.04
        elif win_streak >= 2:
            factor *= 1.02

        profile = self._recent_profile.get(pocket, {}) or {}
        weight_hint = self._weight_hint.get(pocket, 0.0)
        win_rate = float(profile.get("win_rate", 0.0) or 0.0)
        sample_size = int(profile.get("sample_size", 0) or 0)
        avg_win = float(profile.get("avg_win_pips", 0.0) or 0.0)
        avg_loss = float(profile.get("avg_loss_pips", 0.0) or 0.0)
        rr = avg_win / avg_loss if avg_loss > 0 else avg_win

        hot_sample = sample_size >= 6 and win_rate >= 0.58 and rr >= 1.05
        elite_sample = sample_size >= 10 and win_rate >= 0.62 and rr >= 1.15
        if hot_sample:
            boost = 1.05
            if elite_sample:
                boost = 1.1
            if pocket == "scalp" and weight_hint <= 0.05:
                boost = min(boost, 1.04)
            factor *= boost
        else:
            # スキャル pocket はフォーカスから外れている場合に微縮小する
            if pocket == "scalp" and weight_hint < 0.02:
                factor *= 0.9

        return max(0.5, min(1.1, round(factor, 3)))

    def get_recent_profile(self, pocket: str) -> Dict[str, float]:
        return dict(self._recent_profile.get(pocket, {}))

    def get_recent_profiles(self) -> Dict[str, Dict[str, float]]:
        return {key: dict(val) for key, val in self._recent_profile.items()}

    def set_weight_hint(self, pocket: str, weight: Optional[float]) -> None:
        if weight is None:
            self._weight_hint.pop(pocket, None)
            return
        try:
            value = float(weight)
        except (TypeError, ValueError):
            return
        if value < 0:
            value = 0.0
        self._weight_hint[pocket] = value

    def _apply_loss_clusters(
        self,
        cluster_stats: Dict[str, Dict[str, float]],
        *,
        range_active: bool,
        atr_pips: Optional[float],
        vol_5m: Optional[float],
        now: datetime,
    ) -> None:
        fallback_cd = 900
        if cluster_stats:
            try:
                fallback_cd = int(
                    min(
                        stats.get("cooldown", 900) or 900
                        for stats in cluster_stats.values()
                    )
                )
            except Exception:
                fallback_cd = 900

        for pocket, stats in cluster_stats.items():
            thresholds = self._compute_cluster_thresholds(
                pocket,
                base_cooldown=int(stats.get("cooldown", fallback_cd) or fallback_cd),
                range_active=range_active,
                atr_pips=atr_pips,
                vol_5m=vol_5m,
            )
            count = int(stats.get("count", 0))
            loss_pips = float(stats.get("loss_pips", 0.0))
            apply_needed = count >= thresholds["count"] or loss_pips >= thresholds["pips"]
            if not apply_needed:
                recovery_count = max(0, thresholds["count"] * 0.2)
                recovery_pips = thresholds["pips"] * 0.2
                if count <= recovery_count and loss_pips <= recovery_pips:
                    for direction in ("long", "short"):
                        info = self.get_cooldown(pocket, direction, now)
                        if info and info.reason == "loss_cluster":
                            self.clear_cooldown(pocket, direction, reason="loss_cluster")
                            logging.info(
                                "[STAGE] loss cluster recovery pocket=%s direction=%s remaining_count=%d loss_pips=%.2f",
                                pocket,
                                direction,
                                count,
                                loss_pips,
                            )
                continue
            seconds = thresholds["cooldown"]
            applicable_dirs: List[str] = []
            for direction in ("long", "short"):
                lose_streak, _ = self.get_loss_profile(pocket, direction)
                if lose_streak > 0:
                    applicable_dirs.append(direction)
            if not applicable_dirs:
                applicable_dirs = ["long", "short"]

            for direction in ("long", "short"):
                if direction not in applicable_dirs:
                    self.clear_cooldown(pocket, direction, reason="loss_cluster")

            applied = False
            for direction in applicable_dirs:
                info = self.get_cooldown(pocket, direction, now)
                if info and info.reason == "loss_cluster":
                    remaining = int((info.cooldown_until - now).total_seconds())
                    if remaining >= seconds * 0.6:
                        continue
                self.set_cooldown(
                    pocket,
                    direction,
                    reason="loss_cluster",
                    seconds=seconds,
                    now=now,
                )
                applied = True
            if applied:
                logging.info(
                    "[STAGE] loss cluster cooldown pocket=%s count=%d loss_pips=%.2f cooldown=%ds",
                    pocket,
                    count,
                    loss_pips,
                    seconds,
                )

    def _compute_cluster_thresholds(
        self,
        pocket: str,
        *,
        base_cooldown: int,
        range_active: bool,
        atr_pips: Optional[float],
        vol_5m: Optional[float],
    ) -> Dict[str, float]:
        count = 3
        pips = 6.0
        cooldown = max(120, base_cooldown)

        if pocket == "micro":
            count = 2
            pips = 4.0
        elif pocket == "scalp":
            count = 2
            pips = 3.0

        if range_active:
            count = max(1, count - 1)
            pips *= 0.7
            cooldown = int(cooldown * 1.2)

        atr = atr_pips or 0.0
        if pocket == "macro":
            if atr and atr < 9.5:
                count = max(1, count - 1)
                pips *= 0.85
                cooldown = int(cooldown * 1.1)
            elif atr and atr > 14:
                count += 1
                pips *= 1.1
                cooldown = max(int(cooldown * 0.9), 300)
        else:
            if atr and atr < 6.5:
                count = max(1, count - 1)
                pips *= 0.8
                cooldown = int(cooldown * 1.15)
            elif atr and atr > 11.5:
                count += 1
                pips *= 1.15
                cooldown = max(int(cooldown * 0.85), 180)

        if vol_5m is not None:
            if vol_5m < 0.8:
                cooldown = int(cooldown * 1.1)
            elif vol_5m > 1.6:
                cooldown = max(int(cooldown * 0.9), 150)
                pips *= 1.05

        return {"count": count, "pips": pips, "cooldown": max(120, cooldown)}

    def _build_recent_profile(
        self, rows: List[sqlite3.Row], now: datetime
    ) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        if not rows:
            return result
        recent = rows[-150:]
        stats: Dict[str, Dict[str, List[float]]] = {}
        cutoff = now - timedelta(days=3)
        for row in recent:
            pocket = row["pocket"] or ""
            if not pocket:
                continue
            pl_pips = float(row["pl_pips"] or 0.0)
            if abs(pl_pips) < 0.05:
                continue
            close_dt = self._parse_iso(row["close_time"], now)
            if close_dt < cutoff:
                continue
            bucket = stats.setdefault(pocket, {"wins": [], "losses": [], "all": []})
            bucket["all"].append(pl_pips)
            if pl_pips > 0:
                bucket["wins"].append(pl_pips)
            elif pl_pips < 0:
                bucket["losses"].append(abs(pl_pips))

        for pocket, data in stats.items():
            total = len(data["all"])
            if total == 0:
                continue
            wins = data["wins"]
            losses = data["losses"]
            win_rate = len(wins) / total
            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            result[pocket] = {
                "win_rate": round(win_rate, 4),
                "avg_win_pips": round(avg_win, 3),
                "avg_loss_pips": round(avg_loss, 3),
                "sample_size": total,
            }
        return result

    @staticmethod
    def _parse_iso(raw: Optional[str], fallback: datetime) -> datetime:
        if not raw:
            return fallback
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return fallback
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
