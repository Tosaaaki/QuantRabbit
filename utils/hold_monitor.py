from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Optional, Tuple


class HoldMonitor:
    """
    Utility to compute hold-time ratios from logs/trades.db (or a specified db).
    Used to gate strategies when <60s exits dominate.
    """

    def __init__(
        self,
        db_path: str | Path = "logs/trades.db",
        *,
        lookback_hours: float = 6.0,
        min_samples: int = 60,
    ) -> None:
        self.db_path = Path(db_path)
        self.lookback_hours = max(0.1, float(lookback_hours))
        self.min_samples = max(1, int(min_samples))

    def sample(self) -> Tuple[Optional[float], int, int]:
        """
        Returns (ratio, total_samples, lt60_samples).
        Ratio is None if total < min_samples or no data.
        """
        if not self.db_path.exists():
            return None, 0, 0
        threshold = (
            dt.datetime.now(dt.timezone.utc)
            - dt.timedelta(hours=self.lookback_hours)
        )
        threshold_iso = threshold.isoformat().replace("+00:00", "Z")
        try:
            con = sqlite3.connect(self.db_path)
            con.row_factory = sqlite3.Row
            rows = con.execute(
                """
                WITH stats AS (
                  SELECT
                    (strftime('%s', close_time) - strftime('%s', entry_time)) AS hold_sec
                  FROM trades
                  WHERE state='CLOSED'
                    AND entry_time IS NOT NULL
                    AND close_time IS NOT NULL
                    AND close_time >= ?
                )
                SELECT
                  COUNT(*) AS total,
                  SUM(CASE WHEN hold_sec < 60 THEN 1 ELSE 0 END) AS lt60
                FROM stats
                """,
                (threshold_iso,),
            ).fetchone()
        except sqlite3.Error:
            return None, 0, 0
        finally:
            try:
                con.close()
            except Exception:
                pass
        total = int(rows["total"] or 0)
        lt60 = int(rows["lt60"] or 0)
        if total < self.min_samples or total == 0:
            return None, total, lt60
        ratio = lt60 / total
        return ratio, total, lt60
