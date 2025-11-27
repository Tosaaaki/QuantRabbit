"""
analysis.pattern_stats
~~~~~~~~~~~~~~~~~~~~~~
Pattern-based win-rate tracker for sizing boosts.

- Derives a compact pattern tag from the current M1 candle + indicators.
- Loads recent trades from logs/trades.db and aggregates win-rate/PF by
  (pattern, pocket, direction).
- Exposes a boost factor that can be applied to confidence/lot sizing with
  range-mode caps and sample-size guards.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    raw = str(value)
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _bucket(value: float, thresholds: Tuple[float, ...], labels: Tuple[str, ...]) -> str:
    for th, lb in zip(thresholds, labels):
        if value < th:
            return lb
    return labels[-1] if labels else ""


def derive_pattern_signature(
    fac_m1: Mapping[str, Any],
    *,
    action: Optional[str] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Build a compact pattern tag from the latest M1 factors/candle.
    Returns (pattern_tag, meta) where pattern_tag may be None if inputs are incomplete.
    """
    open_px = _safe_float(fac_m1.get("open"), default=None)
    high_px = _safe_float(fac_m1.get("high"), default=None)
    low_px = _safe_float(fac_m1.get("low"), default=None)
    close_px = _safe_float(fac_m1.get("close"), default=None)
    if None in (open_px, high_px, low_px, close_px):
        return None, {}

    spread_pips = max((high_px - low_px) * 100, 0.0)
    body_pips = (close_px - open_px) * 100
    body_ratio = abs(body_pips) / max(spread_pips, 0.01)
    upper_wick = max(high_px - max(open_px, close_px), 0.0) * 100
    lower_wick = max(min(open_px, close_px) - low_px, 0.0) * 100

    if spread_pips < 0.15:
        shape = "flat"
    elif body_ratio >= 0.7:
        shape = "maru_up" if body_pips > 0 else "maru_dn"
    elif body_ratio >= 0.45:
        shape = "trend_up" if body_pips > 0 else "trend_dn"
    elif body_ratio <= 0.2:
        shape = "doji_up" if body_pips > 0 else "doji_dn"
    else:
        shape = "spin_up" if body_pips > 0 else "spin_dn"

    if max(upper_wick, lower_wick) < 0.1:
        wick_bias = "none"
    elif upper_wick > lower_wick * 1.6:
        wick_bias = "upper"
    elif lower_wick > upper_wick * 1.6:
        wick_bias = "lower"
    else:
        wick_bias = "balanced"

    ma10 = _safe_float(fac_m1.get("ma10"))
    ma20 = _safe_float(fac_m1.get("ma20"))
    gap_pips = (ma10 - ma20) * 100 if ma10 and ma20 else (close_px - ma20) * 100 if ma20 else 0.0
    trend_bucket = _bucket(
        gap_pips,
        thresholds=(-0.6, -0.25, 0.25, 0.6),
        labels=("dn_strong", "dn_mild", "flat", "up_mild", "up_strong"),
    )

    rsi = _safe_float(fac_m1.get("rsi"))
    rsi_bucket = _bucket(
        rsi,
        thresholds=(35, 45, 55, 65),
        labels=("os", "mid_low", "neutral", "mid_high", "ob"),
    )

    bbw = _safe_float(fac_m1.get("bbw"))
    vol_bucket = _bucket(
        bbw,
        thresholds=(0.12, 0.2, 0.3),
        labels=("tight", "normal", "wide", "blown"),
    )

    atr_pips = _safe_float(fac_m1.get("atr_pips"), default=None)
    if atr_pips is None or atr_pips == 0.0:
        atr_pips = _safe_float(fac_m1.get("atr")) * 100
    atr_bucket = _bucket(
        atr_pips,
        thresholds=(3.0, 5.0, 7.5, 10.0),
        labels=("ultra_low", "low", "mid", "elevated", "high"),
    )

    meta = {
        "shape": shape,
        "wick": wick_bias,
        "body_pips": round(body_pips, 3),
        "range_pips": round(spread_pips, 3),
        "trend_gap_pips": round(gap_pips, 3),
        "trend_bucket": trend_bucket,
        "rsi": round(rsi, 2),
        "rsi_bucket": rsi_bucket,
        "bbw": round(bbw, 4),
        "vol_bucket": vol_bucket,
        "atr_pips": round(atr_pips, 3),
        "atr_bucket": atr_bucket,
    }
    direction_hint = "long" if (action or "").upper() == "OPEN_LONG" else "short" if action else None
    parts = [
        f"c:{shape}",
        f"w:{wick_bias}",
        f"tr:{trend_bucket}",
        f"rsi:{rsi_bucket}",
        f"vol:{vol_bucket}",
        f"atr:{atr_bucket}",
    ]
    if direction_hint:
        parts.append(f"d:{direction_hint}")
    tag = "|".join(parts)
    return tag, meta


@dataclass
class PatternSummary:
    count: int
    wins: int
    total_pips: float
    gross_profit: float
    gross_loss: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.count if self.count else 0.0

    @property
    def profit_factor(self) -> float:
        if self.gross_loss <= 0:
            return float("inf") if self.gross_profit > 0 else 0.0
        return self.gross_profit / self.gross_loss


@dataclass
class PatternBoost:
    factor: float
    sample_size: int
    win_rate: float
    profit_factor: float


class PatternStats:
    def __init__(
        self,
        *,
        trades_db: Path | str = Path("logs/trades.db"),
        lookback_days: int = 45,
        min_samples: int = 15,
        refresh_seconds: int = 300,
        max_rows: int = 4000,
        max_boost: float = 1.35,
        range_cap: float = 1.1,
    ) -> None:
        self._db_path = Path(trades_db)
        self._lookback = timedelta(days=max(1, lookback_days))
        self._min_samples = max(1, min_samples)
        self._refresh_seconds = max(60, refresh_seconds)
        self._max_rows = max_rows
        self._max_boost = max_boost
        self._range_cap = range_cap
        self._last_refresh = datetime.min.replace(tzinfo=timezone.utc)
        self._cache: dict[tuple[str, str, str], PatternSummary] = {}

    def refresh(self, now: Optional[datetime] = None) -> None:
        if now is None:
            now = datetime.now(timezone.utc)
        else:
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            else:
                now = now.astimezone(timezone.utc)
        if (now - self._last_refresh).total_seconds() < self._refresh_seconds:
            return
        self._last_refresh = now
        self._cache.clear()
        if not self._db_path.exists():
            logger.debug("[PATTERN] trades db not found: %s", self._db_path)
            return

        cutoff = now - self._lookback
        try:
            con = sqlite3.connect(self._db_path)
            con.row_factory = sqlite3.Row
            rows = con.execute(
                """
                SELECT pocket, units, pl_pips, entry_thesis, close_time
                FROM trades
                WHERE close_time IS NOT NULL
                ORDER BY id DESC
                LIMIT ?
                """,
                (self._max_rows,),
            ).fetchall()
            con.close()
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning("[PATTERN] failed to load trades: %s", exc)
            return

        for row in rows:
            close_time = _parse_dt(row["close_time"])
            if close_time and close_time < cutoff:
                continue
            pocket = (row["pocket"] or "").strip().lower()
            if not pocket:
                continue
            units = _safe_float(row["units"])
            direction = "long" if units > 0 else "short"
            try:
                thesis_raw = row["entry_thesis"]
                thesis = json.loads(thesis_raw) if isinstance(thesis_raw, str) else thesis_raw
            except Exception:
                thesis = {}
            pattern = None
            if isinstance(thesis, dict):
                pattern = thesis.get("pattern_tag") or thesis.get("pattern") or thesis.get("pattern_id")
            if not pattern or not isinstance(pattern, str):
                continue
            pl_pips = _safe_float(row["pl_pips"])
            key = (pattern, pocket, direction)
            entry = self._cache.get(key)
            is_win = pl_pips > 0.0
            gain = pl_pips if pl_pips > 0 else 0.0
            loss = abs(pl_pips) if pl_pips < 0 else 0.0
            if entry:
                entry.count += 1
                entry.wins += 1 if is_win else 0
                entry.total_pips += pl_pips
                entry.gross_profit += gain
                entry.gross_loss += loss
            else:
                self._cache[key] = PatternSummary(
                    count=1,
                    wins=1 if is_win else 0,
                    total_pips=pl_pips,
                    gross_profit=gain,
                    gross_loss=loss,
                )

        logger.info("[PATTERN] refreshed %d pattern buckets (lookback=%dd)", len(self._cache), self._lookback.days)

    def evaluate(
        self,
        *,
        pattern_tag: Optional[str],
        pocket: str,
        direction: str,
        range_mode: bool = False,
        now: Optional[datetime] = None,
    ) -> PatternBoost:
        self.refresh(now=now)
        if not pattern_tag:
            return PatternBoost(1.0, 0, 0.0, 0.0)
        key = (pattern_tag, pocket, direction)
        summary = self._cache.get(key)
        if not summary or summary.count < self._min_samples:
            return PatternBoost(1.0, summary.count if summary else 0, summary.win_rate if summary else 0.0, summary.profit_factor if summary else 0.0)
        factor = self._score_to_factor(summary, range_mode=range_mode)
        return PatternBoost(factor=factor, sample_size=summary.count, win_rate=summary.win_rate, profit_factor=summary.profit_factor)

    def _score_to_factor(self, summary: PatternSummary, *, range_mode: bool) -> float:
        wr = summary.win_rate
        pf = summary.profit_factor
        factor = 1.0
        if wr >= 0.62 and pf >= 1.15:
            factor = 1.28
        elif wr >= 0.58 and pf >= 1.05:
            factor = 1.15
        elif wr >= 0.54 and pf >= 1.0:
            factor = 1.08
        else:
            factor = 1.0
        if pf < 0.95:
            factor = min(factor, 1.02)
        factor = min(factor, self._max_boost)
        if range_mode:
            factor = min(factor, self._range_cap)
        return max(0.8, round(factor, 3))
