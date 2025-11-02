"""Utility helpers for accessing trade history across backends.

Provides a Firestore-backed repository with transparent SQLite fallback so
Cloud Run services share a consistent trading ledger while local tools keep
working without additional configuration.  Read operations are cached for a
short period to avoid hammering Firestore on the 60s scheduler loop.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - optional dependency in local tests
    from google.cloud import firestore  # type: ignore

    _FIRESTORE_AVAILABLE = True
except Exception:  # noqa: BLE001
    firestore = None  # type: ignore
    _FIRESTORE_AVAILABLE = False


_DEFAULT_COLLECTION = os.getenv("TRADE_REPOSITORY_COLLECTION", "trades")
_FIRESTORE_PROJECT = os.getenv("TRADE_REPOSITORY_PROJECT")
_CACHE_TTL = float(os.getenv("TRADE_REPOSITORY_CACHE_TTL", "30"))
_MAX_FETCH = int(os.getenv("TRADE_REPOSITORY_MAX_FETCH", "1200"))
_SQLITE_PATH = Path(os.getenv("TRADE_REPOSITORY_SQLITE", "logs/trades.db"))


@dataclass(frozen=True)
class TradeSnapshot:
    pocket: Optional[str]
    strategy: Optional[str]
    units: Optional[int]
    pl_pips: Optional[float]
    state: Optional[str]
    close_time: Optional[datetime]
    close_reason: Optional[str]
    realized_pl: Optional[float]


def _parse_close_time(raw: object) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.astimezone(timezone.utc)
    try:
        value = str(raw)
        if not value:
            return None
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except Exception:  # noqa: BLE001
        return None


class TradeRepository:
    """Read-only trade history accessor with Firestore primary backend."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[Tuple[str, Optional[str], int], Tuple[float, List[TradeSnapshot]]] = {}
        self._firestore_client = self._init_firestore()
        self._sqlite_conn = self._init_sqlite()

    def _init_firestore(self):  # type: ignore[override]
        if not _FIRESTORE_AVAILABLE:
            return None
        prefer_firestore = os.getenv("TRADE_REPOSITORY_DISABLE_FIRESTORE", "false").lower() != "true"
        if not prefer_firestore:
            return None
        try:
            if _FIRESTORE_PROJECT:
                return firestore.Client(project=_FIRESTORE_PROJECT)
            return firestore.Client()
        except Exception as exc:  # noqa: BLE001
            logging.warning("[trade_repo] Firestore init failed: %s", exc)
            return None

    def _init_sqlite(self) -> Optional[sqlite3.Connection]:
        if os.getenv("TRADE_REPOSITORY_DISABLE_SQLITE", "false").lower() == "true":
            return None
        try:
            _SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(_SQLITE_PATH, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as exc:  # noqa: BLE001
            logging.warning("[trade_repo] SQLite init failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Public helpers used by risk guards / strategies
    # ------------------------------------------------------------------

    def recent_trades(
        self,
        pocket: Optional[str] = None,
        *,
        limit: int = 200,
        closed_only: bool = True,
        since: Optional[datetime] = None,
    ) -> List[TradeSnapshot]:
        """Return recent trades ordered most-recent first.

        Results are cached for ``TRADE_REPOSITORY_CACHE_TTL`` seconds per key.
        """

        if limit <= 0:
            return []

        key = ("closed" if closed_only else "all", pocket or "*", min(limit, _MAX_FETCH))
        now = time.time()

        with self._lock:
            cached = self._cache.get(key)
            if cached and now - cached[0] < _CACHE_TTL:
                records = cached[1]
            else:
                records = self._fetch_recent(pocket, closed_only=closed_only, limit=min(limit, _MAX_FETCH))
                self._cache[key] = (now, records)

        if since is None:
            return records[:limit]

        cutoff = since.astimezone(timezone.utc)
        filtered = [rec for rec in records if rec.close_time and rec.close_time >= cutoff]
        if len(filtered) >= limit or len(records) < limit:
            return filtered[:limit]

        # Cache may not hold enough items because of the cutoff; refresh once without cache.
        records = self._fetch_recent(pocket, closed_only=closed_only, limit=min(limit * 2, _MAX_FETCH))
        with self._lock:
            self._cache[key] = (now, records)
        filtered = [rec for rec in records if rec.close_time and rec.close_time >= cutoff]
        return filtered[:limit]

    def open_trades(
        self,
        pocket: Optional[str] = None,
        *,
        limit: int = 200,
    ) -> List[TradeSnapshot]:
        limit = min(max(limit, 1), _MAX_FETCH)
        key = ("open", pocket or "*", limit)
        now = time.time()

        with self._lock:
            cached = self._cache.get(key)
            if cached and now - cached[0] < _CACHE_TTL:
                return cached[1]

        records = self._fetch_open(pocket, limit=limit)
        with self._lock:
            self._cache[key] = (now, records)
        return records

    def count_open_trades(self, pocket: Optional[str] = None) -> Optional[int]:
        records = self.open_trades(pocket, limit=_MAX_FETCH)
        if not records and not (self._firestore_client or self._sqlite_conn):
            return None
        return sum(1 for rec in records if (rec.state or "").upper() == "OPEN")

    # ------------------------------------------------------------------
    # Write helpers (Firestore primary; SQLite write-through is not provided)
    # ------------------------------------------------------------------

    def upsert_trades(self, rows: Iterable[Dict[str, object]]) -> int:
        """Upsert trade documents into Firestore.

        - Uses `ticket_id` as document ID.
        - Merge semantics so partial updates are safe.
        - No-ops if Firestore client is unavailable.
        Returns number of rows attempted.
        """
        rows_list = list(rows)
        if not rows_list:
            return 0
        if self._firestore_client is None:
            return 0

        batch = self._firestore_client.batch()
        count = 0
        committed = 0
        for row in rows_list:
            ticket = str(row.get("ticket_id") or "").strip()
            if not ticket:
                continue
            doc = self._firestore_client.collection(_DEFAULT_COLLECTION).document(ticket)
            data = self._sanitize_row_for_firestore(row)
            batch.set(doc, data, merge=True)
            count += 1
            if count % 400 == 0:
                try:
                    batch.commit()
                    committed += 400
                except Exception as exc:  # noqa: BLE001
                    logging.warning("[trade_repo] Firestore batch commit failed: %s", exc)
                batch = self._firestore_client.batch()
        if count % 400:
            try:
                batch.commit()
                committed += count % 400
            except Exception as exc:  # noqa: BLE001
                logging.warning("[trade_repo] Firestore final batch commit failed: %s", exc)
        return committed

    @staticmethod
    def _sanitize_row_for_firestore(row: Dict[str, object]) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for key in (
            "pocket",
            "instrument",
            "units",
            "entry_price",
            "close_price",
            "pl_pips",
            "entry_time",
            "close_time",
            "strategy",
            "macro_regime",
            "micro_regime",
            "close_reason",
            "realized_pl",
            "unrealized_pl",
            "state",
            "version",
            "updated_at",
        ):
            if key not in row:
                continue
            val = row.get(key)
            if key in ("entry_price", "close_price", "pl_pips", "realized_pl", "unrealized_pl"):
                try:
                    if val is not None:
                        val = float(val)  # type: ignore[assignment]
                except (TypeError, ValueError):
                    val = None
            elif key == "units":
                try:
                    if val is not None:
                        val = int(val)  # type: ignore[assignment]
                except (TypeError, ValueError):
                    val = None
            elif key in ("entry_time", "close_time", "updated_at"):
                if isinstance(val, str):
                    # leave as ISO string; Firestore client can store string
                    pass
                elif isinstance(val, datetime):  # type: ignore[name-defined]
                    # keep datetime as-is (if ever passed)
                    pass
                elif val is not None:
                    try:
                        val = str(val)
                    except Exception:
                        val = None
            out[key] = val
        return out

    def sum_closed_pips(
        self,
        *,
        pocket: Optional[str] = None,
        lookback: Optional[timedelta] = None,
    ) -> Optional[float]:
        since = None
        if lookback:
            since = datetime.now(timezone.utc) - lookback
        records = self.recent_trades(pocket, limit=_MAX_FETCH, closed_only=True, since=since)
        if not records:
            return 0.0 if self._firestore_client or self._sqlite_conn else None
        total = 0.0
        for rec in records:
            if rec.pl_pips is None:
                continue
            if since and rec.close_time and rec.close_time < since:
                continue
            total += rec.pl_pips
        return total

    def last_closed_trade(self, pocket: Optional[str] = None) -> Optional[TradeSnapshot]:
        trades = self.recent_trades(pocket, limit=1, closed_only=True)
        return trades[0] if trades else None

    # ------------------------------------------------------------------
    # Backend fetchers
    # ------------------------------------------------------------------

    def _fetch_recent(
        self,
        pocket: Optional[str],
        *,
        closed_only: bool,
        limit: int,
    ) -> List[TradeSnapshot]:
        records: List[TradeSnapshot] = []
        if self._firestore_client is not None:
            try:
                records = self._fetch_recent_firestore(pocket, closed_only=closed_only, limit=limit)
            except Exception as exc:  # noqa: BLE001
                logging.warning("[trade_repo] Firestore fetch failed: %s", exc)
        if records:
            return records
        if self._sqlite_conn is not None:
            try:
                return self._fetch_recent_sqlite(pocket, closed_only=closed_only, limit=limit)
            except Exception as exc:  # noqa: BLE001
                logging.warning("[trade_repo] SQLite fetch failed: %s", exc)
        return []

    def _fetch_open(self, pocket: Optional[str], *, limit: int) -> List[TradeSnapshot]:
        records: List[TradeSnapshot] = []
        if self._firestore_client is not None:
            try:
                records = self._fetch_open_firestore(pocket, limit=limit)
            except Exception as exc:  # noqa: BLE001
                logging.warning("[trade_repo] Firestore open fetch failed: %s", exc)
        if records:
            return records
        if self._sqlite_conn is not None:
            try:
                return self._fetch_open_sqlite(pocket, limit=limit)
            except Exception as exc:  # noqa: BLE001
                logging.warning("[trade_repo] SQLite open fetch failed: %s", exc)
        return []

    def _fetch_recent_firestore(
        self,
        pocket: Optional[str],
        *,
        closed_only: bool,
        limit: int,
    ) -> List[TradeSnapshot]:
        assert self._firestore_client is not None  # for mypy
        query = self._firestore_client.collection(_DEFAULT_COLLECTION)
        try:
            query = query.order_by("close_time", direction=firestore.Query.DESCENDING)
        except Exception:  # noqa: BLE001 - Firestore may lack index for order_by
            pass

        fetch_limit = min(max(limit * 3, limit), _MAX_FETCH)
        docs = query.limit(fetch_limit).stream()
        records = [self._snapshot_from_dict(doc.to_dict()) for doc in docs]
        if closed_only:
            records = [r for r in records if (r.state or "").upper() == "CLOSED"]
        if pocket:
            records = [r for r in records if r.pocket == pocket]
        return records[:limit]

    def _fetch_open_firestore(
        self,
        pocket: Optional[str],
        *,
        limit: int,
    ) -> List[TradeSnapshot]:
        assert self._firestore_client is not None
        query = self._firestore_client.collection(_DEFAULT_COLLECTION)
        try:
            query = query.order_by("updated_at", direction=firestore.Query.DESCENDING)
        except Exception:
            pass
        docs = query.limit(min(limit * 3, _MAX_FETCH)).stream()
        records = [self._snapshot_from_dict(doc.to_dict()) for doc in docs]
        records = [r for r in records if (r.state or "").upper() == "OPEN"]
        if pocket:
            records = [r for r in records if r.pocket == pocket]
        return records[:limit]

    def _fetch_recent_sqlite(
        self,
        pocket: Optional[str],
        *,
        closed_only: bool,
        limit: int,
    ) -> List[TradeSnapshot]:
        assert self._sqlite_conn is not None  # for mypy
        cur = self._sqlite_conn.cursor()
        params: List[object] = []
        where: List[str] = []
        if pocket:
            where.append("pocket=?")
            params.append(pocket)
        if closed_only:
            where.append("state='CLOSED'")
        where_clause = f"WHERE {' AND '.join(where)}" if where else ""
        sql = (
            "SELECT pocket, strategy, units, pl_pips, state, close_time, close_reason, realized_pl "
            "FROM trades "
            f"{where_clause} "
            "ORDER BY datetime(close_time) DESC "
            "LIMIT ?"
        )
        params.append(limit)
        rows = cur.execute(sql, params).fetchall()
        return [
            TradeSnapshot(
                pocket=row["pocket"],
                strategy=row["strategy"] if "strategy" in row.keys() else None,
                units=int(row["units"]) if "units" in row.keys() and row["units"] is not None else None,
                pl_pips=float(row["pl_pips"]) if row["pl_pips"] is not None else None,
                state=row["state"],
                close_time=_parse_close_time(row["close_time"]),
                close_reason=row["close_reason"] if "close_reason" in row.keys() else None,
                realized_pl=float(row["realized_pl"]) if row["realized_pl"] is not None else None,
            )
            for row in rows
        ]

    def _fetch_open_sqlite(
        self,
        pocket: Optional[str],
        *,
        limit: int,
    ) -> List[TradeSnapshot]:
        assert self._sqlite_conn is not None
        cur = self._sqlite_conn.cursor()
        params: List[object] = []
        where: List[str] = ["state='OPEN'"]
        if pocket:
            where.append("pocket=?")
            params.append(pocket)
        where_clause = f"WHERE {' AND '.join(where)}"
        sql = (
            "SELECT pocket, strategy, units, pl_pips, state, close_time, close_reason, realized_pl "
            "FROM trades "
            f"{where_clause} "
            "ORDER BY datetime(entry_time) DESC "
            "LIMIT ?"
        )
        params.append(limit)
        rows = cur.execute(sql, params).fetchall()
        return [
            TradeSnapshot(
                pocket=row["pocket"],
                strategy=row["strategy"] if "strategy" in row.keys() else None,
                units=int(row["units"]) if "units" in row.keys() and row["units"] is not None else None,
                pl_pips=float(row["pl_pips"]) if row["pl_pips"] is not None else None,
                state=row["state"],
                close_time=_parse_close_time(row["close_time"]),
                close_reason=row["close_reason"] if "close_reason" in row.keys() else None,
                realized_pl=float(row["realized_pl"]) if row["realized_pl"] is not None else None,
            )
            for row in rows
        ]

    @staticmethod
    def _snapshot_from_dict(data: Dict[str, object]) -> TradeSnapshot:
        return TradeSnapshot(
            pocket=data.get("pocket") if isinstance(data.get("pocket"), str) else None,
            strategy=str(data.get("strategy")) if data.get("strategy") is not None else None,
            units=int(data["units"]) if isinstance(data.get("units"), (int, float)) else None,
            pl_pips=float(data["pl_pips"]) if isinstance(data.get("pl_pips"), (int, float)) else None,
            state=str(data.get("state")) if data.get("state") is not None else None,
            close_time=_parse_close_time(data.get("close_time")),
            close_reason=str(data.get("close_reason")) if data.get("close_reason") is not None else None,
            realized_pl=float(data["realized_pl"]) if isinstance(data.get("realized_pl"), (int, float)) else None,
        )


# Singleton accessor used by code paths that need shared caching.
_REPOSITORY_SINGLETON: Optional[TradeRepository] = None
_SINGLETON_LOCK = threading.Lock()


def get_trade_repository() -> TradeRepository:
    global _REPOSITORY_SINGLETON
    if _REPOSITORY_SINGLETON is None:
        with _SINGLETON_LOCK:
            if _REPOSITORY_SINGLETON is None:
                _REPOSITORY_SINGLETON = TradeRepository()
    return _REPOSITORY_SINGLETON
