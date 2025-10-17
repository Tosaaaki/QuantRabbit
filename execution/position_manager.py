from __future__ import annotations

import os
import pathlib
import sqlite3
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

import requests

from analysis.learning import record_trade_performance
from utils.secrets import get_secret
from strategies.scalp.basic import BasicScalpStrategy


_DB = pathlib.Path("logs/trades.db")
_DB.parent.mkdir(exist_ok=True)

_SCALP_DB_PATH = pathlib.Path("logs/scalp_trades.db")
try:
    _SCALP_CONN = sqlite3.connect(_SCALP_DB_PATH, check_same_thread=False)
    _SCALP_CONN.row_factory = sqlite3.Row
except Exception:
    _SCALP_CONN = None

SCALP_STRATEGY_NAME = BasicScalpStrategy.name
SCALP_POCKET = BasicScalpStrategy.pocket


def _resolve_system_version() -> str:
    try:
        return get_secret("system_version")
    except Exception:
        env_version = os.environ.get("SYSTEM_VERSION")
        if env_version:
            return env_version
        return "V2"


SYSTEM_VERSION = _resolve_system_version()
LEGACY_VERSION = os.environ.get("SYSTEM_VERSION_LEGACY", "V1")

_SYNC_KEY_LAST_TX = "last_transaction_id"


def _auth() -> tuple[str, dict] | None:
    """Return (host, headers) when OANDA credentials are available."""
    try:
        token = get_secret("oanda_token")
        account = get_secret("oanda_account_id")  # noqa: F841 (ensure secret present)
        try:
            practice = get_secret("oanda_practice").lower() == "true"
        except Exception:
            practice = True
        host = "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"
        return host, {"Authorization": f"Bearer {token}"}
    except Exception:
        return None


def _pip_size(instrument: Optional[str]) -> float:
    if instrument and instrument.endswith("_JPY"):
        return 0.01
    return 0.0001


def _normalize_time(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return value


def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_pocket(units: int) -> str:
    return "macro" if abs(units) >= 100000 else "micro"


def _scalp_metadata(trade_id: str) -> Dict[str, Any]:
    if not trade_id or _SCALP_CONN is None:
        return {}
    try:
        row = _SCALP_CONN.execute(
            "SELECT ts, price, units FROM scalp_trades WHERE trade_id=? ORDER BY id DESC LIMIT 1",
            (trade_id,),
        ).fetchone()
    except Exception:
        return {}
    if not row:
        return {}
    entry_time = _normalize_time(row["ts"])
    info: Dict[str, Any] = {
        "pocket": SCALP_POCKET,
        "strategy": SCALP_STRATEGY_NAME,
        "version": SYSTEM_VERSION,
    }
    if entry_time:
        info["entry_time"] = entry_time
    price = row["price"]
    if price is not None:
        info["entry_price"] = float(price)
    units = row["units"]
    if units is not None:
        info["units"] = _safe_int(units)
    return info


def _parse_client_extensions(
    ext: Optional[Dict[str, Any]]
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    if not ext:
        return None, None, None, None, None
    pocket = None
    tag = ext.get("tag") or ""
    if "=" in tag:
        _, v = tag.split("=", 1)
        pocket = v.strip() or None
    strategy = macro = micro = version = None
    comment = ext.get("comment") or ""
    for part in comment.split("|"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        key = k.strip().lower()
        val = v.strip()
        if key == "strategy":
            strategy = val
        elif key == "macro":
            macro = val
        elif key == "micro":
            micro = val
        elif key in ("ver", "version"):
            version = val
    return pocket, strategy, macro, micro, version


class PositionManager:
    def __init__(self) -> None:
        self.con = sqlite3.connect(_DB)
        self._ensure_schema()
        self._last_tx_id = self._get_last_transaction_id()

    def _ensure_schema(self) -> None:
        cur = self.con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ticket_id TEXT UNIQUE,
              pocket TEXT,
              instrument TEXT,
              units INTEGER,
              entry_price REAL,
              close_price REAL,
              pl_pips REAL,
              entry_time TEXT,
              close_time TEXT,
              strategy TEXT,
              macro_regime TEXT,
              micro_regime TEXT,
              close_reason TEXT,
              realized_pl REAL,
              unrealized_pl REAL,
              state TEXT,
              updated_at TEXT,
              version TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_state (
              key TEXT PRIMARY KEY,
              value TEXT
            )
            """
        )
        cur.execute("PRAGMA table_info(trades)")
        cols = {row[1] for row in cur.fetchall()}
        for name, typ in (
            ("pocket", "TEXT"),
            ("instrument", "TEXT"),
            ("units", "INTEGER"),
            ("entry_price", "REAL"),
            ("close_price", "REAL"),
            ("pl_pips", "REAL"),
            ("entry_time", "TEXT"),
            ("close_time", "TEXT"),
            ("strategy", "TEXT"),
            ("macro_regime", "TEXT"),
            ("micro_regime", "TEXT"),
            ("close_reason", "TEXT"),
            ("realized_pl", "REAL"),
            ("unrealized_pl", "REAL"),
            ("state", "TEXT"),
            ("updated_at", "TEXT"),
            ("version", "TEXT"),
        ):
            if name not in cols:
                cur.execute(f"ALTER TABLE trades ADD COLUMN {name} {typ}")
        cur.execute("UPDATE trades SET version = COALESCE(version, ?)", (LEGACY_VERSION,))
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_ticket ON trades(ticket_id)")
        self.con.commit()

    def _get_last_transaction_id(self) -> int:
        cur = self.con.cursor()
        row = cur.execute(
            "SELECT value FROM sync_state WHERE key=?",
            (_SYNC_KEY_LAST_TX,),
        ).fetchone()
        if not row or row[0] is None:
            return 0
        try:
            return int(row[0])
        except ValueError:
            return 0

    def _set_last_transaction_id(self, value: int) -> None:
        cur = self.con.cursor()
        cur.execute(
            """
            INSERT INTO sync_state(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (_SYNC_KEY_LAST_TX, str(value)),
        )
        self.con.commit()
        self._last_tx_id = value

    def _fetch_transactions(self) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        env = _auth()
        if not env:
            print("[PositionManager] OANDA credentials missing; skip fetch transactions")
            return [], None
        host, headers = env
        account = get_secret("oanda_account_id")
        if self._last_tx_id:
            url = f"{host}/v3/accounts/{account}/transactions/sinceid"
            params: Dict[str, Any] = {"id": str(self._last_tx_id), "pageSize": "500"}
        else:
            url = f"{host}/v3/accounts/{account}/transactions"
            params = {"pageSize": "500"}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[PositionManager] Error fetching transactions: {exc}")
            return [], None
        payload = resp.json()
        transactions = payload.get("transactions", []) or []
        last_id = payload.get("lastTransactionID")
        if not last_id and transactions:
            try:
                last_id = transactions[-1].get("id")
            except Exception:
                last_id = None
        try:
            last_tx = int(last_id) if last_id is not None else None
        except (TypeError, ValueError):
            last_tx = None
        return transactions, last_tx

    def _get_trade_details(self, trade_id: str) -> Optional[Dict[str, Any]]:
        env = _auth()
        if not env:
            return None
        host, headers = env
        account = get_secret("oanda_account_id")
        url = f"{host}/v3/accounts/{account}/trades/{trade_id}"
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[PositionManager] Error fetching trade details for {trade_id}: {exc}")
            return None
        trade = resp.json().get("trade", {}) or {}
        pocket, strategy, macro, micro, version = _parse_client_extensions(trade.get("clientExtensions"))
        units = _safe_int(trade.get("initialUnits") or trade.get("currentUnits"))
        entry_price = _safe_float(trade.get("price") or trade.get("averagePrice")) or 0.0
        return {
            "instrument": trade.get("instrument"),
            "entry_price": entry_price,
            "entry_time": _normalize_time(trade.get("openTime")),
            "units": units,
            "pocket": pocket or _infer_pocket(units),
            "strategy": strategy,
            "macro_regime": macro,
            "micro_regime": micro,
            "version": version or LEGACY_VERSION,
        }

    def _calc_pl_pips(self, entry: Optional[float], close: Optional[float], units: int, instrument: Optional[str]) -> Optional[float]:
        if entry is None or close is None or not units:
            return None
        pip = _pip_size(instrument)
        if pip == 0:
            return None
        direction = 1 if units > 0 else -1
        return round((close - entry) * direction / pip, 2)

    def _rows_to_tuples(self, rows: Iterable[Dict[str, Any]]) -> List[Tuple[Any, ...]]:
        order = (
            "ticket_id",
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
        )
        return [tuple(row.get(key) for key in order) for row in rows]

    def _existing_trade(self, trade_id: str) -> Dict[str, Any]:
        cur = self.con.cursor()
        cur.execute(
            """
            SELECT pocket, instrument, units, entry_price, entry_time, strategy,
                   macro_regime, micro_regime, version
            FROM trades WHERE ticket_id=?
            """,
            (trade_id,),
        )
        row = cur.fetchone()
        if not row:
            return {}
        keys = (
            "pocket",
            "instrument",
            "units",
            "entry_price",
            "entry_time",
            "strategy",
            "macro_regime",
            "micro_regime",
            "version",
        )
        return {k: v for k, v in zip(keys, row) if v is not None}

    def _upsert_trades(self, rows: Iterable[Dict[str, Any]]) -> None:
        data = self._rows_to_tuples(rows)
        if not data:
            return
        sql = (
            "INSERT INTO trades("
            "ticket_id,pocket,instrument,units,entry_price,close_price,pl_pips,entry_time,close_time,"
            "strategy,macro_regime,micro_regime,close_reason,realized_pl,unrealized_pl,state,version,updated_at"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(ticket_id) DO UPDATE SET"
            " pocket=excluded.pocket, instrument=excluded.instrument, units=excluded.units,"
            " entry_price=excluded.entry_price, close_price=excluded.close_price, pl_pips=excluded.pl_pips,"
            " entry_time=excluded.entry_time, close_time=excluded.close_time, strategy=excluded.strategy,"
            " macro_regime=excluded.macro_regime, micro_regime=excluded.micro_regime, close_reason=excluded.close_reason,"
            " realized_pl=excluded.realized_pl, unrealized_pl=excluded.unrealized_pl, state=excluded.state,"
            " version=excluded.version, updated_at=excluded.updated_at"
        )
        self.con.executemany(sql, data)
        self.con.commit()

    def _parse_and_save_transactions(self, transactions: List[Dict[str, Any]]) -> Optional[int]:
        if not transactions:
            return None
        rows: List[Dict[str, Any]] = []
        processed: List[int] = []
        now = datetime.utcnow().isoformat(timespec="seconds")
        detail_cache: Dict[str, Dict[str, Any]] = {}

        def merge_scalp_metadata(tid: str, data: Dict[str, Any]) -> Dict[str, Any]:
            info = _scalp_metadata(tid)
            if not info:
                return data
            merged = data.copy()
            for key, value in info.items():
                if value in (None, ""):
                    continue
                if key == "units":
                    if not merged.get("units"):
                        merged["units"] = value
                    continue
                if merged.get(key) in (None, ""):
                    merged[key] = value
            return merged

        def get_detail(tid: str) -> Dict[str, Any]:
            cached = detail_cache.get(tid)
            if cached is not None:
                detail_cache[tid] = merge_scalp_metadata(tid, cached)
                return detail_cache[tid]
            existing = self._existing_trade(tid)
            detail_cache[tid] = merge_scalp_metadata(tid, existing or {})
            return detail_cache[tid]

        def update_cache(tid: str, data: Dict[str, Any]) -> None:
            base = detail_cache.get(tid, {}).copy()
            base.update({k: v for k, v in data.items() if v is not None})
            detail_cache[tid] = merge_scalp_metadata(tid, base)

        def ensure_entry_detail(tid: str, detail: Dict[str, Any]) -> Dict[str, Any]:
            detail = merge_scalp_metadata(tid, detail)
            if detail.get("entry_price") and detail.get("entry_time") and detail.get("units"):
                return detail
            remote = self._get_trade_details(tid)
            if remote:
                update_cache(tid, remote)
                return detail_cache.get(tid, remote)
            return merge_scalp_metadata(tid, detail)

        def resolved_pocket(tid: str, current: Optional[str], units: int) -> str:
            if current:
                return current
            info = _scalp_metadata(tid)
            if info.get("pocket"):
                return info["pocket"]
            return _infer_pocket(units)

        def resolved_strategy(tid: str, current: Optional[str]) -> Optional[str]:
            if current:
                return current
            info = _scalp_metadata(tid)
            return info.get("strategy")

        for tx in transactions:
            tx_type = tx.get("type")
            try:
                tx_id = int(tx.get("id"))
            except (TypeError, ValueError):
                tx_id = None
            if tx_id is not None:
                processed.append(tx_id)
            if tx_type not in {"ORDER_FILL", "TRADE_CLOSE"}:
                continue
            instrument = tx.get("instrument") or "USD_JPY"
            close_price = _safe_float(tx.get("price"))
            close_time = _normalize_time(tx.get("time"))
            reason = tx.get("reason")
            opened = tx.get("tradeOpened") or {}
            closed_list = tx.get("tradesClosed") or []

            def _detail_for(tid: str) -> Dict[str, Any]:
                return get_detail(tid)

            if opened and opened.get("tradeID"):
                trade_id = str(opened.get("tradeID"))
                detail = _detail_for(trade_id)
                units = detail.get("units") or _safe_int(opened.get("units"))
                entry_price = detail.get("entry_price") or _safe_float(opened.get("price") or close_price)
                entry_time = detail.get("entry_time") or close_time
                pocket = resolved_pocket(trade_id, detail.get("pocket"), units)
                strategy = resolved_strategy(trade_id, detail.get("strategy"))
                macro = detail.get("macro_regime")
                micro = detail.get("micro_regime")
                instrument_detail = detail.get("instrument") or instrument
                rows.append(
                    {
                        "ticket_id": trade_id,
                        "pocket": pocket,
                        "instrument": instrument_detail,
                        "units": units,
                        "entry_price": entry_price,
                        "close_price": None,
                        "pl_pips": None,
                        "entry_time": entry_time,
                        "close_time": None,
                        "strategy": strategy,
                        "macro_regime": macro,
                        "micro_regime": micro,
                        "close_reason": None,
                        "realized_pl": 0.0,
                        "unrealized_pl": 0.0,
                        "state": "OPEN",
                        "version": detail.get("version") or LEGACY_VERSION,
                        "updated_at": now,
                    }
                )
                update_cache(
                    trade_id,
                    {
                        "pocket": pocket,
                        "instrument": instrument_detail,
                        "units": units,
                        "entry_price": entry_price,
                        "entry_time": entry_time,
                        "strategy": strategy,
                        "macro_regime": macro,
                        "micro_regime": micro,
                        "version": detail.get("version") or LEGACY_VERSION,
                    },
                )

            for closed in closed_list:
                trade_id = str(closed.get("tradeID") or "")
                if not trade_id:
                    continue
                detail = ensure_entry_detail(trade_id, _detail_for(trade_id))
                units = detail.get("units") or _safe_int(closed.get("units"))
                entry_price = detail.get("entry_price") or _safe_float(closed.get("price"))
                entry_time = detail.get("entry_time") or close_time
                pocket = resolved_pocket(trade_id, detail.get("pocket"), units)
                strategy = resolved_strategy(trade_id, detail.get("strategy"))
                macro = detail.get("macro_regime")
                micro = detail.get("micro_regime")
                instrument_detail = detail.get("instrument") or instrument
                c_price = _safe_float(closed.get("price")) or close_price
                pl_pips = self._calc_pl_pips(entry_price, c_price, units, instrument_detail)
                realized_pl = _safe_float(closed.get("realizedPL") or tx.get("pl"))
                if (pl_pips is None or abs(pl_pips) < 1e-6) and realized_pl is not None and units:
                    pip = _pip_size(instrument_detail)
                    denom = abs(units) * pip
                    if denom:
                        pl_pips = round(realized_pl / denom, 2)
                rows.append(
                    {
                        "ticket_id": trade_id,
                        "pocket": pocket,
                        "instrument": instrument_detail,
                        "units": units,
                        "entry_price": entry_price,
                        "close_price": c_price,
                        "pl_pips": pl_pips,
                        "entry_time": entry_time,
                        "close_time": close_time,
                        "strategy": strategy,
                        "macro_regime": macro,
                        "micro_regime": micro,
                        "close_reason": reason,
                        "realized_pl": realized_pl,
                        "unrealized_pl": 0.0,
                        "state": "CLOSED",
                        "version": detail.get("version") or LEGACY_VERSION,
                        "updated_at": now,
                    }
                )
                if pl_pips is not None:
                    try:
                        record_trade_performance(
                            strategy,
                            macro,
                            micro,
                            pl_pips,
                            context={
                                "entry_time": entry_time,
                                "close_time": close_time,
                                "units": units,
                                "pocket": pocket,
                            },
                        )
                    except Exception as exc:
                        print(f"[PositionManager] learning update error: {exc}")
                update_cache(
                    trade_id,
                    {
                        "pocket": pocket,
                        "instrument": instrument_detail,
                        "units": 0,
                        "entry_price": entry_price,
                        "entry_time": entry_time,
                        "strategy": strategy,
                        "macro_regime": macro,
                        "micro_regime": micro,
                    },
                )
        if rows:
            self._upsert_trades(rows)
        return max(processed) if processed else None

    def _mark_missing_as_closed(self, active_ids: Set[str], timestamp: str) -> None:
        """Close trades that disappeared from the open snapshot."""

        cur = self.con.cursor()
        cur.execute(
            "SELECT ticket_id FROM trades WHERE state IS NULL OR state != 'CLOSED'"
        )
        stale = [row[0] for row in cur.fetchall() if row[0]]
        if not stale:
            return

        missing = [tid for tid in stale if tid not in active_ids]
        if not missing:
            return

        cur.executemany(
            """
            UPDATE trades
            SET state='CLOSED',
                close_time=COALESCE(close_time, ?),
                unrealized_pl=0.0,
                updated_at=?,
                close_reason=COALESCE(close_reason, 'desync_auto_close')
            WHERE ticket_id=?
            """,
            [(timestamp, timestamp, tid) for tid in missing],
        )
        self.con.commit()

    def _sync_open_trades(self) -> None:
        env = _auth()
        if not env:
            return
        host, headers = env
        account = get_secret("oanda_account_id")
        url = f"{host}/v3/accounts/{account}/openTrades"
        try:
            resp = requests.get(url, headers=headers, timeout=7)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[PositionManager] Error fetching open trades: {exc}")
            return
        trades = resp.json().get("trades", []) or []
        now = datetime.utcnow().isoformat(timespec="seconds")
        active_ids: Set[str] = set()
        rows: List[Dict[str, Any]] = []
        for trade in trades:
            trade_id = str(trade.get("id") or trade.get("tradeID") or "")
            if not trade_id:
                continue
            active_ids.add(trade_id)
            instrument = trade.get("instrument") or "USD_JPY"
            units = _safe_int(trade.get("currentUnits") or trade.get("units") or trade.get("initialUnits"))
            entry_price = _safe_float(trade.get("price") or trade.get("averagePrice"))
            open_time = _normalize_time(trade.get("openTime"))
            realized_pl = _safe_float(trade.get("realizedPL"))
            unrealized_pl = _safe_float(trade.get("unrealizedPL"))
            pocket, strategy, macro, micro, version = _parse_client_extensions(trade.get("clientExtensions"))
            pocket = pocket or _infer_pocket(units)
            rows.append(
                {
                    "ticket_id": trade_id,
                    "pocket": pocket,
                    "instrument": instrument,
                    "units": units,
                    "entry_price": entry_price,
                    "close_price": None,
                    "pl_pips": None,
                    "entry_time": open_time,
                    "close_time": None,
                    "strategy": strategy,
                    "macro_regime": macro,
                    "micro_regime": micro,
                    "close_reason": None,
                    "realized_pl": realized_pl,
                    "unrealized_pl": unrealized_pl,
                    "state": "OPEN",
                    "version": version or LEGACY_VERSION,
                    "updated_at": now,
                }
            )
        if rows:
            self._upsert_trades(rows)
        # Mark trades that disappeared from the open snapshot as closed.
        self._mark_missing_as_closed(active_ids, now)

    def sync_trades(self) -> None:
        transactions, last_id = self._fetch_transactions()
        if transactions:
            latest_processed = self._parse_and_save_transactions(transactions)
            if latest_processed is not None:
                self._set_last_transaction_id(latest_processed)
            elif last_id is not None:
                self._set_last_transaction_id(last_id)
        elif last_id is not None and last_id != self._last_tx_id:
            self._set_last_transaction_id(last_id)
        # Always refresh open snapshot so local DB mirrors reality.
        self._sync_open_trades()

    def close(self) -> None:
        self.con.close()
