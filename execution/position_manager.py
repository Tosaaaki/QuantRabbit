from __future__ import annotations

import os
import pathlib
import sqlite3
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from analysis.learning import record_trade_performance
from utils.secrets import get_secret


_DB = pathlib.Path("logs/trades.db")
_DB.parent.mkdir(exist_ok=True)


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

    def _fetch_transactions(self) -> List[Dict[str, Any]]:
        env = _auth()
        if not env:
            print("[PositionManager] OANDA credentials missing; skip fetch transactions")
            return []
        host, headers = env
        account = get_secret("oanda_account_id")
        url = f"{host}/v3/accounts/{account}/transactions"
        params: Dict[str, Any] = {"type": "ORDER_FILL"}
        if self._last_tx_id:
            params["sinceID"] = self._last_tx_id
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[PositionManager] Error fetching transactions: {exc}")
            return []
        return resp.json().get("transactions", []) or []

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

    def _parse_and_save_transactions(self, transactions: List[Dict[str, Any]]) -> None:
        if not transactions:
            return
        rows: List[Dict[str, Any]] = []
        processed: List[int] = []
        now = datetime.utcnow().isoformat(timespec="seconds")
        detail_cache: Dict[str, Optional[Dict[str, Any]]] = {}

        def get_detail(tid: str) -> Optional[Dict[str, Any]]:
            if tid not in detail_cache:
                detail_cache[tid] = self._get_trade_details(tid)
            return detail_cache[tid]

        for tx in transactions:
            tx_type = tx.get("type")
            try:
                tx_id = int(tx.get("id"))
            except (TypeError, ValueError):
                tx_id = None
            if tx_id is not None:
                processed.append(tx_id)
            if tx_type != "ORDER_FILL":
                continue
            instrument = tx.get("instrument") or "USD_JPY"
            close_price = _safe_float(tx.get("price"))
            close_time = _normalize_time(tx.get("time"))
            reason = tx.get("reason")
            closed_list = tx.get("tradesClosed") or []
            if not closed_list:
                continue
            for closed in closed_list:
                trade_id = str(closed.get("tradeID") or "")
                if not trade_id:
                    continue
                detail = get_detail(trade_id)
                if not detail:
                    continue
                units = detail.get("units") or _safe_int(closed.get("units"))
                entry_price = detail.get("entry_price")
                entry_time = detail.get("entry_time")
                pocket = detail.get("pocket") or _infer_pocket(units)
                strategy = detail.get("strategy")
                macro = detail.get("macro_regime")
                micro = detail.get("micro_regime")
                instrument_detail = detail.get("instrument") or instrument
                pl_pips = self._calc_pl_pips(entry_price, close_price, units, instrument_detail)
                realized_pl = _safe_float(closed.get("realizedPL") or tx.get("pl"))
                rows.append(
                    {
                        "ticket_id": trade_id,
                        "pocket": pocket,
                        "instrument": instrument_detail,
                        "units": units,
                        "entry_price": entry_price,
                        "close_price": close_price,
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
                        record_trade_performance(strategy, macro, micro, pl_pips)
                    except Exception as exc:
                        print(f"[PositionManager] learning update error: {exc}")
        if rows:
            self._upsert_trades(rows)
        if processed:
            self._set_last_transaction_id(max(processed))

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
        rows: List[Dict[str, Any]] = []
        for trade in trades:
            trade_id = str(trade.get("id") or trade.get("tradeID") or "")
            if not trade_id:
                continue
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

    def sync_trades(self) -> None:
        transactions = self._fetch_transactions()
        if transactions:
            self._parse_and_save_transactions(transactions)
        # Always refresh open snapshot so local DB mirrors reality.
        self._sync_open_trades()

    def close(self) -> None:
        self.con.close()
