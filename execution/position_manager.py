from __future__ import annotations
import json
import os
import json
import logging
import time
import copy
import sqlite3
import pathlib
import fcntl
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from utils.secrets import get_secret

# --- config ---
# env.toml から OANDA 設定を取得
TOKEN = get_secret("oanda_token")
ACCOUNT = get_secret("oanda_account_id")
PRACT = False  # env.tomlから取得しないため、ここでは固定値とする
REST_HOST = (
    "https://api-fxpractice.oanda.com" if PRACT else "https://api-fxtrade.oanda.com"
)
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

_DB = pathlib.Path("logs/trades.db")
_ORDERS_DB = pathlib.Path("logs/orders.db")
_CHUNK_SIZE = 100
_MAX_FETCH = int(os.getenv("POSITION_MANAGER_MAX_FETCH", "1000"))
_REQUEST_TIMEOUT = float(os.getenv("POSITION_MANAGER_HTTP_TIMEOUT", "7.0"))
_RETRY_STATUS_CODES = tuple(
    int(code.strip())
    for code in os.getenv(
        "POSITION_MANAGER_HTTP_RETRY_CODES", "408,409,425,429,500,502,503,504,520,522"
    ).split(",")
    if code.strip().isdigit()
)
_OPEN_TRADES_CACHE_TTL = float(
    os.getenv("POSITION_MANAGER_OPEN_TRADES_CACHE_TTL", "2.0")
)
_OPEN_TRADES_FAIL_BACKOFF_BASE = float(
    os.getenv("POSITION_MANAGER_OPEN_TRADES_BACKOFF_BASE", "2.5")
)
_OPEN_TRADES_FAIL_BACKOFF_MAX = float(
    os.getenv("POSITION_MANAGER_OPEN_TRADES_BACKOFF_MAX", "60.0")
)
_MANUAL_POCKET_NAME = os.getenv("POSITION_MANAGER_MANUAL_POCKET", "manual")
_KNOWN_POCKETS = {"micro", "macro", "scalp", "scalp_fast"}
# SQLite locking周り
# ロック頻発に備えてデフォルトを広めに取る
_DB_BUSY_TIMEOUT_MS = int(os.getenv("POSITION_MANAGER_DB_BUSY_TIMEOUT_MS", "20000"))
_DB_LOCK_RETRY = int(os.getenv("POSITION_MANAGER_DB_LOCK_RETRY", "6"))
_DB_LOCK_RETRY_SLEEP = float(os.getenv("POSITION_MANAGER_DB_LOCK_RETRY_SLEEP", "0.5"))
# PRAGMA values
_DB_JOURNAL_MODE = os.getenv("POSITION_MANAGER_DB_JOURNAL_MODE", "WAL")
_DB_SYNCHRONOUS = os.getenv("POSITION_MANAGER_DB_SYNCHRONOUS", "NORMAL")
_DB_TEMP_STORE = os.getenv("POSITION_MANAGER_DB_TEMP_STORE", "MEMORY")
_DB_LOCK_PATH = pathlib.Path(os.getenv("POSITION_MANAGER_DB_LOCK_PATH", "logs/trades.db.lock"))
_DB_FILE_LOCK_TIMEOUT = float(os.getenv("POSITION_MANAGER_DB_FILE_LOCK_TIMEOUT", "30.0"))

# Agent-generated client order ID prefixes (qr-...), used to classify pockets.
agent_client_prefixes = tuple(
    os.getenv("AGENT_CLIENT_PREFIXES", "qr-,qs-").split(",")
)
agent_client_prefixes = tuple(p for p in agent_client_prefixes if p)
if not agent_client_prefixes:
    agent_client_prefixes = ("qr-",)
# pockets that belong to this agent
agent_pockets = {"micro", "macro", "scalp", "scalp_fast"}

# Strategy tag normalization: map suffix/abbrev tags to canonical bases for EXIT filtering.
_CANONICAL_STRATEGY_TAGS = {
    # Core strategies
    "TrendMA",
    "Donchian55",
    "H1Momentum",
    "M1Scalper",
    "ImpulseRetrace",
    "RangeFader",
    "PulseBreak",
    "BB_RSI",
    "BB_RSI_Fast",
    "MomentumBurst",
    "MomentumPulse",
    "VolCompressionBreak",
    "MicroMomentumStack",
    "MicroPullbackEMA",
    "MicroLevelReactor",
    "MicroRangeBreak",
    "MicroVWAPBound",
    "MicroVWAPRevert",
    "TrendMomentumMicro",
    # Worker-only tags
    "trend_h1",
    "mirror_spike",
    "mirror_spike_s5",
    "mirror_spike_tight",
    "pullback_s5",
    "pullback_scalp",
    "pullback_runner_s5",
    "impulse_break_s5",
    "impulse_retest_s5",
    "impulse_momentum_s5",
    "squeeze_break_s5",
    "vwap_magnet_s5",
    "fast_scalp",
    "manual_swing",
    "OnePipMakerS1",
    "LondonMomentum",
}
_CANONICAL_TAGS_LOWER = {tag.lower(): tag for tag in _CANONICAL_STRATEGY_TAGS}
_TAG_ALIAS_PREFIXES = {
    "mlr": "MicroLevelReactor",
    "trendma": "TrendMA",
    "donchian": "Donchian55",
    "h1momentum": "H1Momentum",
    "m1scalper": "M1Scalper",
    "bbrsi": "BB_RSI",
    "bb_rsi": "BB_RSI",
}


def _normalize_strategy_tag(tag: object | None) -> str | None:
    if tag is None:
        return None
    tag_str = str(tag).strip()
    if not tag_str:
        return None
    lower = tag_str.lower()
    if lower in _CANONICAL_TAGS_LOWER:
        return _CANONICAL_TAGS_LOWER[lower]
    base = tag_str.split("-", 1)[0].strip()
    if base:
        base_lower = base.lower()
        if base_lower in _CANONICAL_TAGS_LOWER:
            return _CANONICAL_TAGS_LOWER[base_lower]
    for prefix, canonical in _TAG_ALIAS_PREFIXES.items():
        if lower.startswith(prefix):
            return canonical
    alnum = "".join(ch for ch in lower if ch.isalnum())
    if len(alnum) >= 4:
        best = None
        best_len = 0
        for base_lower, canonical in _CANONICAL_TAGS_LOWER.items():
            base_alnum = "".join(ch for ch in base_lower if ch.isalnum())
            if not base_alnum:
                continue
            if alnum.startswith(base_alnum) or base_alnum.startswith(alnum):
                if len(base_alnum) > best_len:
                    best = canonical
                    best_len = len(base_alnum)
        if best:
            return best
    return tag_str


def _apply_strategy_tag_normalization(thesis: dict | None, raw_tag: object | None) -> tuple[dict | None, str | None]:
    """Ensure thesis carries normalized strategy_tag while preserving raw tag."""
    norm = _normalize_strategy_tag(raw_tag)
    if not norm and thesis is None:
        return thesis, None
    if thesis is None or not isinstance(thesis, dict):
        thesis = {}
    if raw_tag and norm and str(raw_tag) != norm:
        thesis.setdefault("strategy_tag_raw", str(raw_tag))
    if norm:
        thesis["strategy_tag"] = norm
    return thesis, norm


def _configure_sqlite(con: sqlite3.Connection) -> sqlite3.Connection:
    """Apply SQLite PRAGMAs to reduce lock contention."""
    try:
        con.execute(f"PRAGMA journal_mode={_DB_JOURNAL_MODE}")
    except sqlite3.Error:
        pass
    try:
        con.execute(f"PRAGMA busy_timeout={_DB_BUSY_TIMEOUT_MS}")
    except sqlite3.Error:
        pass
    try:
        con.execute(f"PRAGMA synchronous={_DB_SYNCHRONOUS}")
    except sqlite3.Error:
        pass
    try:
        con.execute(f"PRAGMA temp_store={_DB_TEMP_STORE}")
    except sqlite3.Error:
        pass
    return con


def _open_trades_db() -> sqlite3.Connection:
    con = sqlite3.connect(_DB, timeout=_DB_BUSY_TIMEOUT_MS / 1000)
    con.row_factory = sqlite3.Row
    return _configure_sqlite(con)


@contextmanager
def _file_lock(path: pathlib.Path, timeout: float = _DB_FILE_LOCK_TIMEOUT):
    """
    Inter-process advisory lock to serialize schema/migration/writes.

    Prevents multiple worker processes from ALTER/INSERT at the same time,
    which is the main source of sqlite 'database is locked' at boot.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as fh:
        start = time.monotonic()
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f"Timed out acquiring file lock: {path}")
                time.sleep(0.1)
        try:
            yield
        finally:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _build_http_session() -> requests.Session:
    total = int(os.getenv("POSITION_MANAGER_HTTP_RETRY_TOTAL", "3"))
    backoff = float(os.getenv("POSITION_MANAGER_HTTP_BACKOFF", "0.6"))
    retry = Retry(
        total=total,
        status_forcelist=_RETRY_STATUS_CODES,
        allowed_methods=frozenset({"GET", "POST"}),
        backoff_factor=backoff,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=4,
        pool_maxsize=8,
    )
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _ensure_orders_db() -> sqlite3.Connection:
    """orders.db が存在しない/テーブル欠損時に安全に初期化する。"""
    _ORDERS_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_ORDERS_DB, timeout=_DB_BUSY_TIMEOUT_MS / 1000)
    con.row_factory = sqlite3.Row
    _configure_sqlite(con)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT,
          pocket TEXT,
          instrument TEXT,
          side TEXT,
          units INTEGER,
          sl_price REAL,
          tp_price REAL,
          client_order_id TEXT,
          status TEXT,
          attempt INTEGER,
          stage_index INTEGER,
          ticket_id TEXT,
          executed_price REAL,
          error_code TEXT,
          error_message TEXT,
          request_json TEXT,
          response_json TEXT
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_orders_client ON orders(client_order_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(ts)")
    con.commit()
    return con


def _tx_sort_key(tx: dict) -> int:
    try:
        return int(tx.get("id") or 0)
    except (TypeError, ValueError):
        return 0


def _parse_timestamp(ts: str) -> datetime:
    """OANDAのISO文字列（ナノ秒精度）をPythonのdatetimeに変換する。"""
    if not ts:
        return datetime.now(timezone.utc)

    ts = ts.replace("Z", "+00:00")
    if "." not in ts:
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    head, tail = ts.split(".", 1)
    tz = ""
    if "+" in tail:
        frac, tz = tail.split("+", 1)
        tz = "+" + tz
    elif "-" in tail[6:]:
        frac, tz = tail.split("-", 1)
        tz = "-" + tz
    else:
        frac, tz = tail, ""

    frac = frac[:6].ljust(6, "0")
    dt = datetime.fromisoformat(f"{head}.{frac}{tz}")
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class PositionManager:
    def __init__(self):
        self.con = _open_trades_db()
        with _file_lock(_DB_LOCK_PATH):
            self._ensure_schema_with_retry()
        self._last_tx_id = self._get_last_transaction_id_with_retry()
        self._pocket_cache: dict[str, str] = {}
        self._client_cache: dict[str, str] = {}
        self._http = _build_http_session()
        self._last_positions: dict[str, dict] = {}
        self._last_positions_ts: float = 0.0
        self._open_trade_failures: int = 0
        self._next_open_fetch_after: float = 0.0
        self._last_positions_meta: dict | None = None
        self._entry_meta_cache: dict[str, dict] = {}

    @staticmethod
    def _infer_strategy_tag(thesis: dict | None, client_id: str | None, pocket: str | None) -> str | None:
        """
        推定ルール:
        - thesis.strategy_tag / strategy を最優先
        - client_id から既知プレフィックスや qr-<ts>-<pocket>-<tag> / qr-<pocket>-<ts>-<tag> を抽出
        """
        if thesis and isinstance(thesis, dict):
            tag = thesis.get("strategy_tag") or thesis.get("strategy")
            if tag:
                return _normalize_strategy_tag(tag)
        if not client_id:
            return None
        cid = str(client_id)
        try:
            import re
            # Known prefixes
            if cid.startswith("qr-fast-"):
                return _normalize_strategy_tag("fast_scalp")
            if cid.startswith("qr-pullback-s5-"):
                return _normalize_strategy_tag("pullback_s5")
            if cid.startswith("qr-pullback-"):
                return _normalize_strategy_tag("pullback_scalp")
            if cid.startswith("qr-mirror-s5-"):
                return _normalize_strategy_tag("mirror_spike_s5")
            # qr-<ts>-<pocket>-<tag...>
            m = re.match(r"^qr-\d+-(micro|macro|scalp|event|hybrid)-(.+)$", cid)
            if m:
                return _normalize_strategy_tag(m.group(2))
            # qr-<pocket>-<ts>-<tag...>
            m2 = re.match(r"^qr-(micro|macro|scalp|event|hybrid)-\d+-([^-]+)", cid)
            if m2:
                return _normalize_strategy_tag(m2.group(2))
            # fallback: qr-<word>-<rest>
            m3 = re.match(r"^qr-([a-zA-Z0-9_]+)-(.*)$", cid)
            if m3 and m3.group(1) not in {"micro", "macro", "scalp", "event", "hybrid"}:
                return _normalize_strategy_tag(m3.group(1))
        except Exception:
            return None
        return None

    @staticmethod
    def _normalize_pocket(pocket: str | None) -> str:
        """Map pocket values to a canonical set.

        Business rule (2025-10): any "unknown" or empty pocket is treated as
        manual. Agent-managed pockets remain unchanged.
        """
        allowed = {"macro", "micro", "scalp", "scalp_fast", "manual"}
        if not pocket:
            return "manual"
        p = str(pocket).strip().lower()
        if p in ("", "unknown"):
            return "manual"
        return p if p in allowed else "manual"

    def _ensure_schema_with_retry(self):
        for attempt in range(_DB_LOCK_RETRY):
            try:
                self._ensure_schema()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                time.sleep(_DB_LOCK_RETRY_SLEEP)
        # 最後のリトライで例外を伝播
        self._ensure_schema()

    def _ensure_schema(self):
        # trades テーブルが存在しない場合のベース定義
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              transaction_id INTEGER,
              ticket_id TEXT,
              pocket TEXT,
              instrument TEXT,
              units INTEGER,
              closed_units INTEGER,
              entry_price REAL,
              close_price REAL,
              fill_price REAL,
              pl_pips REAL,
              realized_pl REAL,
              commission REAL,
              financing REAL,
              entry_time TEXT,
              open_time TEXT,
              close_time TEXT,
              close_reason TEXT,
              state TEXT,
              updated_at TEXT,
              version TEXT DEFAULT 'v1',
              unrealized_pl REAL
            )
            """
        )

        # 欠損カラムを追加（既存データを保持する）
        existing = {row[1] for row in self.con.execute("PRAGMA table_info(trades)")}
        columns: dict[str, str] = {
            "transaction_id": "INTEGER",
            "ticket_id": "TEXT",
            "pocket": "TEXT",
            "instrument": "TEXT",
            "units": "INTEGER",
            "closed_units": "INTEGER",
            "entry_price": "REAL",
            "close_price": "REAL",
            "fill_price": "REAL",
            "pl_pips": "REAL",
            "realized_pl": "REAL",
            "commission": "REAL",
            "financing": "REAL",
            "entry_time": "TEXT",
            "open_time": "TEXT",
            "close_time": "TEXT",
            "close_reason": "TEXT",
            "state": "TEXT",
            "updated_at": "TEXT",
            "version": "TEXT DEFAULT 'v1'",
            "unrealized_pl": "REAL",
            # strategy attribution
            "strategy": "TEXT",
            "client_order_id": "TEXT",
            "strategy_tag": "TEXT",
            "entry_thesis": "TEXT",
        }
        for name, ddl in columns.items():
            if name not in existing:
                self.con.execute(f"ALTER TABLE trades ADD COLUMN {name} {ddl}")
        # 既存データで strategy が空の場合は strategy_tag をコピー
        try:
            self.con.execute(
                "UPDATE trades SET strategy = strategy_tag WHERE (strategy IS NULL OR strategy = '') AND strategy_tag IS NOT NULL"
            )
        except sqlite3.Error:
            pass

        if self._has_ticket_unique_constraint():
            self._migrate_remove_ticket_unique()

        # 旧ユニークインデックス（ticket_id 単独）は部分決済を上書きしてしまうため削除し、
        # (transaction_id, ticket_id) の複合ユニークへ移行、ticket_id は非ユニーク索引に変更
        try:
            self.con.execute("DROP INDEX IF EXISTS idx_trades_ticket")
        except Exception:
            pass
        self.con.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uniq_trades_tx_trade ON trades(transaction_id, ticket_id)"
        )
        self.con.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_ticket ON trades(ticket_id)"
        )
        self.con.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_close_time ON trades(close_time)"
        )
        self._commit_with_retry()

    def _get_last_transaction_id_with_retry(self) -> int:
        for attempt in range(_DB_LOCK_RETRY):
            try:
                return self._get_last_transaction_id()
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                time.sleep(_DB_LOCK_RETRY_SLEEP)
        return self._get_last_transaction_id()

    def _get_last_transaction_id(self) -> int:
        """DBに記録済みの最新トランザクションIDを取得"""
        # 旧スキーマ互換のため、transaction_id 優先で取得する
        cursor = self.con.cursor()
        try:
            row = cursor.execute("SELECT MAX(transaction_id) FROM trades").fetchone()
            if row and row[0]:
                return int(row[0])
        except sqlite3.OperationalError:
            pass

        row = cursor.execute("SELECT MAX(id) FROM trades").fetchone()
        return int(row[0]) if row and row[0] else 0

    def _has_ticket_unique_constraint(self) -> bool:
        try:
            indexes = list(self.con.execute("PRAGMA index_list(trades)"))
        except sqlite3.Error:
            return False
        for idx in indexes:
            name = idx[1]
            unique = idx[2]
            if not unique:
                continue
            if name.startswith("sqlite_autoindex_trades"):
                try:
                    cols = list(self.con.execute(f"PRAGMA index_info('{name}')"))
                except sqlite3.Error:
                    continue
                if len(cols) == 1 and cols[0][2] == "ticket_id":
                    return True
        return False

    def _migrate_remove_ticket_unique(self) -> None:
        try:
            self.con.execute("BEGIN")
            self.con.execute(
                """
                CREATE TABLE IF NOT EXISTS trades_migrated (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  transaction_id INTEGER,
                  ticket_id TEXT,
                  pocket TEXT,
                  instrument TEXT,
                  units INTEGER,
                  closed_units INTEGER,
                  entry_price REAL,
                  close_price REAL,
                  fill_price REAL,
                  pl_pips REAL,
                  realized_pl REAL,
                  commission REAL,
                  financing REAL,
                  entry_time TEXT,
                  open_time TEXT,
                  close_time TEXT,
                  close_reason TEXT,
                  state TEXT,
                  updated_at TEXT,
                  version TEXT DEFAULT 'v1',
                  unrealized_pl REAL
                )
                """
            )
            columns = (
                "id, transaction_id, ticket_id, pocket, instrument, units, closed_units, "
                "entry_price, close_price, fill_price, pl_pips, realized_pl, commission, "
                "financing, entry_time, open_time, close_time, close_reason, state, "
                "updated_at, version, unrealized_pl"
            )
            self.con.execute(
                f"INSERT OR IGNORE INTO trades_migrated ({columns}) "
                f"SELECT {columns} FROM trades"
            )
            self.con.execute("DROP TABLE trades")
            self.con.execute("ALTER TABLE trades_migrated RENAME TO trades")
            self.con.commit()
        except sqlite3.Error as exc:
            self.con.rollback()
            print(f"[PositionManager] Failed to migrate trades table: {exc}")
        finally:
            self._commit_with_retry()

    def _fetch_closed_trades(self):
        """OANDAから決済済みトランザクションを取得"""
        summary_url = f"{REST_HOST}/v3/accounts/{ACCOUNT}/transactions"
        try:
            summary = self._request_json(
                summary_url,
                params={"sinceID": self._last_tx_id},
            ) or {}
        except requests.RequestException as e:
            print(f"[PositionManager] Error fetching transactions summary: {e}")
            self._reset_http_if_needed(e)
            return []

        try:
            last_tx_id = int(summary.get("lastTransactionID") or 0)
        except (TypeError, ValueError):
            last_tx_id = 0

        if last_tx_id <= self._last_tx_id:
            return []

        fetch_from = self._last_tx_id + 1
        min_allowed = max(1, last_tx_id - _MAX_FETCH + 1)
        if fetch_from < min_allowed:
            fetch_from = min_allowed

        transactions: list[dict] = []
        chunk_from = fetch_from
        chunk_url = f"{REST_HOST}/v3/accounts/{ACCOUNT}/transactions/idrange"

        while chunk_from <= last_tx_id:
            chunk_to = min(chunk_from + _CHUNK_SIZE - 1, last_tx_id)
            params = {"from": chunk_from, "to": chunk_to}
            try:
                data = self._request_json(chunk_url, params=params) or {}
            except requests.RequestException as e:
                print(
                    "[PositionManager] Error fetching transaction chunk "
                    f"{chunk_from}-{chunk_to}: {e}"
                )
                self._reset_http_if_needed(e)
                break
            for tx in data.get("transactions") or []:
                try:
                    tx_id = int(tx.get("id"))
                except (TypeError, ValueError):
                    continue
                if tx_id <= self._last_tx_id:
                    continue
                transactions.append(tx)

            chunk_from = chunk_to + 1

        transactions.sort(key=_tx_sort_key)
        return transactions

    def _get_trade_details(self, trade_id: str) -> dict | None:
        """tradeIDを使ってOANDAから取引詳細を取得する"""
        url = f"{REST_HOST}/v3/accounts/{ACCOUNT}/trades/{trade_id}"
        try:
            payload = self._request_json(url) or {}
            trade = payload.get("trade", {})
            client_ext = trade.get("clientExtensions", {}) or {}
            pocket_tag = client_ext.get("tag", "pocket=unknown")
            pocket = pocket_tag.split("=")[1] if "=" in pocket_tag else "unknown"
            client_id = client_ext.get("id")

            details = {
                "entry_price": float(trade.get("price", 0.0)),
                "entry_time": _parse_timestamp(trade.get("openTime")),
                "units": int(trade.get("initialUnits", 0)),
                "pocket": pocket,
                "client_order_id": client_id,
                "strategy_tag": None,
                "entry_thesis": None,
            }
            # Heuristic mapping from client id prefix
            try:
                if client_id:
                    strat = self._infer_strategy_tag(None, client_id, pocket)
                    if strat:
                        details["strategy_tag"] = strat
            except Exception:
                pass
            # Augment with local orders log for strategy_tag/thesis if possible
            try:
                from_orders = self._get_trade_details_from_orders(trade_id)
                if from_orders:
                    for key in ("client_order_id", "strategy_tag", "entry_thesis"):
                        if from_orders.get(key) and not details.get(key):
                            details[key] = from_orders.get(key)
            except Exception:
                pass
            thesis_obj, norm_tag = _apply_strategy_tag_normalization(
                details.get("entry_thesis"),
                details.get("strategy_tag"),
            )
            if thesis_obj is not None:
                details["entry_thesis"] = thesis_obj
            if norm_tag:
                details["strategy_tag"] = norm_tag
            return details
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                fallback = self._get_trade_details_from_orders(trade_id)
                if fallback:
                    return fallback
            print(f"[PositionManager] Error fetching trade details for {trade_id}: {exc}")
            return self._get_trade_details_from_orders(trade_id)
        except requests.RequestException as e:
            print(f"[PositionManager] Error fetching trade details for {trade_id}: {e}")
            self._reset_http_if_needed(e)
            return self._get_trade_details_from_orders(trade_id)

    def _get_trade_details_from_orders(self, trade_id: str) -> dict | None:
        try:
            con = _ensure_orders_db()
            con.row_factory = sqlite3.Row
            row = con.execute(
                """
                SELECT pocket, instrument, units, executed_price, ts, client_order_id
                FROM orders
                WHERE ticket_id = ?
                  AND status = 'filled'
                ORDER BY id ASC
                LIMIT 1
                """,
                (trade_id,),
            ).fetchone()
            con.close()
        except sqlite3.Error as exc:
            print(f"[PositionManager] orders.db lookup failed for trade {trade_id}: {exc}")
            return None
        if not row:
            return None
        ts = row["ts"]
        try:
            entry_time = datetime.fromisoformat(ts)
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)
            else:
                entry_time = entry_time.astimezone(timezone.utc)
        except Exception:
            entry_time = datetime.now(timezone.utc)
        client_id = row["client_order_id"]
        strategy_tag = None
        thesis_obj = None
        if client_id:
            try:
                con2 = _ensure_orders_db()
                con2.row_factory = sqlite3.Row
                att = con2.execute(
                    """
                    SELECT request_json FROM orders
                    WHERE client_order_id = ? AND status = 'submit_attempt'
                    ORDER BY id ASC
                    LIMIT 1
                    """,
                    (client_id,),
                ).fetchone()
                con2.close()
                if att and att["request_json"]:
                    try:
                        payload = json.loads(att["request_json"]) or {}
                        thesis_obj = payload.get("entry_thesis") or {}
                        if isinstance(thesis_obj, dict):
                            strategy_tag = thesis_obj.get("strategy_tag")
                    except Exception:
                        thesis_obj = None
            except sqlite3.Error:
                pass
        if strategy_tag is None:
            strategy_tag = self._infer_strategy_tag(thesis_obj, client_id, row["pocket"])
        raw_tag = None
        if isinstance(thesis_obj, dict):
            raw_tag = thesis_obj.get("strategy_tag") or thesis_obj.get("strategy")
        if not raw_tag:
            raw_tag = strategy_tag
        thesis_obj, norm_tag = _apply_strategy_tag_normalization(thesis_obj, raw_tag)
        if norm_tag:
            strategy_tag = norm_tag
        return {
            "entry_price": float(row["executed_price"] or 0.0),
            "entry_time": entry_time,
            "units": int(row["units"] or 0),
            "pocket": row["pocket"] or "unknown",
            "client_order_id": client_id,
            "strategy_tag": strategy_tag,
            "entry_thesis": thesis_obj,
        }

    def _resolve_entry_meta(self, trade_id: str) -> dict | None:
        if not trade_id:
            return None
        cached = self._entry_meta_cache.get(trade_id)
        if cached:
            return cached
        details = self._get_trade_details_from_orders(trade_id)
        if details:
            self._entry_meta_cache[trade_id] = details
        return details

    def _parse_and_save_trades(self, transactions: list[dict]):
        """トランザクションを解析し、DBに保存"""
        trades_to_save = []
        saved_records: list[dict] = []
        processed_tx_ids = set()

        for tx in transactions:
            tx_id_raw = tx.get("id")
            try:
                tx_id = int(tx_id_raw)
            except (TypeError, ValueError):
                continue
            # ORDER_FILLのみ処理（クローズ/部分約定両対応）
            if tx.get("type") != "ORDER_FILL":
                processed_tx_ids.add(tx_id)
                continue

            closures: list[tuple[str, dict]] = []
            for closed_trade in tx.get("tradesClosed") or []:
                closures.append(("CLOSED", closed_trade))
            trade_reduced = tx.get("tradeReduced")
            if trade_reduced:
                closures.append(("PARTIAL", trade_reduced))
            for reduced in tx.get("tradesReduced") or []:
                closures.append(("PARTIAL", reduced))

            if not closures:
                processed_tx_ids.add(tx_id)
                continue

            for state_label, closed_trade in closures:
                trade_id = closed_trade.get("tradeID")
                if not trade_id:
                    continue

                details = self._get_trade_details(trade_id)
                if not details:
                    continue
                inferred_tag = self._infer_strategy_tag(
                    details.get("entry_thesis"),
                    details.get("client_order_id"),
                    details.get("pocket"),
                )
                if inferred_tag and not details.get("strategy_tag"):
                    details["strategy_tag"] = inferred_tag

                close_price = float(tx.get("price", 0.0))
                close_time = _parse_timestamp(tx.get("time"))

                # USD/JPY の pips を計算 (1 pip = 0.01 JPY)
                # OANDAのPLは通貨額なので、価格差からpipsを計算する
                entry_price = details["entry_price"]
                units = details["units"]
                # 実際にクローズされたユニット（部分決済対応）
                try:
                    closed_units_raw = closed_trade.get("units")
                    # OANDAの tradesClosed[].units は方向に応じて符号が付く場合があるため、絶対値で保存
                    closed_units = abs(int(float(closed_units_raw))) if closed_units_raw is not None else 0
                except Exception:
                    closed_units = 0
                if units > 0:  # Buy
                    pl_pips = (close_price - entry_price) * 100
                else:  # Sell
                    pl_pips = (entry_price - close_price) * 100

                realized_pl = float(closed_trade.get("realizedPL", 0.0) or 0.0)
                # 取引コスト類（存在すれば保存）
                try:
                    commission = float(tx.get("commission", 0.0) or 0.0)
                except Exception:
                    commission = 0.0
                try:
                    financing = float(tx.get("financing", 0.0) or 0.0)
                except Exception:
                    financing = 0.0
                transaction_id = int(tx.get("id", 0) or 0)
                updated_at = datetime.now(timezone.utc).isoformat()
                close_reason = tx.get("reason") or tx.get("type") or "UNKNOWN"

                record_tuple = (
                    transaction_id,
                    trade_id,
                    details["pocket"],
                    tx.get("instrument"),
                    units,
                    closed_units,
                    entry_price,
                    close_price,
                    close_price,
                    pl_pips,
                    realized_pl,
                    commission,
                    financing,
                    details["entry_time"].isoformat(),
                    details["entry_time"].isoformat(),
                    close_time.isoformat(),
                    close_reason,
                    "CLOSED" if state_label == "CLOSED" else "PARTIAL",
                    updated_at,
                    "v3",
                    0.0,
                    # attribution
                    details.get("client_order_id"),
                    details.get("strategy_tag"),
                    details.get("strategy_tag"),
                    json.dumps(details.get("entry_thesis"), ensure_ascii=False)
                )
                trades_to_save.append(record_tuple)
                saved_records.append(
                    {
                        "transaction_id": transaction_id,
                        "ticket_id": trade_id,
                        "pocket": details["pocket"],
                        "instrument": tx.get("instrument"),
                        "units": units,
                        "closed_units": closed_units,
                        "entry_price": entry_price,
                        "close_price": close_price,
                        "fill_price": close_price,
                        "pl_pips": pl_pips,
                        "realized_pl": realized_pl,
                        "commission": commission,
                        "financing": financing,
                        "entry_time": details["entry_time"].isoformat(),
                        "close_time": close_time.isoformat(),
                        "close_reason": close_reason,
                        "state": "CLOSED",
                        "updated_at": updated_at,
                        "version": "v3",
                        "unrealized_pl": 0.0,
                        "client_order_id": details.get("client_order_id"),
                        "strategy": details.get("strategy_tag"),
                        "strategy_tag": details.get("strategy_tag"),
                    }
                )
                if details["pocket"]:
                    self._pocket_cache[trade_id] = details["pocket"]

                processed_tx_ids.add(tx_id)

        if trades_to_save:
            # ticket_id (OANDA tradeID) が重複しないように挿入
            with _file_lock(_DB_LOCK_PATH):
                self._executemany_with_retry(
                    """
                    INSERT OR REPLACE INTO trades (
                        transaction_id,
                        ticket_id,
                        pocket,
                        instrument,
                        units,
                        closed_units,
                        entry_price,
                        close_price,
                        fill_price,
                        pl_pips,
                        realized_pl,
                        commission,
                        financing,
                        entry_time,
                        open_time,
                        close_time,
                        close_reason,
                        state,
                        updated_at,
                        version,
                        unrealized_pl,
                        client_order_id,
                        strategy,
                        strategy_tag,
                        entry_thesis
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    trades_to_save,
                )
                self._commit_with_retry()
            print(f"[PositionManager] Saved {len(trades_to_save)} new trades.")

        if processed_tx_ids:
            self._last_tx_id = max(processed_tx_ids)
        return saved_records

    def sync_trades(self):
        """定期的に呼び出し、決済済みトレードを同期する"""
        transactions = self._fetch_closed_trades()
        if not transactions:
            return []
        return self._parse_and_save_trades(transactions)

    def _request_json(self, url: str, params: dict | None = None) -> dict:
        resp = self._http.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            logging.warning("[PositionManager] Non-JSON response from %s", url)
            return {}

    def _reset_http_if_needed(self, exc: requests.RequestException) -> None:
        if isinstance(
            exc,
            (
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout,
            ),
        ):
            try:
                self._http.close()
            except Exception:
                pass
            self._http = _build_http_session()

    def close(self):
        self.con.close()

    def register_open_trade(self, trade_id: str, pocket: str, client_id: str | None = None):
        if trade_id and pocket:
            self._pocket_cache[str(trade_id)] = pocket
        if client_id and trade_id:
            self._client_cache[client_id] = trade_id

    def get_open_positions(self) -> dict[str, dict]:
        """現在の保有ポジションを pocket 単位で集計して返す"""
        now_mono = time.monotonic()
        if self._last_positions and now_mono < self._next_open_fetch_after:
            age = max(0.0, now_mono - self._last_positions_ts)
            stale = self._open_trade_failures > 0
            return self._package_positions(
                self._last_positions,
                stale=stale,
                age_sec=age,
                extra_meta=self._last_positions_meta,
            )

        url = f"{REST_HOST}/v3/accounts/{ACCOUNT}/openTrades"
        try:
            payload = self._request_json(url) or {}
            trades = payload.get("trades", [])
            self._open_trade_failures = 0
            self._next_open_fetch_after = now_mono + _OPEN_TRADES_CACHE_TTL
        except requests.RequestException as e:
            logging.warning("[PositionManager] Error fetching open trades: %s", e)
            self._reset_http_if_needed(e)
            self._open_trade_failures += 1
            backoff = min(
                _OPEN_TRADES_FAIL_BACKOFF_BASE * (2 ** (self._open_trade_failures - 1)),
                _OPEN_TRADES_FAIL_BACKOFF_MAX,
            )
            self._next_open_fetch_after = now_mono + backoff
            if self._last_positions:
                age = max(0.0, now_mono - self._last_positions_ts)
                return self._package_positions(
                    self._last_positions,
                    stale=True,
                    age_sec=age,
                    extra_meta=self._last_positions_meta,
                )
            return {}

        pockets: dict[str, dict] = {}
        net_units = 0
        manual_trades = 0
        manual_units = 0
        manual_unrealized = 0.0
        client_ids: set[str] = set()
        for tr in trades:
            client_ext = tr.get("clientExtensions", {}) or {}
            client_id_raw = client_ext.get("id")
            client_id = str(client_id_raw or "")
            tag_raw = client_ext.get("tag") or ""
            tag = str(tag_raw)
            trade_id = tr.get("id") or tr.get("tradeID")
            cached_pocket = self._pocket_cache.get(str(trade_id), "") if trade_id else ""

            pocket: str
            if client_id.startswith(agent_client_prefixes):
                if tag.startswith("pocket="):
                    pocket = tag.split("=", 1)[1]
                elif cached_pocket in agent_pockets:
                    pocket = cached_pocket
                else:
                    pocket = "unknown"
            elif cached_pocket in agent_pockets:
                pocket = cached_pocket
            elif tag.startswith("pocket="):
                candidate = tag.split("=", 1)[1]
                pocket = candidate if candidate in agent_pockets else "manual"
            else:
                trade_id = tr.get("id") or tr.get("tradeID")
                pocket = self._pocket_cache.get(str(trade_id), _MANUAL_POCKET_NAME)
            if pocket not in _KNOWN_POCKETS:
                pocket = _MANUAL_POCKET_NAME
            units = int(tr.get("currentUnits", 0))
            if units == 0:
                continue
            trade_id_raw = tr.get("id") or tr.get("tradeID")
            trade_id = str(trade_id_raw)
            price = float(tr.get("price", 0.0))
            if client_id:
                client_ids.add(client_id)
            open_time_raw = tr.get("openTime")
            open_time_iso: str | None = None
            if open_time_raw:
                try:
                    opened_dt = _parse_timestamp(open_time_raw).astimezone(timezone.utc)
                    open_time_iso = opened_dt.isoformat()
                except Exception:
                    open_time_iso = open_time_raw
            info = pockets.setdefault(
                pocket,
                {
                    "units": 0,
                    "avg_price": 0.0,
                    "trades": 0,
                    "long_units": 0,
                    "long_avg_price": 0.0,
                    "short_units": 0,
                    "short_avg_price": 0.0,
                    "open_trades": [],
                    "unrealized_pl": 0.0,
                    "unrealized_pl_pips": 0.0,
                },
            )
            if pocket == "manual":
                info["manual"] = True
            try:
                unrealized_pl = float(tr.get("unrealizedPL", 0.0) or 0.0)
            except Exception:
                unrealized_pl = 0.0
            abs_units = abs(units)
            pip_value = abs_units * 0.01
            unrealized_pl_pips = unrealized_pl / pip_value if pip_value else 0.0
            client_id = client_ext.get("id")
            trade_entry = {
                "trade_id": trade_id,
                "units": units,
                "price": price,
                "client_id": client_id,
                "client_order_id": client_id,
                "side": "long" if units > 0 else "short",
                "unrealized_pl": unrealized_pl,
                "unrealized_pl_pips": unrealized_pl_pips,
                "open_time": open_time_iso or open_time_raw,
            }
            # Fallback: decode clientExtensions.comment to recover entry meta for EXIT判定
            thesis_from_comment = None
            try:
                comment_raw = client_ext.get("comment")
                if isinstance(comment_raw, str) and comment_raw.startswith("{") and len(comment_raw) <= 256:
                    parsed = json.loads(comment_raw)
                    if isinstance(parsed, dict):
                        thesis_from_comment = parsed
            except Exception:
                thesis_from_comment = None
            meta = self._resolve_entry_meta(trade_id)
            if meta:
                thesis = meta.get("entry_thesis")
                if isinstance(thesis, str):
                    try:
                        thesis = json.loads(thesis)
                    except Exception:
                        thesis = None
                if isinstance(thesis, dict):
                    trade_entry["entry_thesis"] = thesis
                if meta.get("client_order_id"):
                    trade_entry["client_order_id"] = meta.get("client_order_id")
                strategy_tag = meta.get("strategy_tag")
                if strategy_tag:
                    trade_entry["strategy_tag"] = strategy_tag
            if thesis_from_comment:
                if not trade_entry.get("entry_thesis"):
                    trade_entry["entry_thesis"] = thesis_from_comment
                if not trade_entry.get("strategy_tag"):
                    tag_val = thesis_from_comment.get("strategy_tag") or thesis_from_comment.get("tag")
                    if tag_val:
                        trade_entry["strategy_tag"] = tag_val
            if not trade_entry.get("strategy_tag"):
                inferred = self._infer_strategy_tag(trade_entry.get("entry_thesis"), client_id, pocket)
                if inferred:
                    trade_entry["strategy_tag"] = inferred
            raw_tag = None
            thesis_obj = trade_entry.get("entry_thesis")
            if isinstance(thesis_obj, dict):
                raw_tag = thesis_obj.get("strategy_tag") or thesis_obj.get("strategy")
            if not raw_tag:
                raw_tag = trade_entry.get("strategy_tag")
            thesis_obj, norm_tag = _apply_strategy_tag_normalization(thesis_obj, raw_tag)
            if thesis_obj is not None:
                trade_entry["entry_thesis"] = thesis_obj
            if norm_tag:
                trade_entry["strategy_tag"] = norm_tag
            # EXIT誤爆防止: エージェント管理ポケットかつ strategy_tag 不明なら除外
            if pocket in agent_pockets and not trade_entry.get("strategy_tag"):
                logging.warning(
                    "[PositionManager] skip open trade without strategy_tag pocket=%s trade_id=%s client_id=%s",
                    pocket,
                    trade_id,
                    trade_entry.get("client_id"),
                )
                continue
            info["open_trades"].append(trade_entry)
            prev_total_units = info["units"]
            new_total_units = prev_total_units + units
            if new_total_units != 0:
                info["avg_price"] = (
                    info["avg_price"] * prev_total_units + price * units
                ) / new_total_units
            else:
                info["avg_price"] = price
            info["units"] = new_total_units
            info["trades"] += 1

            if units > 0:
                prev_units = info["long_units"]
                new_units = prev_units + units
                if new_units > 0:
                    if prev_units == 0 or info["long_avg_price"] == 0.0:
                        info["long_avg_price"] = price
                    else:
                        info["long_avg_price"] = (
                            info["long_avg_price"] * prev_units + price * units
                        ) / new_units
                info["long_units"] = new_units
            elif units < 0:
                abs_units = abs(units)
                prev_units = info["short_units"]
                new_units = prev_units + abs_units
                if new_units > 0:
                    if prev_units == 0 or info["short_avg_price"] == 0.0:
                        info["short_avg_price"] = price
                    else:
                        info["short_avg_price"] = (
                            info["short_avg_price"] * prev_units + price * abs_units
                        ) / new_units
                info["short_units"] = new_units

            info["unrealized_pl"] = info.get("unrealized_pl", 0.0) + unrealized_pl
            info["unrealized_pl_pips"] = info.get("unrealized_pl_pips", 0.0) + unrealized_pl_pips
            net_units += units
            if pocket == _MANUAL_POCKET_NAME:
                manual_trades += 1
                manual_units += units
                manual_unrealized += unrealized_pl

        if client_ids:
            entry_map = self._load_entry_thesis(client_ids)
            for pocket_info in pockets.values():
                trades_list = pocket_info.get("open_trades") if isinstance(pocket_info, dict) else None
                if not trades_list:
                    continue
                for trade in trades_list:
                    cid = trade.get("client_id")
                    if cid and cid in entry_map:
                        thesis_obj = entry_map[cid]
                        raw_tag = None
                        if isinstance(thesis_obj, dict):
                            raw_tag = thesis_obj.get("strategy_tag_raw") or thesis_obj.get("strategy_tag")
                            if not raw_tag:
                                raw_tag = thesis_obj.get("strategy")
                        thesis_obj, norm_tag = _apply_strategy_tag_normalization(thesis_obj, raw_tag)
                        if thesis_obj is not None:
                            trade["entry_thesis"] = thesis_obj
                        if norm_tag:
                            trade["strategy_tag"] = norm_tag

        pockets["__net__"] = {"units": net_units}
        self._last_positions = copy.deepcopy(pockets)
        self._last_positions_ts = now_mono

        extra_meta = {}
        if manual_trades:
            extra_meta = {
                "manual_trades": manual_trades,
                "manual_units": manual_units,
                "manual_unrealized_pl": manual_unrealized,
            }
        self._last_positions_meta = dict(extra_meta)
        return self._package_positions(pockets, stale=False, age_sec=0.0, extra_meta=extra_meta)

    def _commit_with_retry(self) -> None:
        """Commit with retry to survive short lock bursts."""
        for attempt in range(_DB_LOCK_RETRY):
            try:
                self.con.commit()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == _DB_LOCK_RETRY - 1:
                    raise
                time.sleep(_DB_LOCK_RETRY_SLEEP * (attempt + 1))

    def _executemany_with_retry(self, sql: str, params) -> None:
        for attempt in range(_DB_LOCK_RETRY):
            try:
                self.con.executemany(sql, params)
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == _DB_LOCK_RETRY - 1:
                    raise
                time.sleep(_DB_LOCK_RETRY_SLEEP * (attempt + 1))

    def _package_positions(
        self,
        pockets: dict[str, dict],
        *,
        stale: bool,
        age_sec: float,
        extra_meta: dict | None = None,
    ) -> dict[str, dict]:
        snapshot = copy.deepcopy(pockets)
        meta = snapshot.setdefault("__meta__", {})
        meta.update(
            {
                "stale": bool(stale),
                "age_sec": round(max(age_sec, 0.0), 2),
                "consecutive_failures": self._open_trade_failures if stale else 0,
            }
        )
        if extra_meta:
            meta.update(extra_meta)
        return snapshot

    def manual_exposure(
        self,
        positions: Dict[str, Dict] | None = None,
    ) -> Dict[str, float]:
        """
        Return a lightweight snapshot of manual/unknown exposure.
        """
        snapshot = positions or self.get_open_positions()
        units = 0
        unrealized = 0.0

        def _sum(info: Dict) -> Tuple[int, float]:
            if not info:
                return 0, 0.0
            trades = info.get("open_trades") or []
            total_units = 0
            unreal = 0.0
            if trades:
                for tr in trades:
                    try:
                        total_units += abs(int(tr.get("units", 0) or 0))
                    except (TypeError, ValueError):
                        continue
                    try:
                        unreal += float(tr.get("unrealized_pl", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        continue
                return total_units, unreal
            try:
                total_units = abs(int(info.get("units", 0) or 0))
            except (TypeError, ValueError):
                total_units = 0
            try:
                unreal = float(info.get("unrealized_pl", 0.0) or 0.0)
            except (TypeError, ValueError):
                unreal = 0.0
            return total_units, unreal

        for pocket in (_MANUAL_POCKET_NAME, "unknown"):
            info = snapshot.get(pocket) or {}
            pocket_units, pocket_unreal = _sum(info)
            units += pocket_units
            unrealized += pocket_unreal

        return {
            "units": float(units),
            "lots": units / 100000.0,
            "unrealized_pl": float(unrealized),
        }

    def _load_entry_thesis(self, client_ids: list[str]) -> Dict[str, dict]:
        unique_ids = tuple(dict.fromkeys(cid for cid in client_ids if cid))
        if not unique_ids:
            return {}
        try:
            con = _ensure_orders_db()
        except sqlite3.Error as exc:
            print(f"[PositionManager] Failed to open orders.db: {exc}")
            return {}
        placeholders = ",".join("?" for _ in unique_ids)
        try:
            rows = con.execute(
                f"""
                SELECT client_order_id, request_json
                FROM orders
                WHERE client_order_id IN ({placeholders})
                  AND status='submit_attempt'
                ORDER BY id DESC
                """,
                unique_ids,
            ).fetchall()
        except sqlite3.Error as exc:
            print(f"[PositionManager] orders.db query failed: {exc}")
            con.close()
            return {}
        con.close()
        result: Dict[str, dict] = {}
        for row in rows:
            cid = row["client_order_id"]
            if cid in result:
                continue
            payload_raw = row["request_json"]
            if not payload_raw:
                continue
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError:
                continue
            thesis = payload.get("entry_thesis") or (payload.get("meta") or {}).get("entry_thesis") or {}
            result[cid] = thesis
        return result

    def get_performance_summary(self, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=6)

        buckets = {
            "daily": {
                "pips": 0.0,
                "jpy": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
            },
            "weekly": {
                "pips": 0.0,
                "jpy": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
            },
            "total": {
                "pips": 0.0,
                "jpy": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
            },
        }
        latest_close: datetime | None = None

        rows = self.con.execute(
            "SELECT pl_pips, realized_pl, close_time, pocket FROM trades WHERE close_time IS NOT NULL"
        ).fetchall()
        pocket_raw: dict[str, dict[str, float | int]] = {}
        for row in rows:
            try:
                close_dt = _parse_timestamp(row["close_time"])
            except Exception:
                continue
            if latest_close is None or close_dt > latest_close:
                latest_close = close_dt
            pl_pips = float(row["pl_pips"] or 0.0)
            pl_jpy = float(row["realized_pl"] or 0.0)
            pocket = (row["pocket"] or "unknown").lower()
            pkt = pocket_raw.setdefault(
                pocket,
                {"pips": 0.0, "jpy": 0.0, "trades": 0, "wins": 0, "losses": 0},
            )

            def _apply(bucket: dict) -> None:
                bucket["pips"] += pl_pips
                bucket["jpy"] += pl_jpy
                bucket["trades"] += 1
                if pl_pips > 0:
                    bucket["wins"] += 1
                elif pl_pips < 0:
                    bucket["losses"] += 1

            _apply(buckets["total"])
            _apply(pkt)
            if close_dt >= week_start:
                _apply(buckets["weekly"])
            if close_dt >= today_start:
                _apply(buckets["daily"])

        def _finalise(data: dict) -> dict:
            trades = data["trades"]
            win_rate = (data["wins"] / trades) if trades else 0.0
            pf = None
            if data["losses"] > 0:
                loss_sum = abs(data["pips"] - max(0.0, data["pips"]))
                win_sum = max(0.0, data["pips"])
                if loss_sum > 0:
                    pf = win_sum / loss_sum if loss_sum else None
            return {
                "pips": round(data["pips"], 2),
                "jpy": round(data["jpy"], 2),
                "trades": trades,
                "wins": data["wins"],
                "losses": data["losses"],
                "win_rate": win_rate,
                "pf": pf,
            }

        pockets_final: dict[str, dict] = {}
        for name, raw in pocket_raw.items():
            pockets_final[name] = _finalise(raw)

        return {
            "daily": _finalise(buckets["daily"]),
            "weekly": _finalise(buckets["weekly"]),
            "total": _finalise(buckets["total"]),
            "last_trade_at": latest_close.isoformat() if latest_close else None,
            "pockets": pockets_final,
        }

    def fetch_recent_trades(self, limit: int = 50) -> list[dict]:
        """UI 表示用に最新のトレードを取得"""
        cursor = self.con.execute(
            """
            SELECT ticket_id, pocket, instrument, units, closed_units, entry_price, close_price,
                   fill_price, pl_pips, realized_pl, commission, financing,
                   entry_time, close_time, close_reason,
                   state, updated_at
            FROM trades
            ORDER BY datetime(updated_at) DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [
            {
                "ticket_id": row["ticket_id"],
                "pocket": row["pocket"],
                "instrument": row["instrument"],
                "units": row["units"],
                "closed_units": row["closed_units"],
                "entry_price": row["entry_price"],
                "close_price": row["close_price"],
                "fill_price": row["fill_price"],
                "pl_pips": row["pl_pips"],
                "realized_pl": row["realized_pl"],
                "commission": row["commission"],
                "financing": row["financing"],
                "entry_time": row["entry_time"],
                "close_time": row["close_time"],
                "close_reason": row["close_reason"],
                "state": row["state"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
