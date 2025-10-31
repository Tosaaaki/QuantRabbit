from __future__ import annotations
import json
import os
import requests
import sqlite3
import pathlib
from datetime import datetime, timezone, timedelta
from typing import Dict
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
        self.con = sqlite3.connect(_DB)
        self.con.row_factory = sqlite3.Row
        self._ensure_schema()
        self._last_tx_id = self._get_last_transaction_id()
        self._pocket_cache: dict[str, str] = {}
        self._client_cache: dict[str, str] = {}

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
        }
        for name, ddl in columns.items():
            if name not in existing:
                self.con.execute(f"ALTER TABLE trades ADD COLUMN {name} {ddl}")

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
        self.con.commit()

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

    def _fetch_closed_trades(self):
        """OANDAから決済済みトランザクションを取得"""
        summary_url = f"{REST_HOST}/v3/accounts/{ACCOUNT}/transactions"
        try:
            summary_resp = requests.get(
                summary_url,
                headers=HEADERS,
                params={"sinceID": self._last_tx_id},
                timeout=10,
            )
            summary_resp.raise_for_status()
            summary = summary_resp.json() or {}
        except requests.RequestException as e:
            print(f"[PositionManager] Error fetching transactions summary: {e}")
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
                resp = requests.get(
                    chunk_url,
                    headers=HEADERS,
                    params=params,
                    timeout=10,
                )
                resp.raise_for_status()
            except requests.RequestException as e:
                print(
                    "[PositionManager] Error fetching transaction chunk "
                    f"{chunk_from}-{chunk_to}: {e}"
                )
                break

            data = resp.json() or {}
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
            r = requests.get(url, headers=HEADERS, timeout=5)
            r.raise_for_status()
            trade = r.json().get("trade", {})

            pocket_tag = trade.get("clientExtensions", {}).get("tag", "pocket=unknown")
            pocket = pocket_tag.split("=")[1] if "=" in pocket_tag else "unknown"

            return {
                "entry_price": float(trade.get("price", 0.0)),
                "entry_time": _parse_timestamp(trade.get("openTime")),
                "units": int(trade.get("initialUnits", 0)),
                "pocket": pocket,
            }
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                fallback = self._get_trade_details_from_orders(trade_id)
                if fallback:
                    return fallback
            print(f"[PositionManager] Error fetching trade details for {trade_id}: {exc}")
            return self._get_trade_details_from_orders(trade_id)
        except requests.RequestException as e:
            print(f"[PositionManager] Error fetching trade details for {trade_id}: {e}")
            return self._get_trade_details_from_orders(trade_id)

    def _get_trade_details_from_orders(self, trade_id: str) -> dict | None:
        if not _ORDERS_DB.exists():
            return None
        try:
            con = sqlite3.connect(_ORDERS_DB)
            con.row_factory = sqlite3.Row
            row = con.execute(
                """
                SELECT pocket, instrument, units, executed_price, ts
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
        return {
            "entry_price": float(row["executed_price"] or 0.0),
            "entry_time": entry_time,
            "units": int(row["units"] or 0),
            "pocket": row["pocket"] or "unknown",
        }

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
                    }
                )
                if details["pocket"]:
                    self._pocket_cache[trade_id] = details["pocket"]

                processed_tx_ids.add(tx_id)

        if trades_to_save:
            # ticket_id (OANDA tradeID) が重複しないように挿入
            self.con.executemany(
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
                    unrealized_pl
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                trades_to_save,
            )
            self.con.commit()
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

    def close(self):
        self.con.close()

    def register_open_trade(self, trade_id: str, pocket: str, client_id: str | None = None):
        if trade_id and pocket:
            self._pocket_cache[str(trade_id)] = pocket
        if client_id and trade_id:
            self._client_cache[client_id] = trade_id

    def get_open_positions(self) -> dict[str, dict]:
        """現在の保有ポジションを pocket 単位で集計して返す"""
        url = f"{REST_HOST}/v3/accounts/{ACCOUNT}/openTrades"
        try:
            r = requests.get(url, headers=HEADERS, timeout=5)
            r.raise_for_status()
            trades = r.json().get("trades", [])
        except requests.RequestException as e:
            print(f"[PositionManager] Error fetching open trades: {e}")
            return {}

        pockets: dict[str, dict] = {}
        net_units = 0
        client_ids: list[str] = []
        for tr in trades:
            client_ext = tr.get("clientExtensions", {}) or {}
            tag = client_ext.get("tag", "pocket=unknown")
            if "=" in tag:
                pocket = tag.split("=", 1)[1]
            else:
                trade_id = tr.get("id") or tr.get("tradeID")
                pocket = self._pocket_cache.get(str(trade_id), "unknown")
            units = int(tr.get("currentUnits", 0))
            if units == 0:
                continue
            trade_id = tr.get("id") or tr.get("tradeID")
            price = float(tr.get("price", 0.0))
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
            try:
                unrealized_pl = float(tr.get("unrealizedPL", 0.0) or 0.0)
            except Exception:
                unrealized_pl = 0.0
            abs_units = abs(units)
            pip_value = abs_units * 0.01
            unrealized_pl_pips = unrealized_pl / pip_value if pip_value else 0.0
            info["open_trades"].append(
                {
                    "trade_id": str(trade_id),
                    "units": units,
                    "price": price,
                    "client_id": client_ext.get("id"),
                    "side": "long" if units > 0 else "short",
                    "unrealized_pl": unrealized_pl,
                    "unrealized_pl_pips": unrealized_pl_pips,
                    "open_time": open_time_iso or open_time_raw,
                }
            )
            if client_ext.get("id"):
                client_ids.append(client_ext.get("id"))
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

        if client_ids:
            entry_map = self._load_entry_thesis(client_ids)
            for pocket_info in pockets.values():
                trades_list = pocket_info.get("open_trades") if isinstance(pocket_info, dict) else None
                if not trades_list:
                    continue
                for trade in trades_list:
                    cid = trade.get("client_id")
                    if cid and cid in entry_map:
                        trade["entry_thesis"] = entry_map[cid]

        pockets["__net__"] = {"units": net_units}

        return pockets

    def _load_entry_thesis(self, client_ids: list[str]) -> Dict[str, dict]:
        unique_ids = tuple(dict.fromkeys(cid for cid in client_ids if cid))
        if not unique_ids:
            return {}
        try:
            con = sqlite3.connect(_ORDERS_DB)
            con.row_factory = sqlite3.Row
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
            thesis = (payload.get("meta") or {}).get("entry_thesis") or {}
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
            "SELECT pl_pips, realized_pl, close_time FROM trades WHERE close_time IS NOT NULL"
        ).fetchall()
        for row in rows:
            try:
                close_dt = _parse_timestamp(row["close_time"])
            except Exception:
                continue
            if latest_close is None or close_dt > latest_close:
                latest_close = close_dt
            pl_pips = float(row["pl_pips"] or 0.0)
            pl_jpy = float(row["realized_pl"] or 0.0)

            def _apply(bucket: dict) -> None:
                bucket["pips"] += pl_pips
                bucket["jpy"] += pl_jpy
                bucket["trades"] += 1
                if pl_pips > 0:
                    bucket["wins"] += 1
                elif pl_pips < 0:
                    bucket["losses"] += 1

            _apply(buckets["total"])
            if close_dt >= week_start:
                _apply(buckets["weekly"])
            if close_dt >= today_start:
                _apply(buckets["daily"])

        def _finalise(data: dict) -> dict:
            trades = data["trades"]
            win_rate = (data["wins"] / trades) if trades else 0.0
            return {
                "pips": round(data["pips"], 2),
                "jpy": round(data["jpy"], 2),
                "trades": trades,
                "wins": data["wins"],
                "losses": data["losses"],
                "win_rate": win_rate,
            }

        return {
            "daily": _finalise(buckets["daily"]),
            "weekly": _finalise(buckets["weekly"]),
            "total": _finalise(buckets["total"]),
            "last_trade_at": latest_close.isoformat() if latest_close else None,
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
