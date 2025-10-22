from __future__ import annotations
import requests
import sqlite3
import pathlib
from datetime import datetime, timezone
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


class PositionManager:
    def __init__(self):
        self.con = sqlite3.connect(_DB)
        self.con.row_factory = sqlite3.Row
        self._ensure_schema()
        self._last_tx_id = self._get_last_transaction_id()
        self._pocket_cache: dict[str, str] = {}

    def _ensure_schema(self):
        # trades テーブルが存在しない場合のベース定義
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              transaction_id INTEGER,
              ticket_id TEXT UNIQUE,
              pocket TEXT,
              instrument TEXT,
              units INTEGER,
              entry_price REAL,
              close_price REAL,
              pl_pips REAL,
              realized_pl REAL,
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
            "entry_price": "REAL",
            "close_price": "REAL",
            "pl_pips": "REAL",
            "realized_pl": "REAL",
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

        self.con.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_ticket ON trades(ticket_id)"
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

    def _fetch_closed_trades(self):
        """OANDAから決済済みトランザクションを取得"""
        url = f"{REST_HOST}/v3/accounts/{ACCOUNT}/transactions"
        params = {"sinceID": self._last_tx_id, "type": "ORDER_FILL"}  # 約定した注文のみ
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=10)
            r.raise_for_status()
            return r.json().get("transactions", [])
        except requests.RequestException as e:
            print(f"[PositionManager] Error fetching transactions: {e}")
            return []

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
                "entry_time": datetime.fromisoformat(
                    trade.get("openTime").replace("Z", "+00:00")
                ),
                "units": int(trade.get("initialUnits", 0)),
                "pocket": pocket,
            }
        except requests.RequestException as e:
            print(f"[PositionManager] Error fetching trade details for {trade_id}: {e}")
            return None

    def _parse_and_save_trades(self, transactions: list[dict]):
        """トランザクションを解析し、DBに保存"""
        trades_to_save = []
        saved_records: list[dict] = []
        processed_tx_ids = set()

        for tx in transactions:
            # ORDER_FILLかつ、ポジションを閉じる取引のみ対象
            if tx.get("type") != "ORDER_FILL" or "tradesClosed" not in tx:
                processed_tx_ids.add(tx["id"])
                continue

            for closed_trade in tx.get("tradesClosed", []):
                trade_id = closed_trade.get("tradeID")
                if not trade_id:
                    continue

                details = self._get_trade_details(trade_id)
                if not details:
                    continue

                close_price = float(tx.get("price", 0.0))
                close_time = datetime.fromisoformat(
                    tx.get("time").replace("Z", "+00:00")
                )

                # USD/JPY の pips を計算 (1 pip = 0.01 JPY)
                # OANDAのPLは通貨額なので、価格差からpipsを計算する
                entry_price = details["entry_price"]
                units = details["units"]
                if units > 0:  # Buy
                    pl_pips = (close_price - entry_price) * 100
                else:  # Sell
                    pl_pips = (entry_price - close_price) * 100

                realized_pl = float(closed_trade.get("realizedPL", 0.0) or 0.0)
                transaction_id = int(tx.get("id", 0) or 0)
                updated_at = datetime.now(timezone.utc).isoformat()
                close_reason = tx.get("reason") or tx.get("type") or "UNKNOWN"

                record_tuple = (
                    transaction_id,
                    trade_id,
                    details["pocket"],
                    tx.get("instrument"),
                    units,
                    entry_price,
                    close_price,
                    pl_pips,
                    realized_pl,
                    details["entry_time"].isoformat(),
                    details["entry_time"].isoformat(),
                    close_time.isoformat(),
                    close_reason,
                    "CLOSED",
                    updated_at,
                    "v2",
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
                        "entry_price": entry_price,
                        "close_price": close_price,
                        "pl_pips": pl_pips,
                        "realized_pl": realized_pl,
                        "entry_time": details["entry_time"].isoformat(),
                        "close_time": close_time.isoformat(),
                        "close_reason": close_reason,
                        "state": "CLOSED",
                        "updated_at": updated_at,
                        "version": "v2",
                        "unrealized_pl": 0.0,
                    }
                )
                if details["pocket"]:
                    self._pocket_cache[trade_id] = details["pocket"]

            processed_tx_ids.add(int(tx["id"]))

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
                    entry_price,
                    close_price,
                    pl_pips,
                    realized_pl,
                    entry_time,
                    open_time,
                    close_time,
                    close_reason,
                    state,
                    updated_at,
                    version,
                    unrealized_pl
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def register_open_trade(self, trade_id: str, pocket: str):
        if trade_id and pocket:
            self._pocket_cache[str(trade_id)] = pocket

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
        for tr in trades:
            tag = tr.get("clientExtensions", {}).get("tag", "pocket=unknown")
            if "=" in tag:
                pocket = tag.split("=", 1)[1]
            else:
                trade_id = tr.get("id") or tr.get("tradeID")
                pocket = self._pocket_cache.get(str(trade_id), "unknown")
            units = int(tr.get("currentUnits", 0))
            if units == 0:
                continue
            price = float(tr.get("price", 0.0))
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
                },
            )
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

            net_units += units

        pockets["__net__"] = {"units": net_units}

        return pockets

    def fetch_recent_trades(self, limit: int = 50) -> list[dict]:
        """UI 表示用に最新のトレードを取得"""
        cursor = self.con.execute(
            """
            SELECT ticket_id, pocket, instrument, units, entry_price, close_price,
                   pl_pips, realized_pl, entry_time, close_time, close_reason,
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
                "entry_price": row["entry_price"],
                "close_price": row["close_price"],
                "pl_pips": row["pl_pips"],
                "realized_pl": row["realized_pl"],
                "entry_time": row["entry_time"],
                "close_time": row["close_time"],
                "close_reason": row["close_reason"],
                "state": row["state"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
