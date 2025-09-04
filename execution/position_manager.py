from __future__ import annotations
import requests
import sqlite3
import pathlib
from datetime import datetime
from utils.secrets import get_secret
from analysis.learning import record_trade_performance

# --- config ---
TOKEN = get_secret("oanda_token")
ACCOUNT = get_secret("oanda_account_id")
try:
    PRACT = get_secret("oanda_practice").lower() == "true"
except Exception:
    PRACT = True  # デフォルトは practice を安全側に
REST_HOST = (
    "https://api-fxpractice.oanda.com" if PRACT else "https://api-fxtrade.oanda.com"
)
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

_DB = pathlib.Path("logs/trades.db")
_DB.parent.mkdir(exist_ok=True)


class PositionManager:
    def __init__(self):
        self.con = sqlite3.connect(_DB)
        # trades テーブルを用意（他モジュールと統一）+ 既存スキーマを自動マイグレーション
        self._ensure_schema()
        self._last_tx_id = self._get_last_transaction_id()

    def _ensure_schema(self) -> None:
        cur = self.con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
              id INTEGER PRIMARY KEY,
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
              micro_regime TEXT
            )
            """
        )
        # 既存テーブルに不足カラムがあれば追加（古い簡易スキーマ互換）
        cur.execute("PRAGMA table_info(trades)")
        cols = {row[1] for row in cur.fetchall()}
        desired = [
            ("ticket_id", "TEXT"),
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
        ]
        for name, typ in desired:
            if name not in cols:
                cur.execute(f"ALTER TABLE trades ADD COLUMN {name} {typ}")
        self.con.commit()

    def _get_last_transaction_id(self) -> int:
        """DBに記録済みの最新トランザクションIDを取得"""
        row = self.con.execute("SELECT MAX(id) FROM trades").fetchone()
        # OANDAのTransactionIDは1から始まるので、DBが空なら0を返す
        return row[0] if row and row[0] else 0

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

            ext = trade.get("clientExtensions", {}) or {}
            pocket_tag = ext.get("tag", "pocket=unknown")
            pocket = pocket_tag.split("=")[1] if "=" in pocket_tag else "unknown"
            comment = ext.get("comment") or ""
            # comment format: strategy=NAME|macro=Trend|micro=Range
            meta = {}
            for part in comment.split("|"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    meta[k.strip()] = v.strip()

            return {
                "entry_price": float(trade.get("price", 0.0)),
                "entry_time": datetime.fromisoformat(
                    trade.get("openTime").replace("Z", "+00:00")
                ),
                "units": int(trade.get("initialUnits", 0)),
                "pocket": pocket,
                "strategy": meta.get("strategy"),
                "macro_regime": meta.get("macro"),
                "micro_regime": meta.get("micro"),
            }
        except requests.RequestException as e:
            print(f"[PositionManager] Error fetching trade details for {trade_id}: {e}")
            return None

    def _parse_and_save_trades(self, transactions: list[dict]):
        """トランザクションを解析し、DBに保存"""
        trades_to_save = []
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

                trades_to_save.append(
                    (
                        tx["id"],
                        trade_id,
                        details["pocket"],
                        tx.get("instrument"),
                        units,
                        entry_price,
                        close_price,
                        pl_pips,
                        details["entry_time"].isoformat(),
                        close_time.isoformat(),
                        details.get("strategy"),
                        details.get("macro_regime"),
                        details.get("micro_regime"),
                    )
                )

            processed_tx_ids.add(tx["id"])

        if trades_to_save:
            # ticket_id (OANDA tradeID) が重複しないように挿入
            self.con.executemany(
                """
                INSERT OR IGNORE INTO trades (
                  id, ticket_id, pocket, instrument, units, entry_price, close_price, pl_pips, entry_time, close_time, strategy, macro_regime, micro_regime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                trades_to_save,
            )
            self.con.commit()
            print(f"[PositionManager] Saved {len(trades_to_save)} new trades.")

            # 学習テーブル更新
            try:
                for row in trades_to_save:
                    _, _, _, _, _, _, _, pl_pips, _, _, strat, mreg, mcreg = row
                    record_trade_performance(strat, mreg, mcreg, pl_pips)
            except Exception as e:
                print(f"[PositionManager] learning update error: {e}")

        if processed_tx_ids:
            self._last_tx_id = max(processed_tx_ids)

    def sync_trades(self):
        """定期的に呼び出し、決済済みトレードを同期する"""
        transactions = self._fetch_closed_trades()
        if transactions:
            self._parse_and_save_trades(transactions)

    def close(self):
        self.con.close()
