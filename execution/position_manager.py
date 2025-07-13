from __future__ import annotations
import requests, sqlite3, pathlib
from datetime import datetime, timezone
from google.cloud import secretmanager

# --- Secret Managerからシークレットを取得するヘルパー関数 ---
def access_secret_version(secret_id: str, project_id: str = "quantrabbit", version_id: str = "latest") -> str:
    """Secret Managerから指定されたシークレットのバージョンにアクセスします。"""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# --- config ---
# Secret ManagerからOANDAのトークンとアカウントIDを取得
TOKEN = access_secret_version("oanda_token")
ACCOUNT = access_secret_version("oanda_account_id")
PRACT = False # env.tomlから取得しないため、ここでは固定値とする
REST_HOST = "https://api-fxpractice.oanda.com" if PRACT else "https://api-fxtrade.oanda.com"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

_DB = pathlib.Path("logs/trades.db")

class PositionManager:
    def __init__(self):
        self.con = sqlite3.connect(_DB)
        self._last_tx_id = self._get_last_transaction_id()

    def _get_last_transaction_id(self) -> int:
        """DBに記録済みの最新トランザクションIDを取得"""
        row = self.con.execute("SELECT MAX(id) FROM trades").fetchone()
        # OANDAのTransactionIDは1から始まるので、DBが空なら0を返す
        return row[0] if row and row[0] else 0

    def _fetch_closed_trades(self):
        """OANDAから決済済みトランザクションを取得"""
        url = f"{REST_HOST}/v3/accounts/{ACCOUNT}/transactions"
        params = {
            "sinceID": self._last_tx_id,
            "type": "ORDER_FILL" #約定した注文のみ
        }
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
                "entry_time": datetime.fromisoformat(trade.get("openTime").replace("Z", "+00:00")),
                "units": int(trade.get("initialUnits", 0)),
                "pocket": pocket,
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
                close_time = datetime.fromisoformat(tx.get("time").replace("Z", "+00:00"))
                
                # USD/JPY の pips を計算 (1 pip = 0.01 JPY)
                # OANDAのPLは通貨額なので、価格差からpipsを計算する
                entry_price = details["entry_price"]
                units = details["units"]
                if units > 0: # Buy
                    pl_pips = (close_price - entry_price) * 100
                else: # Sell
                    pl_pips = (entry_price - close_price) * 100

                trades_to_save.append((
                    tx["id"],
                    trade_id,
                    details["pocket"],
                    tx.get("instrument"),
                    units,
                    entry_price,
                    close_price,
                    pl_pips,
                    details["entry_time"].isoformat(),
                    close_time.isoformat()
                ))
            
            processed_tx_ids.add(tx["id"])

        if trades_to_save:
            # ticket_id (OANDA tradeID) が重複しないように挿入
            self.con.executemany("""
                INSERT OR IGNORE INTO trades (id, ticket_id, pocket, instrument, units, entry_price, close_price, pl_pips, entry_time, close_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, trades_to_save)
            self.con.commit()
            print(f"[PositionManager] Saved {len(trades_to_save)} new trades.")
        
        if processed_tx_ids:
            self._last_tx_id = max(processed_tx_ids)

    def sync_trades(self):
        """定期的に呼び出し、決済済みトレードを同期する"""
        transactions = self._fetch_closed_trades()
        if transactions:
            self._parse_and_save_trades(transactions)

    def close(self):
        self.con.close()
