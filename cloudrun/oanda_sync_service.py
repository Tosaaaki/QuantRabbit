import os
import logging
from datetime import datetime
from typing import Any, Dict, List

import httpx
from flask import Flask
from google.cloud import firestore

from utils.secrets import get_secret
from utils.firestore_helpers import apply_filter


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
app = Flask(__name__)
fs = firestore.Client()


def _oanda_base() -> Dict[str, str]:
    token = get_secret("oanda_token")
    account = get_secret("oanda_account_id")
    try:
        practice = get_secret("oanda_practice").lower() == "true"
    except Exception:
        practice = True
    host = "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"
    return {"token": token, "account": account, "host": host}


def _get_state() -> Dict[str, Any]:
    doc = fs.collection("state").document("oanda_sync").get()
    return doc.to_dict() if doc.exists else {}


def _set_state(st: Dict[str, Any]):
    fs.collection("state").document("oanda_sync").set(st)


def _save_tx(tx: Dict[str, Any]):
    try:
        txid = tx.get("id") or tx.get("transactionID") or datetime.utcnow().isoformat()
        fs.collection("oanda_transactions").document(str(txid)).set(tx)
    except Exception:
        pass


def _update_trade_open(trade_id: str, fill_price: float, time_iso: str, instrument: str):
    try:
        q = apply_filter(fs.collection("trades"), "trade_id", "==", str(trade_id)).limit(1)
        docs = list(q.stream())
        if not docs:
            return
        ref = docs[0].reference
        ref.update({
            "fill_price": fill_price,
            "fill_time": time_iso,
            "instrument": instrument,
            "state": "OPEN",
        })
    except Exception:
        pass


def _update_trade_close(trade_id: str, close_price: float, pl: float, time_iso: str):
    try:
        q = apply_filter(fs.collection("trades"), "trade_id", "==", str(trade_id)).limit(1)
        docs = list(q.stream())
        if not docs:
            return
        ref = docs[0].reference
        ref.update({
            "close_price": close_price,
            "realized_pl": pl,
            "close_time": time_iso,
            "state": "CLOSED",
        })
    except Exception:
        pass


def _handle_order_fill(tx: Dict[str, Any]):
    time_iso = tx.get("time")
    instrument = tx.get("instrument", "")
    opened = tx.get("tradeOpened")
    if opened and opened.get("tradeID"):
        trade_id = opened.get("tradeID")
        price = float(opened.get("price", tx.get("price", 0.0)))
        _update_trade_open(trade_id, price, time_iso, instrument)
    closed_list: List[Dict[str, Any]] = tx.get("tradesClosed", []) or []
    for c in closed_list:
        trade_id = c.get("tradeID")
        if not trade_id:
            continue
        price = float(c.get("price", tx.get("price", 0.0)))
        pl = float(c.get("realizedPL", tx.get("pl", 0.0)))
        _update_trade_close(trade_id, price, pl, time_iso)


def _handle_trade_close(tx: Dict[str, Any]):
    time_iso = tx.get("time")
    closed_list: List[Dict[str, Any]] = tx.get("tradesClosed", []) or []
    for c in closed_list:
        trade_id = c.get("tradeID")
        if not trade_id:
            continue
        price = float(c.get("price", tx.get("price", 0.0)))
        pl = float(c.get("realizedPL", tx.get("pl", 0.0)))
        _update_trade_close(trade_id, price, pl, time_iso)


@app.route("/", methods=["GET"])  # 1 リクエスト=1回同期
def run_once():
    try:
        base = _oanda_base()
        token, account, host = base["token"], base["account"], base["host"]

        st = _get_state()
        since_id = st.get("last_tx_id")

        headers = {"Authorization": f"Bearer {token}"}
        if since_id:
            url = f"{host}/v3/accounts/{account}/transactions/sinceid"
            params = {"id": str(since_id)}
        else:
            url = f"{host}/v3/accounts/{account}/transactions"
            params = {"pageSize": "50"}
        with httpx.Client(timeout=7.0, follow_redirects=True) as client:
            r = client.get(url, params=params, headers=headers)
            r.raise_for_status()
            data = r.json()

        txs: List[Dict[str, Any]] = data.get("transactions", []) or []
        last_id = data.get("lastTransactionID") or (txs[-1]["id"] if txs else since_id)

        processed = 0
        for tx in txs:
            _save_tx(tx)
            t = tx.get("type")
            if t == "ORDER_FILL":
                _handle_order_fill(tx)
            elif t == "TRADE_CLOSE":
                _handle_trade_close(tx)
            processed += 1

        if last_id:
            _set_state({"last_tx_id": last_id, "updated_at": datetime.utcnow().isoformat(timespec="seconds")})

        return f"synced={processed}, last_tx_id={last_id}", 200
    except Exception as e:
        logging.error(f"oanda_sync error: {e}")
        return "ERROR", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
