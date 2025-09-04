import os
import logging
from datetime import datetime, timedelta

from flask import Flask
from google.cloud import firestore


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
app = Flask(__name__)
fs = firestore.Client()


def _recent_trades(hours=24):
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat(timespec="seconds")
    q = fs.collection("trades").order_by("ts").limit(1000)
    docs = list(q.stream())
    out = []
    for d in docs:
        x = d.to_dict() or {}
        if x.get("ts") and x["ts"] >= cutoff:
            out.append(x)
    return out


@app.route("/", methods=["GET"])  # 1 リクエスト=1回の自動調整
def run_once():
    try:
        trades = _recent_trades(24)
        n = len(trades)
        # 簡易: 失敗（trade_id=='FAIL'）比率でリスク配分を微調整
        fail = sum(1 for t in trades if t.get("trade_id") == "FAIL")
        fail_rate = (fail / n) if n else 0.0

        # 現在値
        params_ref = fs.collection("config").document("params")
        params = params_ref.get().to_dict() or {}
        risk_share = float(params.get("RISK_SHARE_OF_TARGET", os.environ.get("RISK_SHARE_OF_TARGET", "0.5")))
        cooldown_min = int(params.get("DECIDER_COOLDOWN_MIN", os.environ.get("DECIDER_COOLDOWN_MIN", "30")))

        # 調整ロジック（安全範囲内）
        # 失敗率が高ければリスク低下、低ければ微増。クールダウンも同様に増減。
        if fail_rate > 0.4:
            risk_share = max(risk_share - 0.05, 0.30)
            cooldown_min = min(max(cooldown_min + 5, 20), 60)
        elif fail_rate < 0.2:
            risk_share = min(risk_share + 0.05, 0.60)
            cooldown_min = max(min(cooldown_min - 5, 45), 15)

        params_ref.set({
            "RISK_SHARE_OF_TARGET": round(risk_share, 2),
            "DECIDER_COOLDOWN_MIN": int(cooldown_min),
            "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
            "sample": n,
            "fail_rate": round(fail_rate, 2),
        })

        return f"updated risk_share={risk_share}, cooldown={cooldown_min}, n={n}, fail_rate={fail_rate:.2f}", 200
    except Exception as e:
        logging.error(f"autotune error: {e}")
        return "ERROR", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

