"""前向き検定用の M1 bid/ask を OANDA から取得する（読み取り専用）。

既存コーパス `oanda_history_m1_2020_2026` は 2026-07-09 で終わっている。
`docs/RESEARCH_LOG.md` L026 の「次に必要なこと 2」は前向きの未経過期間での確認なので、
その先を取ってくる。**既存コーパスのディレクトリには書かない**（既存の全結果が
変わってしまう）。別ディレクトリに同じ形式で置き、前向きスクリプトだけが読む。

窓は2つに分けて別々に報告する:
  W1 2026-07-10 → 2026-08-06  コーパス外だが、仮説確定(8/6)時点では **経過済み**
  W2 2026-08-06 → 現在        仮説確定時点で **未経過**。これが本当の前向き検定

出力形式は既存コーパスと同一:
  {"time": ISO, "bid": {"h","l","c"}, "ask": {"h","l","c"}}
"""
import gzip
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trend_gate import PAIRS  # noqa: E402

OUT = os.path.join(HERE, "forward_m1")
ENV = "/Users/tossaki/App/QuantRabbit/.env.local"
START = "2026-07-10T00:00:00Z"
MAX_COUNT = 5000


def creds():
    env = {}
    with open(ENV, encoding="utf-8") as fh:
        for line in fh:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k] = v
    return env["QR_OANDA_TOKEN"], env.get("QR_OANDA_BASE_URL", "https://api-fxtrade.oanda.com")


def get(base, token, path, query):
    url = f"{base}{path}?{urllib.parse.urlencode(query)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))


def fetch_pair(base, token, pair):
    """完了足のみを時系列順に返す。未確定足は捨てる（look-ahead 防止）。"""
    out = []
    cursor = START
    seen = set()
    while True:
        payload = get(base, token, f"/v3/instruments/{pair}/candles",
                      {"granularity": "M1", "from": cursor, "count": str(MAX_COUNT), "price": "BA"})
        candles = payload.get("candles") or []
        fresh = [c for c in candles if c.get("complete") and c["time"] not in seen]
        if not fresh:
            break
        for c in fresh:
            seen.add(c["time"])
            out.append({
                "time": c["time"],
                "bid": {k: c["bid"][k] for k in ("h", "l", "c")},
                "ask": {k: c["ask"][k] for k in ("h", "l", "c")},
            })
        cursor = (datetime.fromisoformat(fresh[-1]["time"][:19]).replace(tzinfo=timezone.utc)
                  + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        if len(candles) < MAX_COUNT:
            break
        if cursor >= datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"):
            break
    return out


def main():
    token, base = creds()
    os.makedirs(OUT, exist_ok=True)
    for pair in (sys.argv[1:] or PAIRS):
        path = os.path.join(OUT, f"{pair}_M1_BA_forward.jsonl.gz")
        if os.path.exists(path):
            print(f"{pair}: exists, skip", file=sys.stderr)
            continue
        rows = fetch_pair(base, token, pair)
        with gzip.open(path, "wt") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        span = f"{rows[0]['time'][:16]} -> {rows[-1]['time'][:16]}" if rows else "empty"
        print(f"{pair}: {len(rows):6d} bars  {span}", file=sys.stderr)


if __name__ == "__main__":
    main()
