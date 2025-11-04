#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# 1. 最新の NAV を取得（ログ用）
python3 "$ROOT/scripts/show_account_nav.py" || true

# 2. 監視対象期間の M1 ローソクを取得（例：直近 7 日）
START_DATE=$(date -u -d '7 days ago' +%Y-%m-%d)
END_DATE=$(date -u +%Y-%m-%d)
python3 "$ROOT/scripts/backfill_candles.py" \
  --instrument USD_JPY \
  --timeframes M1 \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --output-dir "$ROOT/logs/oanda_candles" \
  --fetch-missing || true

# 3. リプレイ（オプション：直近 30 日の挙動チェック）
python3 "$ROOT/scripts/replay_manual_swing.py" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --initial-nav 500000 \
  --margin-rate 0.02 \
  --out "$ROOT/tmp/replay_manual_swing_latest.json" || true

# 4. 実トレードワーカー起動
exec python3 -m workers.manual_swing.worker
