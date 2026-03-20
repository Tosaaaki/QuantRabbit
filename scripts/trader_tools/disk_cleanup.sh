#!/bin/bash
# disk_cleanup.sh — ディスク容量監視+キャッシュ自動クリーンアップ
# launchdで1時間ごとに実行。閾値以下なら自動でキャッシュ削除
#
# 手動実行: bash scripts/trader_tools/disk_cleanup.sh
# ドライラン: DRY_RUN=1 bash scripts/trader_tools/disk_cleanup.sh

set -euo pipefail

LOG_DIR="/Users/tossaki/App/QuantRabbit/logs"
LOG_FILE="$LOG_DIR/disk_cleanup.log"
ALERT_FILE="$LOG_DIR/disk_alert.json"
THRESHOLD_PCT=85  # この使用率を超えたらクリーンアップ実行
DRY_RUN="${DRY_RUN:-0}"

log() {
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $1" >> "$LOG_FILE"
}

# 現在の使用率を取得（macOS: dfの出力からCapacity列）
get_usage_pct() {
    df -h / | awk 'NR==2 {gsub(/%/,"",$5); print $5}'
}

get_avail() {
    df -h / | awk 'NR==2 {print $4}'
}

# クリーンアップ対象（安全なキャッシュのみ）
cleanup_targets=(
    "$HOME/Library/Caches/Google"
    "$HOME/Library/Caches/CocoaPods"
    "$HOME/Library/Caches/ms-playwright"
    "$HOME/Library/Caches/com.spotify.client"
    "$HOME/Library/Caches/com.microsoft.EdgeUpdater"
    "$HOME/Library/Caches/com.apple.Safari/WebKitCache"
    "/private/tmp/claude-501"
)

USAGE=$(get_usage_pct)
AVAIL=$(get_avail)

# アラートJSON更新（secretaryが読める）
cat > "$ALERT_FILE" << EOJSON
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "disk_usage_pct": $USAGE,
    "disk_avail": "$AVAIL",
    "threshold_pct": $THRESHOLD_PCT,
    "status": "$([ "$USAGE" -ge "$THRESHOLD_PCT" ] && echo "WARNING" || echo "OK")"
}
EOJSON

if [ "$USAGE" -lt "$THRESHOLD_PCT" ]; then
    log "OK: disk ${USAGE}% used, ${AVAIL} avail — no cleanup needed"
    exit 0
fi

log "WARNING: disk ${USAGE}% used (threshold ${THRESHOLD_PCT}%). Starting cleanup..."

FREED_TOTAL=0
for target in "${cleanup_targets[@]}"; do
    if [ -d "$target" ]; then
        SIZE=$(du -sm "$target" 2>/dev/null | awk '{print $1}' || echo "0")
        if [ "$SIZE" -gt 10 ]; then  # 10MB以上のみ対象
            if [ "$DRY_RUN" = "1" ]; then
                log "DRY_RUN: would delete $target (${SIZE}MB)"
            else
                rm -rf "$target" 2>/dev/null || true
                log "CLEANED: $target (${SIZE}MB)"
                FREED_TOTAL=$((FREED_TOTAL + SIZE))
            fi
        fi
    fi
done

USAGE_AFTER=$(get_usage_pct)
AVAIL_AFTER=$(get_avail)
log "DONE: freed ~${FREED_TOTAL}MB. Before: ${USAGE}%, After: ${USAGE_AFTER}% (${AVAIL_AFTER} avail)"

# アラート更新
cat > "$ALERT_FILE" << EOJSON
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "disk_usage_pct": $USAGE_AFTER,
    "disk_avail": "$AVAIL_AFTER",
    "threshold_pct": $THRESHOLD_PCT,
    "status": "$([ "$USAGE_AFTER" -ge "$THRESHOLD_PCT" ] && echo "WARNING" || echo "OK")",
    "last_cleanup": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "freed_mb": $FREED_TOTAL
}
EOJSON
