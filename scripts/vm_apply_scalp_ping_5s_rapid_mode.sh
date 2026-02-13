#!/usr/bin/env bash
set -euo pipefail

# Usage:
#  scripts/vm_apply_scalp_ping_5s_rapid_mode.sh -p <PROJECT> -z <ZONE> -m <INSTANCE> -t --mode base|safe
#  scripts/vm_apply_scalp_ping_5s_rapid_mode.sh ... --mode auto [--window-min 15]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VM_SH="${SCRIPT_DIR}/vm.sh"

PROJECT=""
ZONE=""
INSTANCE=""
ACCOUNT=""
KEYFILE=""
USE_IAP=""
LOCAL_MODE="${SCALP_PING_5S_AUTO_LOCAL:-0}"
MODE="base"
DRY_RUN=""
WINDOW_MIN="15"
AUTO="0"

BASE_ENV="ops/env/scalp_ping_5s_rapid_mode_base_20260213.env"
SAFE_ENV="ops/env/scalp_ping_5s_rapid_mode_safe_20260213.env"

SL_RATE_SAFE=0.06
SL_RATE_RETURN=0.04
SHORT_AVG_SAFE=-3.0
SHORT_AVG_RETURN=-1.8
LONG_AVG_SAFE=-3.0
LONG_AVG_RETURN=-1.8

BASE_STREAK_TARGET=2

# 市況圧力: 高くなるほど保守化
MARKET_PRESSURE_TO_SAFE=0.30
MARKET_PRESSURE_RETURN=0.14
REJECT_RATE_RETURN=0.12
PERF_BLOCK_RATE_RETURN=0.10
TRADE_COUNT_MIN_FOR_RETURN=20
TRADE_COUNT_MIN_FOR_EVAL=1

# dynamic scaling bounds
BASE_SCALE_MIN=0.70
BASE_SCALE_MAX=1.28
SAFE_SCALE_MIN=0.80
SAFE_SCALE_MAX=1.04

die() { echo "[scalp5 rapid mode] $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--project) PROJECT="$2"; shift 2 ;;
    -z|--zone) ZONE="$2"; shift 2 ;;
    -m|--instance) INSTANCE="$2"; shift 2 ;;
    -A|--account) ACCOUNT="$2"; shift 2 ;;
    -k|--keyfile) KEYFILE="$2"; shift 2 ;;
    -t|--iap) USE_IAP=1; shift ;;
    --mode) MODE="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
    --window-min) WINDOW_MIN="$2"; shift 2 ;;
    --auto) AUTO=1; shift ;;
    --local) LOCAL_MODE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *) die "Unknown arg: $1" ;;
  esac
done

if [[ "$MODE" == "auto" && "$AUTO" == "0" ]]; then
  AUTO=1
  MODE="base"
fi

if [[ -n "${LOCAL_MODE:-}" && "$LOCAL_MODE" == "1" ]]; then
  PROJECT="local"
  ZONE="local"
  INSTANCE="local"
else
  [[ -n "$PROJECT" && -n "$ZONE" && -n "$INSTANCE" ]] || die "-p/--project, -z/--zone, -m/--instance are required"
  [[ -x "$VM_SH" ]] || die "vm.sh not found or not executable: $VM_SH"
fi
[[ -f "$BASE_ENV" ]] || die "Base env file missing: $BASE_ENV"
[[ -f "$SAFE_ENV" ]] || die "Safe env file missing: $SAFE_ENV"

case "$MODE" in
  base|safe|auto) ;;
  *) die "--mode must be base|safe|auto" ;;
esac

run_vm() {
  local -a args=("$VM_SH" -p "$PROJECT" -z "$ZONE" -m "$INSTANCE")
  if [[ -n "$ACCOUNT" ]]; then args+=(-A "$ACCOUNT"); fi
  if [[ -n "$KEYFILE" ]]; then args+=(-k "$KEYFILE"); fi
  if [[ -n "$USE_IAP" ]]; then args+=(-t); fi
  args+=("$@")

  if [[ -n "$DRY_RUN" ]]; then
    printf '[dry-run]'
    printf ' %q' "${args[@]}"
    printf '\n'
    return 0
  fi
  "${args[@]}"
}

query_db() {
  local db_file="$1"
  local query="$2"
  if [[ -n "${LOCAL_MODE:-}" && "$LOCAL_MODE" == "1" ]]; then
    sqlite3 -readonly "${db_file}" "${query}"
  else
    run_vm exec -- "sqlite3 -readonly ${db_file} \"${query}\""
  fi
}

query_db_or_default() {
  local db_file="$1"
  local query="$2"
  local fallback="$3"
  local result
  if result="$(query_db "$db_file" "$query")"; then
    echo "$result"
  else
    echo "$fallback"
  fi
}

ratio() {
  awk -v n="$1" -v d="$2" 'BEGIN { if (d+0 <= 0) print "0"; else printf "%.6f", n / d }'
}

clamp_float() {
  awk -v v="$1" -v lo="$2" -v hi="$3" 'BEGIN { if (v < lo) v = lo; if (v > hi) v = hi; printf "%.6f", v }'
}

round1() {
  awk -v v="$1" 'BEGIN { printf "%.4f", v }'
}

round0() {
  awk -v v="$1" 'BEGIN { if (v == "" || v+0 != v) v = 0; if (v < 1) { if (v > 0) printf "%.3f", v; else print 0; } else print int(v + 0.5) }'
}

scale_int_value() {
  awk -v v="$1" -v s="$2" -v lo="$3" -v hi="$4" 'BEGIN {
    if (v == "" || v+0 != v) { print lo; exit }
    x = v * s
    if (x < lo) x = lo
    if (x > hi) x = hi
    if (x < 1) x = 1
    printf "%d", int(x + 0.5)
  }'
}

scale_float_value() {
  awk -v v="$1" -v s="$2" -v lo="$3" -v hi="$4" 'BEGIN {
    if (v == "" || v+0 != v) { print lo; exit }
    x = v * s
    if (x < lo) x = lo
    if (x > hi) x = hi
    printf "%.4f", x
  }'
}

get_env_value() {
  local env_file="$1"
  local key="$2"
  local default_value="$3"
  local value
  value="$(awk -F= -v key="$key" '$1==key {print $2; exit}' "$env_file" 2>/dev/null || true)"
  echo "${value:-$default_value}"
}

get_current_mode() {
  if [[ -n "${LOCAL_MODE:-}" && "$LOCAL_MODE" == "1" ]]; then
    grep -E '^SCALP_PING_5S_RAPID_MODE=' /etc/quantrabbit/scalp_ping_5s.env 2>/dev/null | sed -n '1p' | cut -d= -f2 || true
  else
    run_vm exec -- "grep -E '^SCALP_PING_5S_RAPID_MODE=' /etc/quantrabbit/scalp_ping_5s.env 2>/dev/null | sed -n '1p' | cut -d= -f2"
  fi
}

collect_trade_metrics() {
  local query
  local stats
  query="SELECT \
    COALESCE(CAST(SUM(CASE WHEN units < 0 AND close_reason='STOP_LOSS_ORDER' THEN 1 ELSE 0 END) AS REAL) / MAX(CAST(SUM(CASE WHEN units < 0 THEN 1 ELSE 0 END) AS REAL), 0.0001), \
    COALESCE(AVG(CASE WHEN units < 0 AND close_reason IS NOT NULL THEN pl_pips END), 0), \
    COALESCE(CAST(SUM(CASE WHEN units > 0 AND close_reason='STOP_LOSS_ORDER' THEN 1 ELSE 0 END) AS REAL) / MAX(CAST(SUM(CASE WHEN units > 0 THEN 1 ELSE 0 END) AS REAL), 0.0001), \
    COALESCE(AVG(CASE WHEN units > 0 AND close_reason IS NOT NULL THEN pl_pips END), 0), \
    COALESCE(CAST(SUM(CASE WHEN close_reason='STOP_LOSS_ORDER' THEN 1 ELSE 0 END) AS REAL) / MAX(CAST(SUM(CASE WHEN close_reason IS NOT NULL THEN 1 ELSE 0 END) AS REAL), 0.0001), \
    COALESCE(AVG(CASE WHEN close_reason IS NOT NULL THEN pl_pips END), 0), \
    COUNT(1), \
    (SELECT COUNT(1) FROM trades t2 WHERE t2.strategy_tag='scalp_ping_5s_live' AND t2.state='open') \
    FROM trades \
    WHERE strategy_tag='scalp_ping_5s_live' \
      AND close_time >= datetime('now','-${WINDOW_MIN} minutes') \
      AND close_time IS NOT NULL;"

  stats="$(query_db_or_default /home/tossaki/QuantRabbit/logs/trades.db "$query" "0|0|0|0|0|0|0|0")"
  IFS='|' read -r SHORT_SL_RATE SHORT_AVG LONG_SL_RATE LONG_AVG OVERALL_SL_RATE OVERALL_AVG TRADE_COUNT OPEN_TRADES <<< "$stats"

  SHORT_SL_RATE="${SHORT_SL_RATE:-0}"
  SHORT_AVG="${SHORT_AVG:-0}"
  LONG_SL_RATE="${LONG_SL_RATE:-0}"
  LONG_AVG="${LONG_AVG:-0}"
  OVERALL_SL_RATE="${OVERALL_SL_RATE:-0}"
  OVERALL_AVG="${OVERALL_AVG:-0}"
  TRADE_COUNT="${TRADE_COUNT:-0}"
  OPEN_TRADES="${OPEN_TRADES:-0}"
}

collect_order_metrics() {
  local query
  query="SELECT \
    COUNT(1), \
    COALESCE(SUM(CASE WHEN lower(status)='filled' THEN 1 ELSE 0 END),0), \
    COALESCE(SUM(CASE WHEN lower(status) IN ('error','fail','failed') OR lower(status) LIKE 'rejected%' OR lower(status) LIKE '%reject%' THEN 1 ELSE 0 END),0), \
    COALESCE(SUM(CASE WHEN lower(status) LIKE 'perf_block%' THEN 1 ELSE 0 END),0), \
    COALESCE(SUM(CASE WHEN lower(status)='blocked' THEN 1 ELSE 0 END),0) \
    FROM orders \
    WHERE pocket='scalp_fast' \
      AND ts >= datetime('now','-${WINDOW_MIN} minutes');"

  ORDER_STATS="$(query_db_or_default /home/tossaki/QuantRabbit/logs/orders.db "$query" "0|0|0|0|0")"
  IFS='|' read -r ORDER_TOTAL ORDER_FILLED ORDER_REJECT ORDER_PERF_BLOCK ORDER_BLOCK <<< "$ORDER_STATS"

  ORDER_TOTAL="${ORDER_TOTAL:-0}"
  ORDER_FILLED="${ORDER_FILLED:-0}"
  ORDER_REJECT="${ORDER_REJECT:-0}"
  ORDER_PERF_BLOCK="${ORDER_PERF_BLOCK:-0}"
  ORDER_BLOCK="${ORDER_BLOCK:-0}"

  ORDER_RATE="$(ratio "$ORDER_TOTAL" "$WINDOW_MIN")"
  ORDER_FILL_RATE="$(ratio "$ORDER_FILLED" "$ORDER_TOTAL")"
  ORDER_REJECT_RATE="$(ratio "$ORDER_REJECT" "$ORDER_TOTAL")"
  ORDER_PERF_BLOCK_RATE="$(ratio "$ORDER_PERF_BLOCK" "$ORDER_TOTAL")"
}

compute_market_pressure_and_tilt() {
  MARKET_PRESSURE="$(awk -v order_rate="$ORDER_RATE" -v reject_rate="$ORDER_REJECT_RATE" -v perf_block_rate="$ORDER_PERF_BLOCK_RATE" -v open_trades="$OPEN_TRADES" 'BEGIN {
    p = 0
    if (order_rate + 0.0 > 0.0) {
      p += (order_rate - 8.0) / 24.0
    }
    p += reject_rate * 1.75
    p += perf_block_rate * 1.5
    p += (open_trades - 8.0) / 24.0
    if (p < 0.0) p = 0.0
    if (p > 1.0) p = 1.0
    printf "%.6f", p
  }')"

  PERF_TILT="$(awk -v tc="$TRADE_COUNT" -v s1="$SHORT_SL_RATE" -v a1="$SHORT_AVG" -v s2="$LONG_SL_RATE" -v a2="$LONG_AVG" -v o1="$OVERALL_SL_RATE" -v a="$OVERALL_AVG" 'BEGIN {
    if (tc+0 < 4) { print 0; exit }
    t = 0.0
    if (s1 <= 0.03) t += 0.20; else if (s1 >= 0.08) t -= 0.25
    if (s2 <= 0.03) t += 0.20; else if (s2 >= 0.08) t -= 0.25
    if (o1 <= 0.03) t += 0.15; else if (o1 >= 0.08) t -= 0.20

    if (a1 >= 0.0) t += 0.15; else if (a1 <= -2.0) t -= 0.15
    if (a2 >= 0.0) t += 0.15; else if (a2 <= -2.0) t -= 0.15
    if (a >= -0.6) t += 0.15; else if (a <= -1.4) t -= 0.15

    if (t > 0.50) t = 0.50
    if (t < -0.50) t = -0.50
    printf "%.4f", t
  }')"

  DYNAMIC_SCALE="$(awk -v mode="$TARGET_MODE" -v m="$MARKET_PRESSURE" -v t="$PERF_TILT" 'BEGIN {
    if (mode == "base") {
      s = 1.0 - (m * 0.45) + (t * 0.55)
      if (s < 0.70) s = 0.70
      if (s > 1.28) s = 1.28
    } else {
      s = 0.95 - (m * 0.28)
      if (t > 0) s += (t * 0.15)
      if (s < 0.80) s = 0.80
      if (s > 1.06) s = 1.06
    }
    if (s < 0) s = 0.001
    printf "%.6f", s
  }')"
}

build_dynamic_overrides() {
  local profile_env="$1"
  local out_file="$2"
  local base_orders base_active base_dir base_cd base_spacing base_loop
  local orders active dir cd spacing loop
  local inv_scale

  base_orders="$(get_env_value "$profile_env" "SCALP_PING_5S_MAX_ORDERS_PER_MINUTE" "80")"
  base_active="$(get_env_value "$profile_env" "SCALP_PING_5S_MAX_ACTIVE_TRADES" "32")"
  base_dir="$(get_env_value "$profile_env" "SCALP_PING_5S_MAX_PER_DIRECTION" "12")"
  base_cd="$(get_env_value "$profile_env" "SCALP_PING_5S_ENTRY_COOLDOWN_SEC" "0.20")"
  base_spacing="$(get_env_value "$profile_env" "SCALP_PING_5S_MIN_ORDER_SPACING_SEC" "0.10")"
  base_loop="$(get_env_value "$profile_env" "SCALP_PING_5S_LOOP_INTERVAL_SEC" "0.20")"

  inv_scale="$(awk -v s="$DYNAMIC_SCALE" 'BEGIN { if (s <= 0) s = 1; printf "%.6f", 1.0/s }')"

  orders="$(scale_int_value "$base_orders" "$DYNAMIC_SCALE" 30 260)"
  active="$(scale_int_value "$base_active" "$DYNAMIC_SCALE" 4 80)"
  dir="$(scale_int_value "$base_dir" "$DYNAMIC_SCALE" 2 40)"
  cd="$(scale_float_value "$base_cd" "$inv_scale" 0.05 0.80)"
  spacing="$(scale_float_value "$base_spacing" "$inv_scale" 0.05 0.80)"
  loop="$(scale_float_value "$base_loop" "$inv_scale" 0.06 0.40)"

  cat > "$out_file" <<EOF
SCALP_PING_5S_AUTO_MODE=auto
SCALP_PING_5S_AUTO_TARGET_MODE=${TARGET_MODE}
SCALP_PING_5S_AUTO_TARGET_SCALE=${DYNAMIC_SCALE}
SCALP_PING_5S_AUTO_MARKET_PRESSURE=${MARKET_PRESSURE}
SCALP_PING_5S_AUTO_PERF_TILT=${PERF_TILT}
SCALP_PING_5S_AUTO_ORDER_RATE=${ORDER_RATE}
SCALP_PING_5S_AUTO_OPEN_TRADES=${OPEN_TRADES}
SCALP_PING_5S_AUTO_ORDER_REJECT_RATE=${ORDER_REJECT_RATE}
SCALP_PING_5S_AUTO_ORDER_PERF_BLOCK_RATE=${ORDER_PERF_BLOCK_RATE}
SCALP_PING_5S_MAX_ORDERS_PER_MINUTE=${orders}
SCALP_PING_5S_MAX_ACTIVE_TRADES=${active}
SCALP_PING_5S_MAX_PER_DIRECTION=${dir}
SCALP_PING_5S_ENTRY_COOLDOWN_SEC=${cd}
SCALP_PING_5S_MIN_ORDER_SPACING_SEC=${spacing}
SCALP_PING_5S_LOOP_INTERVAL_SEC=${loop}
EOF
}

apply_mode() {
  local target_mode="$1"
  local override_file="${2-}"
  local env_file

  if [[ "$target_mode" == "safe" ]]; then
    env_file="$SAFE_ENV"
  else
    env_file="$BASE_ENV"
  fi

  if [[ -n "$override_file" ]]; then
    local tmp_env
    tmp_env="$(mktemp /tmp/qr_scalp_ping_5s_tuning.XXXXXX).env"
    cp "$env_file" "$tmp_env"
    cat "$override_file" >> "$tmp_env"
    if [[ -n "${LOCAL_MODE:-}" && "$LOCAL_MODE" == "1" ]]; then
      bash "$SCRIPT_DIR/vm_apply_scalp_ping_5s_tuning.sh" --local --env-file "$tmp_env" ${DRY_RUN:+--dry-run}
    else
      bash "$SCRIPT_DIR/vm_apply_scalp_ping_5s_tuning.sh" -p "$PROJECT" -z "$ZONE" -m "$INSTANCE" ${USE_IAP:+-t} ${KEYFILE:+-k "$KEYFILE"} ${ACCOUNT:+-A "$ACCOUNT"} --env-file "$tmp_env" ${DRY_RUN:+--dry-run}
    fi
    rm -f "$tmp_env"
  else
    if [[ -n "${LOCAL_MODE:-}" && "$LOCAL_MODE" == "1" ]]; then
      bash "$SCRIPT_DIR/vm_apply_scalp_ping_5s_tuning.sh" --local --env-file "$env_file" ${DRY_RUN:+--dry-run}
    else
      bash "$SCRIPT_DIR/vm_apply_scalp_ping_5s_tuning.sh" -p "$PROJECT" -z "$ZONE" -m "$INSTANCE" ${USE_IAP:+-t} ${KEYFILE:+-k "$KEYFILE"} ${ACCOUNT:+-A "$ACCOUNT"} --env-file "$env_file" ${DRY_RUN:+--dry-run}
    fi
  fi
}

if [[ "$AUTO" == "1" ]]; then
  TARGET_MODE="base"
  SHORT_SL_RATE=0
  SHORT_AVG=0
  LONG_SL_RATE=0
  LONG_AVG=0
  OVERALL_SL_RATE=0
  OVERALL_AVG=0
  TRADE_COUNT=0
  OPEN_TRADES=0
  ORDER_TOTAL=0
  ORDER_RATE=0
  ORDER_FILLED=0
  ORDER_REJECT=0
  ORDER_PERF_BLOCK=0
  ORDER_BLOCK=0
  ORDER_REJECT_RATE=0
  ORDER_PERF_BLOCK_RATE=0
  MARKET_PRESSURE=0
  PERF_TILT=0
  DYNAMIC_SCALE=1

  echo "[INFO] auto mode enabled. window=${WINDOW_MIN}m"
  collect_trade_metrics
  collect_order_metrics
  compute_market_pressure_and_tilt

  if [[ "$TRADE_COUNT" == "0" && "$ORDER_TOTAL" == "0" ]]; then
    echo "[WARN] auto stats empty; fallback to base"
    TARGET_MODE="base"
    STREAK=0
  else
    CURRENT_MODE="$(printf '%s' "$(get_current_mode || true)" | tr '[:upper:]' '[:lower:]')"
    BASE_STREAK_FILE="/tmp/scalp_ping_5s_base_streak"

    if [[ "$CURRENT_MODE" == "base" ]]; then
      if awk -v short_sl="$SHORT_SL_RATE" -v short_avg="$SHORT_AVG" -v long_sl="$LONG_SL_RATE" -v long_avg="$LONG_AVG" -v overall_sl="$OVERALL_SL_RATE" -v overall_avg="$OVERALL_AVG" -v rej="$ORDER_REJECT_RATE" -v perfb="$ORDER_PERF_BLOCK_RATE" -v pressure="$MARKET_PRESSURE" \
        'BEGIN { exit !((short_sl > s2) || (short_avg < s1) || (long_sl > s2) || (long_avg < s1) || (overall_sl > (s2 + 0.02)) || (overall_avg < (s1 - 1.0)) || (rej > 0.18) || (perfb > 0.15) || (pressure > m)) }' \
        s2="$SL_RATE_SAFE" s1="$SHORT_AVG_SAFE" m="$MARKET_PRESSURE_TO_SAFE"; then
        TARGET_MODE="safe"
      else
        TARGET_MODE="base"
      fi
      STREAK=0
    else
      STREAK="$(cat "$BASE_STREAK_FILE" 2>/dev/null || echo 0)"
      if awk -v short_sl="$SHORT_SL_RATE" -v short_avg="$SHORT_AVG" -v long_sl="$LONG_SL_RATE" -v long_avg="$LONG_AVG" -v overall_sl="$OVERALL_SL_RATE" -v overall_avg="$OVERALL_AVG" -v rej="$ORDER_REJECT_RATE" -v perfb="$ORDER_PERF_BLOCK_RATE" -v pressure="$MARKET_PRESSURE" -v tc="$TRADE_COUNT" -v tcmin="$TRADE_COUNT_MIN_FOR_RETURN" -v rej_ret="$REJECT_RATE_RETURN" -v perf_ret="$PERF_BLOCK_RATE_RETURN" -v p_ret="$MARKET_PRESSURE_RETURN" \
        'BEGIN {
          ok= (short_sl <  sr && short_avg > sa && long_sl < sr && long_avg > la && overall_sl < (sr + 0.02) && overall_avg > (sa - 1.0) && rej < rej_ret && perfb < perf_ret && pressure < p_ret && tc >= tcmin);
          exit !ok
        }' \
        sr="$SL_RATE_RETURN" sa="$SHORT_AVG_RETURN" la="$LONG_AVG_RETURN" rej_ret="$REJECT_RATE_RETURN" perf_ret="$PERF_BLOCK_RATE_RETURN" p_ret="$MARKET_PRESSURE_RETURN"; then
        STREAK="$(awk -v s="$STREAK" 'BEGIN { print s+1 }')"
      else
        STREAK=0
      fi

      if [[ "$STREAK" -ge "$BASE_STREAK_TARGET" ]]; then
        TARGET_MODE="base"
      else
        TARGET_MODE="safe"
      fi
      echo "$STREAK" > "$BASE_STREAK_FILE"
    fi
  fi

  compute_market_pressure_and_tilt
  echo "[INFO] short_avg=${SHORT_AVG} short_sl_rate=${SHORT_SL_RATE} long_avg=${LONG_AVG} long_sl_rate=${LONG_SL_RATE} overall_avg=${OVERALL_AVG} overall_sl_rate=${OVERALL_SL_RATE} trade_count=${TRADE_COUNT} orders=${ORDER_TOTAL} order_rate=${ORDER_RATE} reject=${ORDER_REJECT_RATE} perf_block=${ORDER_PERF_BLOCK_RATE} market_pressure=${MARKET_PRESSURE} perf_tilt=${PERF_TILT}"
  if [[ "$TARGET_MODE" == "base" ]]; then
    echo "[INFO] target mode: base (auto)"
  else
    echo "[INFO] target mode: safe (auto)"
  fi
else
  TARGET_MODE="$MODE"
  STREAK=0
fi

if [[ "${TARGET_MODE}" == "base" || "${TARGET_MODE}" == "safe" ]]; then
  if [[ "$AUTO" == "1" ]]; then
    PROFILE_ENV="${BASE_ENV}"
    [[ "$TARGET_MODE" == "safe" ]] && PROFILE_ENV="${SAFE_ENV}"
    DYNAMIC_FILE="$(mktemp /tmp/qr_scalp_ping_5s_dynamic.XXXXXX).env"
    build_dynamic_overrides "$PROFILE_ENV" "$DYNAMIC_FILE"
    echo "[INFO] apply dynamic overrides: scale=${DYNAMIC_SCALE}"
    apply_mode "$TARGET_MODE" "$DYNAMIC_FILE"
    rm -f "$DYNAMIC_FILE"
  else
    apply_mode "$TARGET_MODE"
  fi
else
  die "Target mode invalid: ${TARGET_MODE}"
fi

echo "[INFO] current rapid mode:" && get_current_mode
