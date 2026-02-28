#!/usr/bin/env bash
# Recover unstable IAP OS Login SSH authentication and run a deterministic verification loop.
#
# Usage:
#   scripts/recover_iap_ssh_auth.sh [-p PROJECT] [-z ZONE] [-m INSTANCE] \
#     [-k SSH_KEYFILE] [-t KEY_TTL] [-r RETRY_COUNT] [-s RETRY_SLEEP] \
#     [-K SA_KEYFILE] [-A SA_IMPERSONATE]
#
# Example:
#   scripts/recover_iap_ssh_auth.sh -p quantrabbit -k ~/.ssh/gcp_oslogin_qr -r 6 -s 3

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT=""
ZONE="asia-northeast1-a"
INSTANCE="fx-trader-vm"
KEYFILE="$HOME/.ssh/gcp_oslogin_qr"
KEY_TTL="30d"
MAX_RETRY=6
RETRY_SLEEP=2
SA_KEYFILE=""
SA_IMPERSONATE=""

usage() {
  cat <<'USAGE'
IAP OS Login SSH 認証の復旧用ヘルパー

options:
  -p PROJECT       GCP project (default: gcloud config project)
  -z ZONE          GCE zone (default: asia-northeast1-a)
  -m INSTANCE      VM name (default: fx-trader-vm)
  -k KEYFILE       OS Login private key file (default: ~/.ssh/gcp_oslogin_qr)
  -t TTL           OS Login key TTL (default: 30d, use none for no expiration)
  -r RETRIES       Retry count (default: 6)
  -s SLEEP_SECONDS Sleep between retries (default: 2)
  -K SA_KEYFILE    SA JSON key for commands (optional)
  -A SA_ACCOUNT    service account email to impersonate (optional)
  -h               show this help

Example:
  scripts/recover_iap_ssh_auth.sh -p quantrabbit -k ~/.ssh/gcp_oslogin_qr -r 6 -s 3
USAGE
}

red() { printf "\033[31m%s\033[0m\n" "$*"; }
grn() { printf "\033[32m%s\033[0m\n" "$*"; }
ylw() { printf "\033[33m%s\033[0m\n" "$*"; }

while getopts ":p:z:m:k:t:r:s:K:A:h" opt; do
  case "$opt" in
    p) PROJECT="$OPTARG" ;;
    z) ZONE="$OPTARG" ;;
    m) INSTANCE="$OPTARG" ;;
    k) KEYFILE="$OPTARG" ;;
    t) KEY_TTL="$OPTARG" ;;
    r) MAX_RETRY="$OPTARG" ;;
    s) RETRY_SLEEP="$OPTARG" ;;
    K) SA_KEYFILE="$OPTARG" ;;
    A) SA_IMPERSONATE="$OPTARG" ;;
    h)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$SCRIPT_DIR/gcloud_doctor.sh" ]]; then
  red "scripts/gcloud_doctor.sh が見つからないか実行権限がありません。"
  exit 2
fi

if ! command -v gcloud >/dev/null 2>&1; then
  red "gcloud が見つかりません。先に SDK を導入してください。"
  exit 2
fi

if [[ -z "$PROJECT" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "$PROJECT" ]]; then
  red "PROJECT が未指定です。-p で指定するか gcloud config set project を実行してください。"
  exit 2
fi

if [[ -f "$KEYFILE" && ! -r "$KEYFILE" ]]; then
  red "$KEYFILE は読み取り不可です。chmod 600 を確認してください。"
  exit 2
fi

mkdir -p "$(dirname "$KEYFILE")"
chmod 700 "$(dirname "$KEYFILE")"
chmod 600 "$KEYFILE" 2>/dev/null || true

declare -a DOCTOR_ARGS=(
  -p "$PROJECT"
  -z "$ZONE"
  -m "$INSTANCE"
  -t
  -S
  -G
  -O
  -T "$KEY_TTL"
  -k "$KEYFILE"
)
if [[ -n "$SA_KEYFILE" ]]; then
  DOCTOR_ARGS+=(-K "$SA_KEYFILE")
fi
if [[ -n "$SA_IMPERSONATE" ]]; then
  DOCTOR_ARGS+=(-A "$SA_IMPERSONATE")
fi

for (( attempt=1; attempt<=MAX_RETRY; attempt++ )); do
  ylw "[$attempt/$MAX_RETRY] IAP/OS Login preflight + SSH接続確認"

  if output="$(
    "$SCRIPT_DIR/gcloud_doctor.sh" "${DOCTOR_ARGS[@]}" -c 2>&1
  )"; then
    grn "IAP/OS Login 接続確認が成功しました。"
    echo "$output"
    exit 0
  fi

  echo "$output"

  if (( attempt == MAX_RETRY )); then
    break
  fi

  if grep -q "Permission denied (publickey)" <<<"$output"; then
    ylw "publickey エラーを検知。鍵再登録を実施して再試行します。"
    chmod 600 "$KEYFILE" 2>/dev/null || true
  else
    ylw "認証以外の要因で失敗。メタデータ/VM 側状態が変わる可能性があるため再試行します。"
  fi

  sleep "$RETRY_SLEEP"
done

cat <<EOF
IAP SSH 認証が回復しませんでした。次を順に実行してください。

1) キー紐付き実行ユーザと IAM の再確認
  gcloud auth list --filter=status:ACTIVE
  gcloud projects get-iam-policy ${PROJECT} --flatten="bindings[].members" --filter="bindings.role:roles/iap.tunnelResourceAccessor"

2) 代替デプロイ（SSH 不要）
  scripts/deploy_via_metadata.sh -p ${PROJECT} -z ${ZONE} -m ${INSTANCE} -b main -i

3) VM 側 SSH 回復トリガ（最小）
  sudo python3 /home/tossaki/QuantRabbit/scripts/dedupe_systemd_envfiles.py --apply --services "quant-ssh-watchdog.service quant-ssh-watchdog.timer"
  sudo bash scripts/install_trading_services.sh --repo /home/tossaki/QuantRabbit --units quant-ssh-watchdog.service quant-ssh-watchdog.timer
  sudo systemctl enable --now quant-ssh-watchdog.timer
EOF

red "Recovery failed. 最終結果: IAP/OS Login で公開鍵認証が成立しませんでした。"
exit 2
