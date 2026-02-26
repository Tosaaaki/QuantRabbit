#!/usr/bin/env bash
# gcloud doctor: preflight checks for deployment via gcloud/Compute Engine.
#
# Usage:
#   scripts/gcloud_doctor.sh [-p PROJECT] [-z ZONE] [-m INSTANCE] [-t] [-k SSH_KEYFILE] [-c] [-K SA_KEYFILE] [-A SA_ACCOUNT] [-E] [-S] [-G] [-O] [-T KEY_TTL]
#
# Options:
#   -p PROJECT   GCP project (default: gcloud config get-value project)
#   -z ZONE      GCE zone (default: asia-northeast1-a)
#   -m INSTANCE  VM instance name (default: fx-trader-vm)
#   -t           Use IAP tunneling checks
#   -k SSH_KEYFILE  SSH key file to use for SSH/OS Login (default: ~/.ssh/gcp_oslogin_qr)
#   -c           Attempt SSH connectivity check (non-destructive)
#   -K SA_KEYFILE Service Account JSON key; auto-activate if no active account
#   -A SA_ACCOUNT Service Account email to impersonate for gcloud commands
#   -E           Enable required APIs automatically (compute.googleapis.com)
#   -S           Ensure OS Login has the provided public key (adds if missing)
#   -G           Generate SSH key pair at default path if missing (~/.ssh/gcp_oslogin_qr)
#   -O           Enforce instance metadata (enable-oslogin=TRUE, block-project-ssh-keys=TRUE)
#   -T KEY_TTL   TTL for OS Login key registration when -S is used (default: none; example: 30d)

set -euo pipefail

has() { command -v "$1" >/dev/null 2>&1; }
upper() { printf '%s' "$1" | tr '[:lower:]' '[:upper:]'; }
lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }
red() { printf "\033[31m%s\033[0m\n" "$*"; }
grn() { printf "\033[32m%s\033[0m\n" "$*"; }
ylw() { printf "\033[33m%s\033[0m\n" "$*"; }

ensure_gcloud_in_path() {
  if has gcloud; then return 0; fi
  # Probe common locations and prepend to PATH if found
  candidates=(
    /opt/homebrew/bin/gcloud
    /usr/local/bin/gcloud
    "$HOME/google-cloud-sdk/bin/gcloud"
    /usr/local/Caskroom/google-cloud-sdk/*/google-cloud-sdk/bin/gcloud
    /opt/homebrew/Caskroom/google-cloud-sdk/*/google-cloud-sdk/bin/gcloud
  )
  for c in "${candidates[@]}"; do
    for b in $c; do
      if [ -x "$b" ]; then
        export PATH="$(dirname "$b"):$PATH"
        return 0
      fi
    done
  done
  return 1
}

PROJECT=""; ZONE="asia-northeast1-a"; INSTANCE="fx-trader-vm"; USE_IAP=0; SSH_KEYFILE=""; CHECK_SSH=0
SA_KEYFILE=""; SA_IMPERSONATE=""; ENABLE_APIS=0
ENSURE_OSLOGIN_KEY=0; GENERATE_KEY=0
ENFORCE_INSTANCE_OSLOGIN=0
OSLOGIN_KEY_TTL="none"
while getopts ":p:z:m:k:T:tcK:A:ESGO" opt; do
  case "$opt" in
    p) PROJECT="$OPTARG" ;;
    z) ZONE="$OPTARG" ;;
    m) INSTANCE="$OPTARG" ;;
    k) SSH_KEYFILE="$OPTARG" ;;
    T) OSLOGIN_KEY_TTL="$OPTARG" ;;
    t) USE_IAP=1 ;;
    c) CHECK_SSH=1 ;;
    K) SA_KEYFILE="$OPTARG" ;;
    A) SA_IMPERSONATE="$OPTARG" ;;
    E) ENABLE_APIS=1 ;;
    S) ENSURE_OSLOGIN_KEY=1 ;;
    G) GENERATE_KEY=1 ;;
    O) ENFORCE_INSTANCE_OSLOGIN=1 ;;
    *) echo "Usage: $0 [-p PROJECT] [-z ZONE] [-m INSTANCE] [-t] [-k KEYFILE] [-c] [-K SA_KEYFILE] [-A SA_ACCOUNT] [-E] [-S] [-G] [-O] [-T KEY_TTL]" >&2; exit 2 ;;
  esac
done

step() { echo; ylw "[doctor] $*"; }
fail() { red "[doctor] FAIL: $*"; exit 2; }
warn() { ylw "[doctor] WARN: $*"; }

step "Checking gcloud installation"
if ! ensure_gcloud_in_path || ! has gcloud; then
  fail "gcloud が見つかりません。'scripts/install_gcloud.sh' を実行して導入してください。"
fi
grn "$(gcloud version | head -n1)"

step "Checking authentication"
ACTIVE_ACCT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" || true)
if [[ -z "$ACTIVE_ACCT" ]]; then
  if [[ -n "$SA_KEYFILE" ]]; then
    ylw "アクティブアカウントなし → SA キーで有効化を試行: $SA_KEYFILE"
    gcloud auth activate-service-account --key-file "$SA_KEYFILE" >/dev/null
    ACTIVE_ACCT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" || true)
  fi
fi
if [[ -z "$ACTIVE_ACCT" ]]; then
  fail "認証情報が見つかりません。'gcloud auth login' または '-K <SA_KEYFILE>' を指定してください。"
fi
grn "Active account: $ACTIVE_ACCT"

# Common gcloud runner (impersonation if requested)
g() {
  if [[ -n "$SA_IMPERSONATE" ]]; then
    gcloud --impersonate-service-account="$SA_IMPERSONATE" "$@"
  else
    gcloud "$@"
  fi
}
if [[ -n "$SA_IMPERSONATE" ]]; then
  ylw "Impersonating service account: $SA_IMPERSONATE"
fi

step "Checking project / config"
if [[ -z "$PROJECT" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null || echo "")"
fi
if [[ -z "$PROJECT" ]]; then
  fail "GCP プロジェクト未設定。-p で指定するか 'gcloud config set project <PROJECT>' を実行してください。"
fi
grn "Project: $PROJECT"
grn "Zone:    $ZONE"

if [[ -z "$SSH_KEYFILE" ]]; then
  SSH_KEYFILE="$HOME/.ssh/gcp_oslogin_qr"
fi
PUBKEYFILE="$SSH_KEYFILE"
if [[ "$PUBKEYFILE" != *.pub ]]; then
  PUBKEYFILE="${SSH_KEYFILE}.pub"
fi

step "Checking required APIs"
REQUIRED_APIS=("compute.googleapis.com")
if [[ $USE_IAP -eq 1 ]]; then
  REQUIRED_APIS+=("iap.googleapis.com")
fi

for api in "${REQUIRED_APIS[@]}"; do
  if ! g services list --enabled --project "$PROJECT" --format="value(config.name)" | grep -q "^${api}$"; then
    if [[ $ENABLE_APIS -eq 1 ]]; then
      ylw "${api} を有効化します..."
      if g services enable "$api" --project "$PROJECT" >/dev/null; then
        grn "${api}: enabled"
      else
        fail "${api} の有効化に失敗しました。権限を確認してください（roles/serviceusage.serviceUsageAdmin など）。"
      fi
    else
      warn "${api} が有効化されていません。以下を実行してください:\n  gcloud services enable ${api} --project ${PROJECT}"
    fi
  else
    grn "${api}: enabled"
  fi
done

step "Checking instance existence"
if ! g compute instances describe "$INSTANCE" --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1; then
  fail "インスタンス '$INSTANCE' (zone=$ZONE) が見つかりません。ゾーン/名前/プロジェクトを確認してください。"
fi
grn "Instance: found ($INSTANCE)"

step "Checking instance metadata for OS Login"
INSTANCE_OSLOGIN=$(
  g compute instances describe "$INSTANCE" --zone "$ZONE" --project "$PROJECT" \
    --flatten="metadata.items[]" \
    --filter="metadata.items.key=enable-oslogin" \
    --format="value(metadata.items.value)" 2>/dev/null || true
)
INSTANCE_BLOCK_SSH=$(
  g compute instances describe "$INSTANCE" --zone "$ZONE" --project "$PROJECT" \
    --flatten="metadata.items[]" \
    --filter="metadata.items.key=block-project-ssh-keys" \
    --format="value(metadata.items.value)" 2>/dev/null || true
)

if [[ $ENFORCE_INSTANCE_OSLOGIN -eq 1 ]]; then
  ylw "instance metadata を OS Login 固定値へ更新します (enable-oslogin=TRUE, block-project-ssh-keys=TRUE)"
  if g compute instances add-metadata "$INSTANCE" --zone "$ZONE" --project "$PROJECT" \
    --metadata enable-oslogin=TRUE,block-project-ssh-keys=TRUE >/dev/null; then
    grn "Instance metadata updated for OS Login"
    INSTANCE_OSLOGIN="TRUE"
    INSTANCE_BLOCK_SSH="TRUE"
  else
    fail "instance metadata の更新に失敗しました。権限を確認してください。"
  fi
fi

INSTANCE_OSLOGIN_UPPER="$(upper "$INSTANCE_OSLOGIN")"
INSTANCE_BLOCK_SSH_UPPER="$(upper "$INSTANCE_BLOCK_SSH")"

if [[ -z "$INSTANCE_OSLOGIN" ]]; then
  warn "instance metadata に enable-oslogin が未設定です（project metadata を継承）。"
elif [[ "$INSTANCE_OSLOGIN_UPPER" != "TRUE" ]]; then
  warn "instance metadata enable-oslogin=${INSTANCE_OSLOGIN}。project 側 TRUE を上書きして SSH 不安定化の原因になります。-O で修復してください。"
else
  grn "Instance metadata enable-oslogin=TRUE"
fi

if [[ -z "$INSTANCE_BLOCK_SSH" ]]; then
  warn "instance metadata に block-project-ssh-keys が未設定です。OS Login 運用では TRUE 推奨です。"
elif [[ "$INSTANCE_BLOCK_SSH_UPPER" != "TRUE" ]]; then
  warn "instance metadata block-project-ssh-keys=${INSTANCE_BLOCK_SSH}。OS Login 一貫運用のため TRUE を推奨します。"
else
  grn "Instance metadata block-project-ssh-keys=TRUE"
fi

step "Checking OS Login profile"
if ! g compute os-login describe-profile --project "$PROJECT" >/dev/null 2>&1; then
  warn "OS Login プロファイルが取得できません。権限 'roles/compute.osLogin' が必要です。\n  SSH鍵が未登録の場合: gcloud compute os-login ssh-keys add --key-file ~/.ssh/<your_key>.pub --ttl 30d"
else
  POSIX_USER=$(g compute os-login describe-profile --format='value(posixAccounts[0].username)' --project "$PROJECT" 2>/dev/null || true)
  [[ -n "$POSIX_USER" ]] && grn "OS Login POSIX user: $POSIX_USER" || warn "OS Login POSIX ユーザ名を取得できませんでした。"
fi

step "Checking IAP requirement (if requested)"
if [[ $USE_IAP -eq 1 ]]; then
  ylw "IAP トンネルを使用します。IAM に roles/iap.tunnelResourceAccessor が必要です。"
fi

if [[ $CHECK_SSH -eq 1 ]]; then
  step "Attempting SSH connectivity test"
  ssh_args=("--project" "$PROJECT" "--zone" "$ZONE")
  [[ -n "$SA_IMPERSONATE" ]] && ssh_args+=("--impersonate-service-account=$SA_IMPERSONATE")
  [[ $USE_IAP -eq 1 ]] && ssh_args+=("--tunnel-through-iap")
  [[ -n "$SSH_KEYFILE" ]] && ssh_args+=("--ssh-key-file" "$SSH_KEYFILE")
  if gcloud compute ssh "$INSTANCE" "${ssh_args[@]}" --command "echo '[vm] hello'" >/dev/null 2>&1; then
    grn "SSH connectivity OK"
  else
    fail "SSH 接続に失敗しました。OS Login 有効化/IAP 権限/SSH鍵/ファイアウォール設定を確認してください。"
  fi
fi

if [[ $ENSURE_OSLOGIN_KEY -eq 1 ]]; then
  step "Ensuring OS Login public key registered"
  if [[ ! -f "$PUBKEYFILE" ]]; then
    if [[ $GENERATE_KEY -eq 1 ]]; then
      ylw "公開鍵が見つかりません。新規に鍵を作成します: $SSH_KEYFILE"
      ssh-keygen -t ed25519 -f "$SSH_KEYFILE" -N '' -C "oslogin-$PROJECT" >/dev/null
    else
      fail "公開鍵が見つかりません: $PUBKEYFILE（'-G' で鍵を生成します）"
    fi
  fi
  KEY_REGISTER_ARGS=(compute os-login ssh-keys add --key-file "$PUBKEYFILE")
  TTL_NORM="$(lower "$OSLOGIN_KEY_TTL")"
  if [[ -n "$TTL_NORM" && "$TTL_NORM" != "none" ]]; then
    KEY_REGISTER_ARGS+=(--ttl "$OSLOGIN_KEY_TTL")
  fi
  if g "${KEY_REGISTER_ARGS[@]}" >/dev/null 2>&1; then
    grn "OS Login key registered for principal"
    if [[ -n "$TTL_NORM" && "$TTL_NORM" != "none" ]]; then
      grn "OS Login key TTL: $OSLOGIN_KEY_TTL"
    else
      grn "OS Login key TTL: none (non-expiring)"
    fi
  else
    fail "OS Login への鍵登録に失敗しました。権限（roles/compute.osLogin）と IAP/プロジェクト設定を確認してください。"
  fi
fi

echo
grn "[doctor] All essential checks passed."
