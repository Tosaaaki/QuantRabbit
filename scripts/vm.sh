#!/usr/bin/env bash
set -euo pipefail

# QuantRabbit VM helper (no default gcloud config required)
#
# Common flags (required unless noted):
#   -p <PROJECT>         GCP project ID
#   -z <ZONE>            GCE zone (e.g., asia-northeast1-a)
#   -m <INSTANCE>        GCE VM instance name
#   -A <ACCOUNT>         gcloud account/email to use (optional; must be authenticated already)
#   -k <KEYFILE>         SSH private key for OS Login (optional)
#   -t                   Use IAP tunnel (no external IP)
#   -d <REMOTE_DIR>      Remote repo dir (default: ~/QuantRabbit)
#
# Subcommands:
#   deploy [-b BRANCH] [-i] [--restart SERVICE]
#     -b BRANCH          Branch to deploy (default: current local branch name)
#     -i                 Install requirements in remote venv (if venv exists in REMOTE_DIR/.venv)
#     --restart SERVICE  Restart given systemd service after pull
#
#   exec -- <REMOTE CMD>
#     Run arbitrary command on VM
#
#   tail [-s SERVICE] [-n N]
#     Tail systemd logs for SERVICE (default: quantrabbit.service), last N lines (default: 300)
#
#   pull-logs [-r REMOTE_LOG_DIR] [-o OUT_DIR] [--pattern GLOB]
#     Copy logs/ artifacts from VM (default REMOTE_LOG_DIR: ~/QuantRabbit/logs, OUT_DIR: ./remote_logs)
#
#   scp [--to-remote|--from-remote] <SRC> <DST>
#     Wrapper around gcloud compute scp
#
#   sql [-f DB_PATH] [-q QUERY]
#     Run sqlite3 query on remote DB (default: ~/QuantRabbit/logs/trades.db)
#
#   serial
#     Print VM serial console output
#
# Examples:
#   scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm deploy -b main -i -t
#   scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm tail -s quantrabbit.service -t
#   scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm pull-logs -o ./remote_logs -t
#   scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm sql -q "SELECT COUNT(*) FROM trades;" -t

PROJECT=""
ZONE=""
INSTANCE=""
ACCOUNT=""
KEYFILE=""
USE_IAP=""
REMOTE_DIR="~/QuantRabbit"

die() { echo "[vm.sh] $*" >&2; exit 1; }

need_bin() { command -v "$1" >/dev/null 2>&1 || die "Missing required binary: $1"; }

print_usage() {
  sed -n '1,80p' "$0" | sed -n '1,70p' | sed -n '1,70p' >&2
}

while getopts ":p:z:m:A:k:td:" opt; do
  case "$opt" in
    p) PROJECT="$OPTARG" ;;
    z) ZONE="$OPTARG" ;;
    m) INSTANCE="$OPTARG" ;;
    A) ACCOUNT="$OPTARG" ;;
    k) KEYFILE="$OPTARG" ;;
    t) USE_IAP=1 ;;
    d) REMOTE_DIR="$OPTARG" ;;
    :) die "Option -$OPTARG requires an argument" ;;
    \?) die "Unknown option: -$OPTARG" ;;
  esac
done
shift $((OPTIND-1))

SUBCMD="${1-}"; shift || true

[[ -n "$PROJECT" && -n "$ZONE" && -n "$INSTANCE" ]] || { print_usage; die "-p, -z, -m are required"; }

need_bin gcloud

GCLOUD_BASE=(gcloud)
if [[ -n "$ACCOUNT" ]]; then
  GCLOUD_BASE+=(--account="$ACCOUNT")
fi

ssh_base() {
  local -a cmd=("${GCLOUD_BASE[@]}" compute ssh "$INSTANCE" --project "$PROJECT" --zone "$ZONE")
  if [[ -n "$USE_IAP" ]]; then
    cmd+=(--tunnel-through-iap)
  fi
  if [[ -n "$KEYFILE" ]]; then
    cmd+=(--ssh-key-file "$KEYFILE")
  fi
  printf '%s\n' "${cmd[@]}"
}

scp_base() {
  local -a cmd=("${GCLOUD_BASE[@]}" compute scp --project "$PROJECT" --zone "$ZONE")
  if [[ -n "$USE_IAP" ]]; then
    cmd+=(--tunnel-through-iap)
  fi
  if [[ -n "$KEYFILE" ]]; then
    cmd+=(--ssh-key-file "$KEYFILE")
  fi
  printf '%s\n' "${cmd[@]}"
}

remote_bash() {
  local cmd_str="$1"
  local b64
  b64=$(printf '%s' "$cmd_str" | base64 | tr -d '\n')
  # Decode within a login shell to preserve quoting and run multi-line blocks
  $(ssh_base) --command "bash -lc \"\$(echo $b64 | base64 -d)\""
}

case "$SUBCMD" in
  deploy)
    BRANCH=""
    INSTALL=""
    RESTART_SVC=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        -b|--branch) BRANCH="$2"; shift 2 ;;
        -i|--install) INSTALL=1; shift ;;
        --restart) RESTART_SVC="$2"; shift 2 ;;
        *) die "Unknown deploy option: $1" ;;
      esac
    done
    if [[ -z "$BRANCH" ]]; then
      if command -v git >/dev/null 2>&1; then
        BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
      else
        BRANCH="main"
      fi
    fi
    echo "[vm.sh] Deploying branch '$BRANCH' to $INSTANCE ($PROJECT/$ZONE)"
    
    # Build remote command
    read -r -d '' REMOTE <<EOF || true
set -euo pipefail
cd $REMOTE_DIR
echo "[remote] dir=$(pwd) branch=$BRANCH"
if [ -d .git ]; then
  git fetch --all -q || true
  git checkout -q "$BRANCH" || git checkout -b "$BRANCH" "origin/$BRANCH" || true
  git pull --ff-only -q || true
else
  echo "[remote] No git repo at $REMOTE_DIR" >&2
fi
if [ -n "${INSTALL-}" ]; then
  if [ -d .venv ]; then
    echo "[remote] Installing requirements into .venv"
    source .venv/bin/activate
    pip install -r requirements.txt
  else
    echo "[remote] .venv not found; skipping -i"
  fi
fi
if [ -f ./startup.sh ]; then
  echo "[remote] Running startup.sh"
  bash ./startup.sh || true
fi
if [ -n "${RESTART_SVC-}" ]; then
  echo "[remote] Restarting service: $RESTART_SVC"
  sudo systemctl daemon-reload || true
  sudo systemctl restart "$RESTART_SVC" || true
  sudo systemctl status --no-pager "$RESTART_SVC" -l || true
fi
EOF
    remote_bash "$REMOTE"
    ;;

  exec)
    # everything after -- is the command
    if [[ "${1-}" == "--" ]]; then shift; fi
    [[ $# -gt 0 ]] || die "exec requires a remote command after --"
    remote_bash "$*"
    ;;

  tail)
    SERVICE="quantrabbit.service"
    N=300
    GREP=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        -s|--service) SERVICE="$2"; shift 2 ;;
        -n) N="$2"; shift 2 ;;
        --grep) GREP="$2"; shift 2 ;;
        *) die "Unknown tail option: $1" ;;
      esac
    done
    if [[ -n "$GREP" ]]; then
      remote_bash "journalctl -u '$SERVICE' -n $N -f --output=short-iso | grep -E $GREP"
    else
      remote_bash "journalctl -u '$SERVICE' -n $N -f --output=short-iso"
    fi
    ;;

  pull-logs)
    REMOTE_LOG_DIR="${REMOTE_DIR}/logs"
    OUT_DIR="./remote_logs"
    PATTERN="*.db"
    while [[ $# -gt 0 ]]; do
      case "$1" in
        -r) REMOTE_LOG_DIR="$2"; shift 2 ;;
        -o) OUT_DIR="$2"; shift 2 ;;
        --pattern) PATTERN="$2"; shift 2 ;;
        *) die "Unknown pull-logs option: $1" ;;
      esac
    done
    mkdir -p "$OUT_DIR"
    TS=$(date +%Y%m%d-%H%M%S)
    DEST="$OUT_DIR/$INSTANCE-$TS"
    mkdir -p "$DEST"
    echo "[vm.sh] Copying $REMOTE_LOG_DIR/$PATTERN -> $DEST"
    # shellcheck disable=SC2046
    $(scp_base) --recurse "$INSTANCE:$REMOTE_LOG_DIR/$PATTERN" "$DEST/" 2>/dev/null || true
    $(scp_base) --recurse "$INSTANCE:$REMOTE_LOG_DIR/replay" "$DEST/replay" 2>/dev/null || true
    echo "[vm.sh] Done. Files in: $DEST"
    ;;

  scp)
    DIRECTION="from"
    if [[ "${1-}" == "--to-remote" ]]; then DIRECTION="to"; shift; fi
    if [[ "${1-}" == "--from-remote" ]]; then DIRECTION="from"; shift; fi
    [[ $# -ge 2 ]] || die "scp requires SRC and DST"
    SRC="$1"; DST="$2"
    if [[ "$DIRECTION" == "to" ]]; then
      $(scp_base) "$SRC" "$INSTANCE:$DST"
    else
      $(scp_base) "$INSTANCE:$SRC" "$DST"
    fi
    ;;

  sql)
    DB_PATH="${REMOTE_DIR}/logs/trades.db"
    QUERY=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        -f) DB_PATH="$2"; shift 2 ;;
        -q) QUERY="$2"; shift 2 ;;
        *) die "Unknown sql option: $1" ;;
      esac
    done
    [[ -n "$QUERY" ]] || QUERY="SELECT DATE(close_time), COUNT(*), ROUND(SUM(pl_pips),2) FROM trades WHERE DATE(close_time)=DATE('now') GROUP BY 1;"
    remote_bash "sqlite3 '$DB_PATH' \"$QUERY\""
    ;;

  serial)
    # shellcheck disable=SC2046
    ${GCLOUD_BASE[@]} compute instances get-serial-port-output "$INSTANCE" --project "$PROJECT" --zone "$ZONE"
    ;;

  *)
    print_usage
    die "Unknown subcommand: ${SUBCMD:-<none>}"
    ;;
esac
