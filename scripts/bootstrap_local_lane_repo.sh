#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_TARGET="${HOME}/Documents/App/QuantRabbitLocalLane"
DEFAULT_BOT_SRC="/tmp/codex_long_autotrade.py"
DEFAULT_ENV_SOURCE="${ROOT_DIR}/ops/env/local-llm-lane.env"
BOOTSTRAP_VERSION="2026-03-03.1"

TARGET_DIR="${DEFAULT_TARGET}"
BOT_SRC="${DEFAULT_BOT_SRC}"
ENV_SOURCE="${DEFAULT_ENV_SOURCE}"
MODE="update" # init|update|sync
FORCE=0
DRY_RUN=0
INIT_GIT=0

usage() {
  cat <<'EOF'
Usage:
  scripts/bootstrap_local_lane_repo.sh [--target DIR] [--bot-src FILE] [--env-source FILE]
                                      [--mode init|update|sync] [--force] [--dry-run] [--init-git]

Options:
  --target DIR      Destination directory for local lane repo.
  --bot-src FILE    Source trading bot script to copy (default: /tmp/codex_long_autotrade.py).
  --env-source FILE Source env template (default: ops/env/local-llm-lane.env).
  --mode MODE       init|update|sync (default: update).
  --force           Overwrite managed files when content differs.
  --dry-run         Show planned changes only.
  --init-git        Run `git init` in target repo if not already a git repo.
  -h, --help        Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET_DIR="${2:-}"
      shift 2
      ;;
    --bot-src)
      BOT_SRC="${2:-}"
      shift 2
      ;;
    --env-source)
      ENV_SOURCE="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --init-git)
      INIT_GIT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "${MODE}" != "init" && "${MODE}" != "update" && "${MODE}" != "sync" ]]; then
  echo "[error] --mode must be init|update|sync" >&2
  exit 2
fi

TARGET_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${TARGET_DIR}")"
BOT_SRC="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${BOT_SRC}")"
ENV_SOURCE="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${ENV_SOURCE}")"

if [[ "${TARGET_DIR}" == "${ROOT_DIR}" ]]; then
  echo "[error] target repo must be different from main repo: ${ROOT_DIR}" >&2
  exit 2
fi
if [[ ! -f "${BOT_SRC}" ]]; then
  echo "[error] bot source not found: ${BOT_SRC}" >&2
  exit 2
fi
if [[ ! -f "${ENV_SOURCE}" ]]; then
  echo "[error] env source not found: ${ENV_SOURCE}" >&2
  exit 2
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${TARGET_DIR}"
fi

if [[ "${MODE}" == "init" && -f "${TARGET_DIR}/.bootstrap/local_lane_manifest.json" && "${FORCE}" != "1" ]]; then
  echo "[error] target already looks bootstrapped: ${TARGET_DIR}" >&2
  echo "Use --mode update, or re-run with --force if you want to overwrite managed files." >&2
  exit 2
fi

apply_file() {
  local rel="$1"
  local src="$2"
  local dst="${TARGET_DIR}/${rel}"
  local dst_new="${dst}.bootstrap.new"

  if [[ "${MODE}" == "sync" && -f "${dst}" ]]; then
    echo "[sync-skip] ${rel}"
    return 0
  fi

  if [[ -f "${dst}" ]]; then
    if cmp -s "${src}" "${dst}"; then
      echo "[skip] ${rel}"
      return 0
    fi
    if [[ "${FORCE}" != "1" ]]; then
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[plan-conflict] ${rel} -> ${rel}.bootstrap.new"
        return 0
      fi
      mkdir -p "$(dirname "${dst_new}")"
      cp "${src}" "${dst_new}"
      echo "[conflict] ${rel} differs; wrote ${rel}.bootstrap.new (re-run with --force to overwrite)"
      return 0
    fi
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[plan-write] ${rel}"
    return 0
  fi

  mkdir -p "$(dirname "${dst}")"
  cp "${src}" "${dst}"
  echo "[write] ${rel}"
}

write_content() {
  local rel="$1"
  local tmp
  tmp="$(mktemp)"
  cat >"${tmp}"
  apply_file "${rel}" "${tmp}"
  rm -f "${tmp}"
}

tmp_bot="$(mktemp)"
awk '
  /^REPO = "/ { skip=2; next }
  skip > 0 { skip--; next }
  /^from utils\.secrets import get_secret/ {
    print "from local_secrets import get_secret  # local lane secret loader"
    next
  }
  { print }
' "${BOT_SRC}" > "${tmp_bot}"

if ! rg -q "^from local_secrets import get_secret" "${tmp_bot}"; then
  echo "[error] failed to patch bot import path" >&2
  rm -f "${tmp_bot}"
  exit 3
fi

write_content "README.md" <<EOF
# QuantRabbit Local LLM Lane

This repository is an isolated local trading lane for real USD/JPY trading.

## Purpose
- Keep VM production repo and local lane git history separated.
- Run local lane with local LLM gate (Ollama) and compare performance against VM lane.

## Setup
1. Create virtualenv:
   \`\`\`bash
   python3 -m venv .venv
   . .venv/bin/activate
   pip install -r requirements.txt
   \`\`\`
2. Create \`.env\`:
   \`\`\`bash
   cp .env.example .env
   \`\`\`
3. Start local lane:
   \`\`\`bash
   bash scripts/run_local_lane.sh
   \`\`\`

## Output Contract
- Trade log: \`state/codex_long_autotrade.log\` (JSONL).
- Required events:
  - \`trade_opened\`
  - \`trade_closed\` with \`trade_id\` or \`client_id\`, and \`pl\` + \`pnl_pips\` (or \`pl_pips\`).

Main VM repo can consume this file via:
- \`scripts/watch_lane_winner_external.sh\`
- \`LOCAL_LOG=/ABS/PATH/QuantRabbitLocalLane/state/codex_long_autotrade.log\`
EOF

write_content ".gitignore" <<'EOF'
.env
.venv/
__pycache__/
*.pyc
state/*.log
state/*.json
*.bootstrap.new
EOF

write_content "requirements.txt" <<'EOF'
requests>=2.31.0
EOF

write_content "scripts/run_local_lane.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "${ROOT_DIR}/.env"
  set +a
fi

mkdir -p "${ROOT_DIR}/state"
export CODEX_LOG_FILE="${CODEX_LOG_FILE:-${ROOT_DIR}/state/codex_long_autotrade.log}"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PY_BIN="${ROOT_DIR}/.venv/bin/python"
else
  PY_BIN="${PY_BIN:-python3}"
fi

exec "${PY_BIN}" "${ROOT_DIR}/bot/codex_long_autotrade.py" "$@"
EOF

write_content "scripts/validate_local_lane.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PY_BIN="${ROOT_DIR}/.venv/bin/python"
else
  PY_BIN="${PY_BIN:-python3}"
fi

for f in "${ROOT_DIR}/bot/codex_long_autotrade.py" "${ROOT_DIR}/bot/local_secrets.py" "${ROOT_DIR}/scripts/run_local_lane.sh"; do
  if [[ ! -f "${f}" ]]; then
    echo "[error] missing file: ${f}" >&2
    exit 1
  fi
done

"${PY_BIN}" -m py_compile "${ROOT_DIR}/bot/codex_long_autotrade.py" "${ROOT_DIR}/bot/local_secrets.py"
echo "[ok] compile passed"
EOF

write_content "bot/local_secrets.py" <<'EOF'
from __future__ import annotations

import os
from pathlib import Path

ENV_MAP = {
    "oanda_token": "OANDA_TOKEN",
    "oanda_account_id": "OANDA_ACCOUNT",
    "oanda_account": "OANDA_ACCOUNT",
    "oanda_practice": "OANDA_PRACTICE",
}


def _load_dotenv(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key:
            data[key] = val
    return data


def get_secret(key: str) -> str:
    dotenv = _load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    names = [key, key.upper()]
    mapped = ENV_MAP.get(key)
    if mapped:
        names.append(mapped)
    for name in names:
        v = os.environ.get(name)
        if v:
            return v
    for name in names:
        v = dotenv.get(name)
        if v:
            return v
    raise KeyError(f"Secret '{key}' not found in env/.env")
EOF

write_content ".env.example" <<EOF
# Required OANDA credentials (do not commit real values)
OANDA_TOKEN=
OANDA_ACCOUNT=
OANDA_PRACTICE=false

# Local lane identity
CODEX_TRADE_PREFIX=codexlhf
CODEX_TRADE_TAG=codex_bi_hf
CODEX_LOG_FILE=\${PWD}/state/codex_long_autotrade.log

# Local LLM lane baseline
$(cat "${ENV_SOURCE}")
EOF

apply_file "bot/codex_long_autotrade.py" "${tmp_bot}"
rm -f "${tmp_bot}"

write_content ".bootstrap/local_lane_manifest.json" <<EOF
{
  "bootstrap_version": "${BOOTSTRAP_VERSION}",
  "source_bot": "${BOT_SRC}",
  "source_env": "${ENV_SOURCE}",
  "mode": "${MODE}",
  "managed_files": [
    "README.md",
    ".gitignore",
    "requirements.txt",
    ".env.example",
    "bot/codex_long_autotrade.py",
    "bot/local_secrets.py",
    "scripts/run_local_lane.sh",
    "scripts/validate_local_lane.sh"
  ]
}
EOF

if [[ "${DRY_RUN}" != "1" ]]; then
  chmod +x \
    "${TARGET_DIR}/bot/codex_long_autotrade.py" \
    "${TARGET_DIR}/scripts/run_local_lane.sh" \
    "${TARGET_DIR}/scripts/validate_local_lane.sh" || true
fi

if [[ "${INIT_GIT}" == "1" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[plan] git init ${TARGET_DIR}"
  elif [[ ! -d "${TARGET_DIR}/.git" ]]; then
    git init "${TARGET_DIR}" >/dev/null
    echo "[write] initialized git repo: ${TARGET_DIR}"
  else
    echo "[skip] git already initialized: ${TARGET_DIR}"
  fi
fi

echo "[done] bootstrap_local_lane_repo target=${TARGET_DIR} mode=${MODE} force=${FORCE} dry_run=${DRY_RUN}"
