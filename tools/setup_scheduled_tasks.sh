#!/usr/bin/env bash
# Repair Claude compatibility task prompt links.
#
# Live routine operation is driven by Codex automations in ~/.codex/automations.
# This script only keeps Claude scheduled-task compatibility assets pointed at
# the canonical repo prompts; it does not install old inline prompts or enable
# disabled tasks.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASKS_DIR="${HOME}/.claude/scheduled-tasks"

echo "=== QuantRabbit scheduled-task compatibility repair ==="
echo "Repo: ${REPO_DIR}"
echo "Claude task dir: ${TASKS_DIR}"
echo

while IFS='|' read -r task rel_path; do
  [[ -n "${task}" ]] || continue

  canonical="${REPO_DIR}/${rel_path}"
  if [[ ! -f "${canonical}" ]]; then
    echo "[MISSING] ${task}: ${canonical}"
    exit 1
  fi

  task_dir="${TASKS_DIR}/${task}"
  if [[ ! -d "${task_dir}" && -d "${task_dir}.DISABLED" ]]; then
    task_dir="${task_dir}.DISABLED"
  fi
  mkdir -p "${task_dir}"

  ln -sfn "${canonical}" "${task_dir}/SKILL.md"
  rm -f "${task_dir}/SKILL_ja.md"
  echo "[OK] ${task} -> ${canonical}"
done <<'TASKS'
trader|docs/SKILL_trader.md
daily-review|docs/SKILL_daily-review.md
quality-audit|docs/SKILL_quality-audit.md
inventory-director|docs/SKILL_inventory-director.md
range-bot|docs/SKILL_range-bot.md
bot-trade-manager|docs/SKILL_bot-trade-manager.md
daily-performance-report|docs/SKILL_daily-performance-report.md
daily-slack-summary|docs/SKILL_daily-slack-summary.md
intraday-pl-update|docs/SKILL_intraday-pl-update.md
TASKS

echo
python3 "${REPO_DIR}/tools/check_task_sync.py"
