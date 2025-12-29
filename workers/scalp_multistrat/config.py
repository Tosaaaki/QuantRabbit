from __future__ import annotations

import os

POCKET = "scalp"
LOOP_INTERVAL_SEC = float(os.getenv("SCALP_MULTI_LOOP_INTERVAL_SEC", "6.0"))
ENABLED = os.getenv("SCALP_MULTI_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
LOG_PREFIX = "[ScalpMulti]"

CONFIDENCE_FLOOR = 30
CONFIDENCE_CEIL = 90
MIN_UNITS = int(os.getenv("SCALP_MULTI_MIN_UNITS", "2000"))
BASE_ENTRY_UNITS = int(os.getenv("SCALP_MULTI_BASE_UNITS", "2000"))
MAX_MARGIN_USAGE = float(os.getenv("SCALP_MULTI_MAX_MARGIN_USAGE", "0.82"))

CAP_MIN = float(os.getenv("SCALP_MULTI_CAP_MIN", "0.1"))
CAP_MAX = float(os.getenv("SCALP_MULTI_CAP_MAX", "0.25"))

# 新規エントリーのクールダウン（秒）
COOLDOWN_SEC = float(os.getenv("SCALP_MULTI_COOLDOWN_SEC", "90"))
# 同ポケットの最大同時建玉数
MAX_OPEN_TRADES = int(os.getenv("SCALP_MULTI_MAX_OPEN_TRADES", "2"))
# 経済指標ブロック（分）。未設定で AttributeError とならないようデフォルトを明示。
NEWS_BLOCK_MINUTES = float(os.getenv("SCALP_MULTI_NEWS_BLOCK_MINUTES", "0"))
