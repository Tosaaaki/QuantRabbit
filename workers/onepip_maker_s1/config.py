"""Configuration for the one-pip maker (S1) worker."""

from __future__ import annotations

import os
from pathlib import Path

PIP_VALUE = 0.01


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _float(key: str, default: float, minimum: float | None = None) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if minimum is not None and value < minimum:
        return default
    return value


def _int(key: str, default: int, minimum: int | None = None) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = int(float(raw))
    except ValueError:
        return default
    if minimum is not None and value < minimum:
        return default
    return value


LOG_PREFIX = "[ONEPIP-MAKER]"
# 強制有効化（環境変数に依存しない）
ENABLED = True
SHADOW_MODE = False
SHADOW_LOG_ALL = _bool("ONEPIP_MAKER_S1_SHADOW_LOG_ALL", False)
LOOP_INTERVAL_SEC = max(0.1, _float("ONEPIP_MAKER_S1_LOOP_INTERVAL_SEC", 0.25))

MAX_SPREAD_PIPS = _float("ONEPIP_MAKER_S1_MAX_SPREAD_PIPS", 1.0, minimum=0.05)
MAX_COST_PIPS = _float("ONEPIP_MAKER_S1_MAX_COST_PIPS", 0.45, minimum=0.1)
COST_WINDOW_SEC = _float("ONEPIP_MAKER_S1_COST_WINDOW_SEC", 480.0, minimum=60.0)

QUEUE_IMBALANCE_MIN = _float("ONEPIP_MAKER_S1_QUEUE_IMBALANCE_MIN", 0.35, minimum=0.05)
QUEUE_IMBALANCE_DEPTH = _int("ONEPIP_MAKER_S1_QUEUE_IMBALANCE_DEPTH", 1, minimum=1)
MAX_SNAPSHOT_AGE_MS = _float("ONEPIP_MAKER_S1_MAX_SNAPSHOT_AGE_MS", 400.0, minimum=10.0)

# 実スプレッドへの適応: ベースライン（移動窓の平均/95%）でゲートする
BASELINE_SPREAD_P50_MAX = _float("ONEPIP_BASELINE_SPREAD_P50_MAX", 0.22)
BASELINE_SPREAD_P95_MAX = _float("ONEPIP_BASELINE_SPREAD_P95_MAX", 0.30)

MICRO_WINDOW_SEC = _float("ONEPIP_MAKER_S1_MICRO_WINDOW_SEC", 1.2, minimum=0.2)
MICRO_DRIFT_MAX_PIPS = _float("ONEPIP_MAKER_S1_MICRO_DRIFT_MAX_PIPS", 0.35, minimum=0.05)
MICRO_INSTANT_MOVE_MAX_PIPS = _float("ONEPIP_MAKER_S1_MICRO_INSTANT_MOVE_MAX_PIPS", 0.6, minimum=0.05)

TTL_MS = _float("ONEPIP_MAKER_S1_TTL_MS", 700.0, minimum=100.0)
COOLDOWN_AFTER_CANCEL_MS = _float("ONEPIP_MAKER_S1_CANCEL_COOLDOWN_MS", 1800.0, minimum=100.0)
SESSION_RESET_SEC = _float("ONEPIP_MAKER_S1_SESSION_RESET_SEC", 1200.0, minimum=60.0)

ENTRY_UNITS = _int("ONEPIP_MAKER_S1_ENTRY_UNITS", 4000, minimum=800)
SL_PIPS = _float("ONEPIP_MAKER_S1_SL_PIPS", 1.15, minimum=0.2)
TP_PIPS = _float("ONEPIP_MAKER_S1_TP_PIPS", 1.25, minimum=0.2)
POCKET = os.getenv("ONEPIP_MAKER_S1_POCKET", "micro").strip() or "micro"
MIN_UNITS = _int("ONEPIP_MAKER_S1_MIN_UNITS", 1200, minimum=200)
MAX_UNITS = _int("ONEPIP_MAKER_S1_MAX_UNITS", 7000, minimum=1000)
ACCOUNT_REFRESH_SEC = _float("ONEPIP_MAKER_S1_ACCOUNT_REFRESH_SEC", 90.0, minimum=10.0)
MIN_COST_SAMPLES = _int("ONEPIP_MAKER_S1_MIN_COST_SAMPLES", 50, minimum=1)
MAX_SNAPSHOT_LATENCY_MS = _float("ONEPIP_MAKER_S1_MAX_LATENCY_MS", 400.0, minimum=5.0)
MIN_TOP_LIQUIDITY = _float("ONEPIP_MAKER_S1_MIN_TOP_LIQUIDITY", 400000.0, minimum=50000.0)
DEPTH_LEVELS = _int("ONEPIP_MAKER_S1_DEPTH_LEVELS", 1, minimum=1)

# Margin health guard: when free margin ratio is below this, block entries
MARGIN_FREE_MIN = _float("ONEPIP_MAKER_S1_MARGIN_FREE_MIN", 0.025, minimum=0.0)

# 実コストに合わせて TP を自動引き上げ（net + MIN_NET_GAIN_PIPS を確保）
MIN_NET_GAIN_PIPS = _float("ONEPIP_MAKER_S1_MIN_NET_GAIN_PIPS", 1.0, minimum=0.2)
TP_PIPS_MAX = _float("ONEPIP_MAKER_S1_TP_PIPS_MAX", 3.0, minimum=0.8)

NEWS_BLOCK_MINUTES = _float("ONEPIP_MAKER_S1_NEWS_BLOCK_MINUTES", 20.0, minimum=0.0)
NEWS_BLOCK_MIN_IMPACT = _int("ONEPIP_MAKER_S1_NEWS_BLOCK_MIN_IMPACT", 2, minimum=1)

SHADOW_LOG_PATH = Path(
    os.getenv("ONEPIP_MAKER_S1_SHADOW_LOG_PATH", "logs/onepip_maker_s1_shadow.jsonl")
)
SHADOW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

COST_BOOTSTRAP_FILES = _int("ONEPIP_MAKER_S1_COST_BOOTSTRAP_FILES", 2, minimum=0)
COST_BOOTSTRAP_LINES = _int("ONEPIP_MAKER_S1_COST_BOOTSTRAP_LINES", 300, minimum=10)

RANGE_ONLY = _bool("ONEPIP_MAKER_S1_RANGE_ONLY", True)
