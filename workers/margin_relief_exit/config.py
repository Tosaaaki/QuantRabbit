import os

ENV_PREFIX = "MARGIN_RELIEF"


def _bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _csv_set(raw: str | None, default: set[str]) -> set[str]:
    if raw is None:
        return default
    items = {t.strip().lower() for t in raw.replace(";", ",").split(",") if t.strip()}
    return items or default


ENABLED = _bool_env("MARGIN_RELIEF_EXIT_ENABLED", True)
LOOP_INTERVAL_SEC = _float_env("MARGIN_RELIEF_EXIT_INTERVAL_SEC", 10.0)
COOLDOWN_SEC = _float_env("MARGIN_RELIEF_EXIT_COOLDOWN_SEC", 30.0)
MAX_ACTIONS_PER_RUN = max(1, _int_env("MARGIN_RELIEF_EXIT_MAX_ACTIONS", 1))

TRIGGER_MARGIN_USAGE = _float_env("MARGIN_RELIEF_TRIGGER_MARGIN_USAGE", 0.93)
TRIGGER_FREE_MARGIN_RATIO = _float_env("MARGIN_RELIEF_TRIGGER_FREE_MARGIN_RATIO", 0.06)
RELEASE_MARGIN_USAGE = _float_env("MARGIN_RELIEF_RELEASE_MARGIN_USAGE", 0.88)
RELEASE_FREE_MARGIN_RATIO = _float_env("MARGIN_RELIEF_RELEASE_FREE_MARGIN_RATIO", 0.10)

BIAS_THRESHOLD = _float_env("MARGIN_RELIEF_BIAS_THRESHOLD", 0.60)
MIN_GROSS_UNITS = _int_env("MARGIN_RELIEF_MIN_GROSS_UNITS", 20000)
MIN_NET_UNITS = _int_env("MARGIN_RELIEF_MIN_NET_UNITS", 15000)

MIN_HOLD_SEC = _float_env("MARGIN_RELIEF_MIN_HOLD_SEC", 2400.0)
MAX_HOLD_SEC = _float_env("MARGIN_RELIEF_MAX_HOLD_SEC", 7200.0)
RANGE_MIN_HOLD_SEC = _float_env("MARGIN_RELIEF_RANGE_MIN_HOLD_SEC", 1800.0)
RANGE_MAX_HOLD_SEC = _float_env("MARGIN_RELIEF_RANGE_MAX_HOLD_SEC", 3600.0)

LOSS_TRIGGER_PIPS = _float_env("MARGIN_RELIEF_LOSS_TRIGGER_PIPS", -0.4)
RANGE_LOSS_TRIGGER_PIPS = _float_env("MARGIN_RELIEF_RANGE_LOSS_TRIGGER_PIPS", -0.3)

PARTIAL_FRACTION = _float_env("MARGIN_RELIEF_PARTIAL_FRACTION", 0.25)
RANGE_PARTIAL_FRACTION = _float_env("MARGIN_RELIEF_RANGE_PARTIAL_FRACTION", 0.35)
PARTIAL_MIN_UNITS = _int_env("MARGIN_RELIEF_PARTIAL_MIN_UNITS", 500)
PARTIAL_MIN_REMAIN = _int_env("MARGIN_RELIEF_PARTIAL_MIN_REMAIN", 500)

TECH_EXIT_ENABLED = _bool_env("MARGIN_RELIEF_TECH_EXIT_ENABLED", True)
TECH_EXIT_REQUIRED = _bool_env("MARGIN_RELIEF_TECH_EXIT_REQUIRED", True)
TECH_EXIT_MIN_HOLD_SEC = _float_env("MARGIN_RELIEF_TECH_EXIT_MIN_HOLD_SEC", 1200.0)

POCKETS = _csv_set(
    os.getenv("MARGIN_RELIEF_POCKETS"),
    {"micro", "macro", "scalp"},
)

START_TS_PATH = os.getenv("MARGIN_RELIEF_START_TS_PATH", "logs/margin_relief_start_ts.txt")
STATE_PATH = os.getenv("MARGIN_RELIEF_STATE_PATH", "logs/margin_relief_state.json")
STATE_TTL_SEC = _float_env("MARGIN_RELIEF_STATE_TTL_SEC", 21600.0)

LOG_PREFIX = "[MARGIN_RELIEF]"
