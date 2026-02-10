from typing import Optional

from utils.env_utils import env_bool, env_float, env_get


def _strategy_env_bool(name: str, default: bool, *, env_prefix: Optional[str]) -> bool:
    return env_bool(
        name,
        default,
        prefix=env_prefix,
        allow_global_fallback=False,
    )


def _strategy_env_float(name: str, default: float, *, env_prefix: Optional[str]) -> float:
    return env_float(
        name,
        default,
        prefix=env_prefix,
        allow_global_fallback=False,
    )


def _strategy_env_get(name: str, default: Optional[str], *, env_prefix: Optional[str]) -> Optional[str]:
    return env_get(
        name,
        default,
        prefix=env_prefix,
        allow_global_fallback=False,
    )


def scale_base_units(
    base_units: int,
    *,
    equity: float,
    ref_equity: Optional[float] = None,
    min_units: Optional[int] = None,
    max_units: Optional[int] = None,
    env_prefix: Optional[str] = None,
) -> int:
    """Scale base entry units by equity to allow compounding.

    Controlled via env:
    - BASE_UNITS_EQUITY_SCALE_ENABLED (default: 1)
    - BASE_UNITS_EQUITY_REF (default: 1000000)
    - BASE_UNITS_EQUITY_SCALE_MIN (default: 1.0)
    - BASE_UNITS_EQUITY_SCALE_MAX (optional)
    - ref_equity (optional): fallback reference when BASE_UNITS_EQUITY_REF is non-numeric
    """
    if base_units <= 0:
        return base_units
    if equity <= 0:
        return base_units
    if not _strategy_env_bool("BASE_UNITS_EQUITY_SCALE_ENABLED", True, env_prefix=env_prefix):
        return base_units

    raw_ref = _strategy_env_get("BASE_UNITS_EQUITY_REF", "1000000", env_prefix=env_prefix)
    ref_value = None
    if raw_ref is not None and str(raw_ref).strip() != "":
        try:
            ref_value = float(raw_ref)
        except (TypeError, ValueError):
            ref_value = None

    if (ref_value is None or ref_value <= 0) and ref_equity is not None and ref_equity > 0:
        ref_value = float(ref_equity)

    if ref_value is None or ref_value <= 0:
        ref_value = 1000000.0

    if ref_value <= 0:
        return base_units

    scale = equity / ref_value
    scale_min = max(0.0, _strategy_env_float("BASE_UNITS_EQUITY_SCALE_MIN", 1.0, env_prefix=env_prefix))
    scale = max(scale, scale_min)

    scale_max = None
    raw_max = _strategy_env_get("BASE_UNITS_EQUITY_SCALE_MAX", None, env_prefix=env_prefix)
    if raw_max is not None and str(raw_max).strip() != "":
        try:
            scale_max = float(raw_max)
        except (TypeError, ValueError):
            scale_max = None
    if scale_max is not None:
        scale = min(scale, max(scale_min, scale_max))

    scaled = int(round(base_units * scale))
    if min_units is not None:
        scaled = max(int(min_units), scaled)
    if max_units is not None and max_units > 0:
        scaled = min(int(max_units), scaled)
    return scaled
