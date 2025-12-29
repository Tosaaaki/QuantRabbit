"""
Common gating utilities shared by worker processes.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from analysis.regime_classifier import classify
from indicators.factor_cache import all_factors

_REGIME_CACHE_TTL_SEC = 5.0

_regime_cache: dict[str, tuple[float, Optional[str]]] = {}

LOG = logging.getLogger(__name__)


def current_regime(tf: str = "M1", *, event_mode: bool = False) -> Optional[str]:
    """
    Retrieve the latest technical regime label for the given timeframe.
    """
    tf_key = tf.upper()
    cache_key = f"{tf_key}:{event_mode}"
    now = time.monotonic()
    cached_ts, cached_value = _regime_cache.get(cache_key, (0.0, None))
    if now - cached_ts <= _REGIME_CACHE_TTL_SEC:
        return cached_value

    try:
        fac = all_factors().get(tf_key)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("[GATE] factor lookup failed: %s", exc)
        fac = None

    regime = None
    if fac:
        try:
            regime = classify(fac, tf_key, event_mode=event_mode)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("[GATE] regime classification failed: %s", exc)
            regime = None

    _regime_cache[cache_key] = (now, regime)
    return regime
